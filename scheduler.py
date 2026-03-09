"""
Request Scheduler / Router — Part 1 of the Multi-LoRA Inference Simulation.

Purpose: Accept incoming inference requests, group them into batches by adapter,
and dispatch them to the AdapterManager in an order that minimises VRAM swaps.

Design goals
------------
- Swap-aware batching: requests that share an adapter are coalesced so the
  adapter is loaded once and used N times before being evicted.
- Priority support: each request carries an integer priority (higher = sooner).
- Prefetch hints: after dispatching a batch the scheduler looks ahead at the
  next batch and calls AdapterManager.ensure_in_vram() so the GPU transfer
  overlaps with the current batch's compute (useful when non_blocking=True is
  later enabled).
- Back-pressure: a configurable max_queue_depth prevents unbounded memory
  growth if the consumer falls behind.
- Observable: rich metrics (queued, dispatched, swaps avoided, etc.) exposed
  via a SchedulerStats dataclass.

Typical call pattern
--------------------
    scheduler = RequestScheduler(adapter_manager, vram_capacity=4)
    scheduler.submit(InferenceRequest("req-1", adapter_id="lora-A", payload=...))
    batch = scheduler.next_batch(max_batch_size=8)
    for req in batch:
        tensor, _ = adapter_manager.get_or_create_adapter(req.adapter_id)
        # ... run inference ...
        scheduler.mark_complete(req.request_id)
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from adapter_manager import AdapterManager


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(order=False)
class InferenceRequest:
    """A single inference request bound to one LoRA adapter."""

    request_id: str
    adapter_id: str
    payload: Any = None                  # opaque; real impl: token ids, etc.
    priority: int = 0                    # higher value = higher priority
    arrival_time: float = field(default_factory=time.monotonic)

    # Internal: tie-break counter so the heap never compares payloads.
    _seq: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Validate types early so errors surface at submission time.
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string.")
        if not isinstance(self.adapter_id, str) or not self.adapter_id:
            raise ValueError("adapter_id must be a non-empty string.")
        if not isinstance(self.priority, int):
            raise TypeError("priority must be an int.")

    # Heap ordering: we want a max-heap on priority, min-heap on arrival_time
    # as a tie-breaker.  Python's heapq is a min-heap, so we negate priority.
    def _heap_key(self) -> tuple[int, float, int]:
        return (-self.priority, self.arrival_time, self._seq)


@dataclass
class SchedulerStats:
    total_submitted: int = 0
    total_dispatched: int = 0
    total_completed: int = 0
    total_rejected: int = 0       # dropped due to queue depth limit
    batches_dispatched: int = 0
    swap_avoiding_batches: int = 0  # batches where top adapter was already in VRAM
    prefetch_calls: int = 0


# ---------------------------------------------------------------------------
# Internal heap entry (avoids comparing InferenceRequest objects directly)
# ---------------------------------------------------------------------------

class _HeapEntry:
    """Thin wrapper that gives heapq a stable comparison key."""

    __slots__ = ("key", "request", "valid")

    def __init__(self, request: InferenceRequest, seq: int) -> None:
        request._seq = seq
        self.key = request._heap_key()
        self.request = request
        self.valid = True  # set False when request is cancelled / popped logically

    def __lt__(self, other: "_HeapEntry") -> bool:
        return self.key < other.key


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class RequestScheduler:
    """
    Swap-minimising request scheduler for multi-LoRA inference.

    The scheduler maintains a priority queue of pending requests. When
    ``next_batch`` is called it applies the following selection strategy:

    1. **VRAM-first selection**: if any pending adapter is already resident in
       VRAM, prefer requests for those adapters (sorted by priority/arrival).
    2. **Largest-group fallback**: if no pending request maps to a VRAM-resident
       adapter, pick the adapter with the most queued requests (minimises total
       swaps) and take up to ``max_batch_size`` of them in priority order.
    3. **Prefetch**: after selecting the current batch, peek at what the *next*
       batch adapter would be and call ``ensure_in_vram`` on it.
    """

    def __init__(
        self,
        adapter_manager: "AdapterManager",
        max_queue_depth: int = 10_000,
        prefetch: bool = True,
    ) -> None:
        """
        Args:
            adapter_manager: The Part-2 AdapterManager instance that owns VRAM.
            max_queue_depth: Hard cap on total pending requests. Submissions
                beyond this limit are rejected (stats.total_rejected incremented).
            prefetch: Whether to call ensure_in_vram for the predicted next
                adapter after each batch is formed.
        """
        self._am = adapter_manager
        self.max_queue_depth = max_queue_depth
        self.prefetch = prefetch

        # Global min-heap of _HeapEntry objects.
        self._heap: list[_HeapEntry] = []

        # adapter_id -> list[_HeapEntry] for O(1) VRAM-resident lookup and
        # for largest-group selection.
        self._by_adapter: defaultdict[str, list[_HeapEntry]] = defaultdict(list)

        # Requests currently being processed (dispatched but not yet complete).
        self._in_flight: dict[str, InferenceRequest] = {}

        # Monotonic sequence counter for stable heap ordering.
        self._seq = itertools.count()

        self.stats = SchedulerStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, request: InferenceRequest) -> bool:
        """
        Enqueue an inference request.

        Returns:
            True if accepted, False if rejected due to queue depth limit.
        """
        if self.queue_depth() >= self.max_queue_depth:
            self.stats.total_rejected += 1
            return False

        entry = _HeapEntry(request, next(self._seq))
        heapq.heappush(self._heap, entry)
        self._by_adapter[request.adapter_id].append(entry)
        self.stats.total_submitted += 1
        return True

    def submit_many(self, requests: list[InferenceRequest]) -> int:
        """Bulk submit. Returns number of accepted requests."""
        return sum(self.submit(r) for r in requests)

    def next_batch(self, max_batch_size: int = 8) -> list[InferenceRequest]:
        """
        Select and return the next batch of requests to process.

        The batch is *not* automatically marked complete — callers must call
        ``mark_complete(request_id)`` after each request finishes.

        Returns an empty list if no requests are pending.
        """
        if not self._heap:
            return []

        # --- Step 1: choose the adapter for this batch ---
        chosen_adapter = self._choose_adapter(max_batch_size)
        if chosen_adapter is None:
            return []

        # --- Step 2: drain up to max_batch_size valid entries for that adapter ---
        pool = self._by_adapter[chosen_adapter]
        batch: list[InferenceRequest] = []

        # Sort pool by heap key so we respect priority / arrival order.
        pool.sort(key=lambda e: e.key)

        kept: list[_HeapEntry] = []
        for entry in pool:
            if not entry.valid:
                continue
            if len(batch) < max_batch_size:
                entry.valid = False   # logically remove from heap & pool
                batch.append(entry.request)
                self._in_flight[entry.request.request_id] = entry.request
            else:
                kept.append(entry)

        self._by_adapter[chosen_adapter] = kept
        if not kept:
            del self._by_adapter[chosen_adapter]

        # Purge stale entries from the global heap lazily (they are invalid).
        # We do a bounded clean rather than a full heapify to keep O(log n).
        self._purge_heap_top()

        # --- Step 3: update stats ---
        self.stats.batches_dispatched += 1
        self.stats.total_dispatched += len(batch)
        if self._am.is_in_vram(chosen_adapter):
            self.stats.swap_avoiding_batches += 1

        # --- Step 4: prefetch next adapter ---
        if self.prefetch:
            # ! TODO: this should be modified to be non-blocking, and we should add a lock in adapter manager
            next_adapter = self._predict_next_adapter(exclude=chosen_adapter)
            if next_adapter:
                self._am.ensure_adapter_in_vram(next_adapter)
                self.stats.prefetch_calls += 1

        return batch

    def mark_complete(self, request_id: str) -> bool:
        """
        Signal that a dispatched request has finished processing.

        Returns True if the request was found in-flight, False otherwise.
        """
        req = self._in_flight.pop(request_id, None)
        if req is None:
            return False
        self.stats.total_completed += 1
        return True

    def cancel(self, request_id: str) -> bool:
        """
        Cancel a pending (not yet dispatched) request.

        The entry is lazily removed from the heap on the next pop.
        Returns True if found and cancelled, False if not found or already
        dispatched.
        """
        for adapter_entries in self._by_adapter.values():
            for entry in adapter_entries:
                if entry.valid and entry.request.request_id == request_id:
                    entry.valid = False
                    return True
        return False

    def queue_depth(self) -> int:
        """Number of pending (not yet dispatched) requests."""
        return sum(
            1 for entries in self._by_adapter.values()
            for e in entries if e.valid
        )

    def in_flight_count(self) -> int:
        """Number of dispatched but not yet completed requests."""
        return len(self._in_flight)

    def pending_adapters(self) -> list[str]:
        """Adapter IDs that have at least one pending request."""
        return list(self._by_adapter.keys())

    def reset_stats(self) -> None:
        """Reset observable metrics without affecting queue state."""
        self.stats = SchedulerStats()

    def __repr__(self) -> str:
        return (
            f"RequestScheduler(queued={self.queue_depth()}, "
            f"in_flight={self.in_flight_count()}, "
            f"dispatched={self.stats.total_dispatched}, "
            f"batches={self.stats.batches_dispatched})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _choose_adapter(self, max_batch_size: int) -> Optional[str]:
        """
        Pick which adapter's requests to serve next.

        Priority order:
        1. Adapters already in VRAM (swap-free), ranked by highest-priority
           pending request.
        2. Adapter with the largest pending queue (minimises future swaps),
           ranked by highest-priority pending request as a tie-breaker.
        """
        # Filter to adapters with at least one valid pending request.
        candidates: list[tuple[str, _HeapEntry]] = []
        for adapter_id, entries in self._by_adapter.items():
            best = self._best_entry(entries)
            if best is not None:
                candidates.append((adapter_id, best))

        if not candidates:
            return None

        # Partition into VRAM-resident and cold.
        vram_candidates = [
            (aid, e) for aid, e in candidates if self._am.is_in_vram(aid)
        ]

        if vram_candidates:
            # Among VRAM-resident adapters pick the one with the highest-
            # priority pending request (lowest heap key).
            vram_candidates.sort(key=lambda x: x[1].key)
            return vram_candidates[0][0]

        # No VRAM-resident candidates: pick adapter with most pending requests,
        # breaking ties by highest-priority pending request.
        def _fallback_key(item: tuple[str, _HeapEntry]) -> tuple[int, tuple]:
            aid, best_entry = item
            count = sum(1 for e in self._by_adapter[aid] if e.valid)
            return (-count, best_entry.key)   # negate count for max-first sort

        candidates.sort(key=_fallback_key)
        return candidates[0][0]

    def _best_entry(self, entries: list[_HeapEntry]) -> Optional[_HeapEntry]:
        """Return the highest-priority valid entry from a list, or None."""
        best: Optional[_HeapEntry] = None
        for e in entries:
            if not e.valid:
                continue
            if best is None or e.key < best.key:
                best = e
        return best

    def _predict_next_adapter(self, exclude: str) -> Optional[str]:
        """Peek at which adapter would be chosen for the *next* batch."""
        candidates: list[tuple[str, _HeapEntry]] = []
        for adapter_id, entries in self._by_adapter.items():
            if adapter_id == exclude:
                continue
            best = self._best_entry(entries)
            if best is not None:
                candidates.append((adapter_id, best))

        if not candidates:
            return None

        vram_candidates = [
            (aid, e) for aid, e in candidates if self._am.is_in_vram(aid)
        ]
        if vram_candidates:
            vram_candidates.sort(key=lambda x: x[1].key)
            return vram_candidates[0][0]

        def _fallback_key(item: tuple[str, _HeapEntry]) -> tuple[int, tuple]:
            aid, best_entry = item
            count = sum(1 for e in self._by_adapter[aid] if e.valid)
            return (-count, best_entry.key)

        candidates.sort(key=_fallback_key)
        return candidates[0][0]

    def _purge_heap_top(self, limit: int = 32) -> None:
        """
        Lazily remove invalid entries from the top of the heap.
        Bounded by ``limit`` pops to avoid O(n) work per batch.
        """
        purged = 0
        while self._heap and not self._heap[0].valid and purged < limit:
            heapq.heappop(self._heap)
            purged += 1