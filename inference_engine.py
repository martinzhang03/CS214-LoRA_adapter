"""
Inference Engine — Part 3 of the Multi-LoRA Inference Simulation.

Purpose: Bridge the RequestScheduler (Part 1), AdapterManager (Part 2), and
BaseModelManager (Part 4) into a single, runnable inference pipeline.

Responsibilities
----------------
- Drive the batch loop: call scheduler.next_batch(), load the adapter via
  AdapterManager, run BaseModelManager.merge_and_forward(), collect results.
- Produce InferenceResult objects (one per request) with output tensors,
  latency measurements, and swap metadata.
- Expose both a synchronous step() for single-batch execution and a run()
  method that drains the queue completely.
- Aggregate per-run EngineStats: throughput, mean/p99 latency, swap rate.

Design notes
------------
- The engine owns no GPU state itself — that lives in AdapterManager and
  BaseModelManager.  The engine is purely a coordinator.
- Input tensors: real systems would derive these from tokenised payloads.
  Here, if request.payload is already a torch.Tensor we use it directly;
  otherwise we create a zero tensor so the simulation still runs end-to-end.
- Thread-safety: step() is intentionally synchronous and single-threaded.
  The async wrapper (Part 6) will call step() from an executor thread.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from adapter_manager import AdapterManager
    from base_model_manager import BaseModelManager
    from scheduler import RequestScheduler, InferenceRequest


# ---------------------------------------------------------------------------
# Result + Stats
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Output produced for a single InferenceRequest."""

    request_id: str
    adapter_id: str
    output: Any                        # torch.Tensor in normal operation
    latency_s: float                   # wall-clock seconds for this request
    was_swap: bool                     # True if an eviction occurred for its batch
    error: Optional[str] = None        # non-None if inference raised an exception


@dataclass
class EngineStats:
    total_requests: int = 0
    total_batches: int = 0
    total_swaps: int = 0               # batches that caused at least one eviction
    total_errors: int = 0
    total_wall_s: float = 0.0          # cumulative wall time across all step() calls

    # Latency tracking (per-request seconds)
    _latencies: list[float] = field(default_factory=list, repr=False)

    def record_latency(self, latency_s: float) -> None:
        self._latencies.append(latency_s)

    @property
    def mean_latency_s(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def p99_latency_s(self) -> float:
        if not self._latencies:
            return 0.0
        sorted_lat = sorted(self._latencies)
        idx = max(0, int(len(sorted_lat) * 0.99) - 1)
        return sorted_lat[idx]

    @property
    def throughput_rps(self) -> float:
        """Requests per second over total wall time."""
        if self.total_wall_s == 0:
            return 0.0
        return self.total_requests / self.total_wall_s

    @property
    def swap_rate(self) -> float:
        """Fraction of batches that triggered a swap."""
        if self.total_batches == 0:
            return 0.0
        return self.total_swaps / self.total_batches


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Coordinates the full inference pipeline for one step (batch) at a time.

    Typical synchronous usage
    -------------------------
        engine = InferenceEngine(scheduler, adapter_manager, base_model_manager)
        while scheduler.queue_depth() > 0 or scheduler.in_flight_count() > 0:
            results = engine.step(max_batch_size=8)
            for r in results:
                print(r.request_id, r.latency_s)

    Or use the convenience wrapper:
        all_results = engine.run(max_batch_size=8)
    """

    def __init__(
        self,
        scheduler: "RequestScheduler",
        adapter_manager: "AdapterManager",
        base_model_manager: "BaseModelManager",
        max_batch_size: int = 8,
    ) -> None:
        """
        Args:
            scheduler:           Part 1 — supplies ordered batches of requests.
            adapter_manager:     Part 2 — manages LoRA tensors in VRAM/RAM.
            base_model_manager:  Part 4 — owns the frozen base model on GPU.
            max_batch_size:      Default batch size passed to scheduler.next_batch().
                                 Can be overridden per step() call.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("InferenceEngine requires PyTorch.")

        self._sched = scheduler
        self._am = adapter_manager
        self._bmm = base_model_manager
        self.max_batch_size = max_batch_size
        self.stats = EngineStats()

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(
        self,
        max_batch_size: Optional[int] = None,
    ) -> list[InferenceResult]:
        """
        Execute one batch of inference requests.

        Steps
        -----
        1. Ask the scheduler for the next batch (swap-minimised order).
        2. Load the adapter for this batch into VRAM (one get_adapter call;
           all requests in a batch share the same adapter_id by design).
        3. Build input tensors from request payloads.
        4. Call BaseModelManager.merge_and_forward() once per request.
        5. Mark each request complete in the scheduler.
        6. Return InferenceResult objects with outputs and timing.

        Returns:
            List of InferenceResult, one per request in the batch.
            Empty list if the queue is empty.
        """
        bsz = max_batch_size if max_batch_size is not None else self.max_batch_size
        batch: list["InferenceRequest"] = self._sched.next_batch(bsz)

        if not batch:
            return []

        step_start = time.monotonic()
        results: list[InferenceResult] = []

        # All requests in a batch share the same adapter_id (scheduler guarantees this).
        adapter_id = batch[0].adapter_id

        # --- Load adapter into VRAM ---
        try:
            # adapter_tensor, was_swap = self._am.get_adapter(adapter_id)
            adapter_tensor, was_swap = self._am.get_or_create_adapter_in_vram(adapter_id)
        except Exception as exc:
            # Adapter load failed — fail the entire batch gracefully.
            error_msg = f"AdapterManager.get_adapter failed: {exc}"
            for req in batch:
                results.append(InferenceResult(
                    request_id=req.request_id,
                    adapter_id=req.adapter_id,
                    output=None,
                    latency_s=0.0,
                    was_swap=False,
                    error=error_msg,
                ))
                self._sched.mark_complete(req.request_id)
                self.stats.total_errors += 1
            self.stats.total_requests += len(batch)
            self.stats.total_batches += 1
            return results

        if was_swap:
            self.stats.total_swaps += 1

        # --- Run inference per request ---
        for req in batch:
            req_start = time.monotonic()
            output = None
            error = None

            try:
                x = self._build_input_tensor(req)
                output = self._bmm.merge_and_forward(x, adapter_tensor)
            except Exception as exc:
                error = str(exc)
                self.stats.total_errors += 1

            latency_s = time.monotonic() - req_start
            self.stats.record_latency(latency_s)
            self._sched.mark_complete(req.request_id)

            results.append(InferenceResult(
                request_id=req.request_id,
                adapter_id=adapter_id,
                output=output,
                latency_s=latency_s,
                was_swap=was_swap,
                error=error,
            ))

        # --- Update aggregate stats ---
        self.stats.total_requests += len(batch)
        self.stats.total_batches += 1
        self.stats.total_wall_s += time.monotonic() - step_start

        return results

    # ------------------------------------------------------------------
    # Convenience: drain entire queue
    # ------------------------------------------------------------------

    def run(
        self,
        max_batch_size: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> list[InferenceResult]:
        """
        Drain the scheduler queue completely, running step() in a loop.

        Args:
            max_batch_size: Passed to each step() call.
            max_steps: Safety cap — stop after this many batches even if
                       the queue is not empty.  None means run until empty.

        Returns:
            All InferenceResult objects collected across all steps.
        """
        all_results: list[InferenceResult] = []
        steps = 0

        while True:
            if max_steps is not None and steps >= max_steps:
                break
            if self._sched.queue_depth() == 0 and self._sched.in_flight_count() == 0:
                break

            batch_results = self.step(max_batch_size)
            all_results.extend(batch_results)
            steps += 1

            # Guard against infinite loop if in-flight requests are never
            # completed externally (shouldn't happen in sync mode, but be safe).
            if not batch_results and self._sched.queue_depth() == 0:
                break

        return all_results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_input_tensor(self, req: "InferenceRequest") -> "torch.Tensor":
        """
        Extract or synthesise a (1, hidden_dim) input tensor from a request.

        - If payload is already a torch.Tensor, use it directly (moving to
          the correct device if needed).
        - If payload is a list/tuple of floats, convert to tensor.
        - Otherwise (None or opaque), create a zero tensor as a placeholder.
        """
        d = self._bmm.hidden_dim
        device = self._bmm.device

        if isinstance(req.payload, torch.Tensor):
            x = req.payload.float()
            if x.dim() == 1:
                x = x.unsqueeze(0)          # (d,) -> (1, d)
            return x.to(device)

        if isinstance(req.payload, (list, tuple)):
            x = torch.tensor(req.payload, dtype=torch.float32, device=device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return x

        # Fallback: zero tensor
        return torch.zeros(1, d, dtype=torch.float32, device=device)

    def reset_stats(self) -> None:
        """Reset engine stats without affecting queue or model state."""
        self.stats = EngineStats()

    def __repr__(self) -> str:
        return (
            f"InferenceEngine("
            f"requests={self.stats.total_requests}, "
            f"batches={self.stats.total_batches}, "
            f"mean_latency={self.stats.mean_latency_s*1000:.2f}ms, "
            f"swap_rate={self.stats.swap_rate:.2%}, "
            f"throughput={self.stats.throughput_rps:.1f} rps)"
        )