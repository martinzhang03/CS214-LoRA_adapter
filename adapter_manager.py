"""
Adapter Manager (VRAM/RAM Control) — Part 2 of the Multi-LoRA Inference Simulation.

Purpose: Manage the "VRAM Shelf" using LRU (Least Recently Used) logic.
- Host RAM: large dictionary of CPU tensors (unbounded).
- GPU VRAM: fixed-size pool of GPU tensors (strict O(1) memory growth).

When an adapter is requested:
  - If in GPU pool → mark as recently used (move to MRU), return.
  - If not in GPU → evict coldest adapter from VRAM to Host RAM, then load
    requested adapter from Host RAM (or create if new) into VRAM.
"""

from __future__ import annotations

import collections
from typing import Optional

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class AdapterManager:
    """
    Two-tier memory manager: Host RAM (CPU) + GPU VRAM (fixed-size LRU pool).
    Enforces O(1) VRAM usage regardless of total number of adapters.
    """

    def __init__(
        self,
        vram_capacity: int,
        adapter_tensor_size: int = 1024,
        device: Optional[str] = None,
    ):
        """
        Args:
            vram_capacity: Maximum number of adapters allowed in the GPU pool.
            adapter_tensor_size: Number of elements in each adapter's placeholder
                tensor (for simulation; real impl would use actual LoRA weights).
            device: PyTorch device for "GPU" (e.g. 'cuda:0'). If None, uses
                'cuda' if available else 'cpu' (simulated VRAM still has fixed size).
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("AdapterManager requires PyTorch (pip install torch).")

        self.vram_capacity = vram_capacity
        self.adapter_tensor_size = adapter_tensor_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_device = torch.device(device)
        self.cpu_device = torch.device("cpu")

        # Host RAM: adapter_id -> CPU tensor (unbounded)
        self._host_ram: dict[str, torch.Tensor] = {}

        # GPU VRAM: OrderedDict preserves order; first = LRU, last = MRU.
        # We use move_to_end(key) on access to implement LRU.
        self._vram_pool: collections.OrderedDict[str, torch.Tensor] = (
            collections.OrderedDict()
        )

        # Metrics
        self._swap_count = 0   # number of evictions (VRAM -> RAM)
        self._load_count = 0  # number of loads (RAM -> VRAM or new -> VRAM)

    def get_adapter(self, adapter_id: str) -> tuple[torch.Tensor, bool]:
        """
        Ensure the adapter is in VRAM and return its tensor. Uses LRU eviction
        when VRAM is full and the adapter is not present.

        Returns:
            (tensor, was_swap): The adapter's GPU tensor, and whether an eviction
            (swap out + load) was performed (useful for latency/throughput metrics).
        """
        # Already in VRAM → move to MRU (end) and return
        if adapter_id in self._vram_pool:
            self._vram_pool.move_to_end(adapter_id)
            return self._vram_pool[adapter_id], False

        # Need to load into VRAM (from RAM or create new)
        was_swap = False
        while len(self._vram_pool) >= self.vram_capacity:
            # Evict LRU (first item)
            evicted_id, evicted_tensor = self._vram_pool.popitem(last=False)
            evicted_cpu = evicted_tensor.to(self.cpu_device, non_blocking=False)
            self._host_ram[evicted_id] = evicted_cpu
            self._swap_count += 1
            was_swap = True

        # Get or create adapter tensor
        if adapter_id in self._host_ram:
            tensor = self._host_ram.pop(adapter_id).to(
                self.gpu_device, non_blocking=False
            )
        else:
            # New adapter: create placeholder on CPU then move to GPU
            tensor = torch.zeros(
                self.adapter_tensor_size,
                dtype=torch.float32,
                device=self.gpu_device,
            )
        self._load_count += 1
        self._vram_pool[adapter_id] = tensor
        return tensor, was_swap

    def ensure_in_vram(self, adapter_id: str) -> bool:
        """
        Prefetch: ensure adapter is in VRAM without returning the tensor.
        Returns True if a swap (eviction + load) occurred.
        """
        _, was_swap = self.get_adapter(adapter_id)
        return was_swap

    def create_adapter_in_ram(self, adapter_id: str) -> None:
        """
        Register a new adapter in Host RAM only (e.g. when loading from disk).
        If adapter_id already exists in RAM or VRAM, this is a no-op.
        """
        if adapter_id in self._vram_pool or adapter_id in self._host_ram:
            return
        tensor = torch.zeros(
            self.adapter_tensor_size,
            dtype=torch.float32,
            device=self.cpu_device,
        )
        self._host_ram[adapter_id] = tensor

    def vram_count(self) -> int:
        """Current number of adapters in the GPU pool."""
        return len(self._vram_pool)

    def ram_count(self) -> int:
        """Current number of adapters in Host RAM."""
        return len(self._host_ram)

    def swap_count(self) -> int:
        """Total number of evictions (VRAM → RAM) so far."""
        return self._swap_count

    def load_count(self) -> int:
        """Total number of loads into VRAM (from RAM or new) so far."""
        return self._load_count

    def is_in_vram(self, adapter_id: str) -> bool:
        """Return True if the adapter is currently in the GPU pool."""
        return adapter_id in self._vram_pool

    def vram_ids(self) -> list[str]:
        """Return adapter IDs currently in VRAM, in LRU order (first = coldest)."""
        return list(self._vram_pool.keys())

    def reset_metrics(self) -> None:
        """Reset swap and load counters (state is unchanged)."""
        self._swap_count = 0
        self._load_count = 0

    def __repr__(self) -> str:
        return (
            f"AdapterManager(vram_capacity={self.vram_capacity}, "
            f"vram_count={self.vram_count()}, ram_count={self.ram_count()}, "
            f"swaps={self._swap_count}, loads={self._load_count})"
        )
