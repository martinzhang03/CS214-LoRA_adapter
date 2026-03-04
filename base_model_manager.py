"""
Base Model Manager — Part 4 of the Multi-LoRA Inference Simulation.

Purpose: Own and expose the frozen base model that never gets evicted from VRAM.
Unlike LoRA adapters (which are swapped in/out by AdapterManager), the base
model is loaded once and lives in VRAM for the lifetime of the process.

Responsibilities
----------------
- Load a base model (real nn.Module or a simulated placeholder) onto the GPU.
- Expose a forward() method that accepts an input tensor + a merged LoRA delta
  and returns an output tensor.
- Track whether the model is loaded and on which device.
- Provide a merge_and_forward() convenience that handles the LoRA math:
      output = (W_base + alpha * delta_W) @ x
  where delta_W comes from the AdapterManager's GPU tensor.
- Expose basic stats (device, param count, load time).

Design notes
------------
- The base model is intentionally kept separate from AdapterManager so that
  the two concerns — "what's on GPU permanently" vs "what's swapped in/out" —
  have clear ownership.
- In a real system W_base would be a HuggingFace / vLLM model; here we use a
  configurable linear layer as a placeholder so the simulation runs without
  model weights.
- Thread-safety: forward() acquires a lock so concurrent callers (async loop
  in Part 6) don't interleave weight merges.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class BaseModelStats:
    device: str = "unloaded"
    param_count: int = 0
    load_time_s: float = 0.0
    forward_calls: int = 0
    merged_forward_calls: int = 0   # calls that included a LoRA delta


# ---------------------------------------------------------------------------
# Simulated base model (placeholder nn.Module)
# ---------------------------------------------------------------------------

class _SimulatedBaseModel(nn.Module):
    """
    Minimal linear model used when no real checkpoint is provided.
    Acts as W_base: output = x @ W^T + b
    Shape: (hidden_dim, hidden_dim) weight matrix.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear(x)


# ---------------------------------------------------------------------------
# BaseModelManager
# ---------------------------------------------------------------------------

class BaseModelManager:
    """
    Owns the frozen base model in VRAM.

    The base model is never evicted. LoRA adapters are merged transiently
    during forward() and then discarded — the base weights are never mutated.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        device: Optional[str] = None,
        model: Optional["nn.Module"] = None,
    ) -> None:
        """
        Args:
            hidden_dim: Dimensionality of the base model's hidden layer
                (used when creating the simulated placeholder).
            lora_rank: Rank r of the LoRA decomposition.  The adapter tensor
                from AdapterManager has shape (r * hidden_dim,) which we
                reshape to (r, hidden_dim) inside merge_and_forward().
            lora_alpha: LoRA scaling factor; effective scale = alpha / rank.
            device: Target device string ('cuda', 'cuda:0', 'cpu', …).
                Defaults to 'cuda' if available, else 'cpu'.
            model: Optional pre-built nn.Module to use instead of the
                simulated placeholder.  Must accept a (batch, hidden_dim)
                tensor and return a (batch, hidden_dim) tensor.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("BaseModelManager requires PyTorch.")

        self.hidden_dim = hidden_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scale: float = lora_alpha / lora_rank

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self._lock = threading.Lock()
        self._model: Optional[nn.Module] = None
        self.stats = BaseModelStats()

        # Load immediately if a model is provided; otherwise defer to load().
        if model is not None:
            self._load_model(model)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, model: Optional["nn.Module"] = None) -> None:
        """
        Load the base model onto the GPU.  Safe to call multiple times;
        subsequent calls replace the current model.

        Args:
            model: An nn.Module to load.  If None, creates the simulated
                placeholder using self.hidden_dim.
        """
        if model is None:
            model = _SimulatedBaseModel(self.hidden_dim)
        self._load_model(model)

    def _load_model(self, model: "nn.Module") -> None:
        t0 = time.monotonic()
        model = model.to(self.device)
        model.eval()

        # Freeze all parameters — base weights are never updated at runtime.
        for param in model.parameters():
            param.requires_grad_(False)

        self._model = model
        elapsed = time.monotonic() - t0

        param_count = sum(p.numel() for p in model.parameters())
        self.stats = BaseModelStats(
            device=str(self.device),
            param_count=param_count,
            load_time_s=elapsed,
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "Base model not loaded. Call BaseModelManager.load() first."
            )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Plain forward pass through the base model (no LoRA).

        Args:
            x: Input tensor of shape (batch_size, hidden_dim).

        Returns:
            Output tensor of shape (batch_size, hidden_dim).
        """
        self._require_loaded()
        with self._lock:
            self.stats.forward_calls += 1
            with torch.no_grad():
                return self._model(x)  # type: ignore[misc]

    def merge_and_forward(
        self,
        x: torch.Tensor,
        adapter_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with a transiently merged LoRA adapter.

        The base weights are NOT mutated.  Instead we compute:

            h      = W_base @ x          (base model hidden state)
            delta  = (B @ A) @ x         (LoRA correction)
            output = h + scale * delta

        where A = adapter_tensor[:r*d].reshape(r, d)  (down-projection)
              B = adapter_tensor[r*d:].reshape(d, r)  (up-projection)
              scale = lora_alpha / lora_rank

        The adapter_tensor from AdapterManager has shape
        (adapter_tensor_size,) = (2 * lora_rank * hidden_dim,).

        Args:
            x: Input tensor of shape (batch_size, hidden_dim).
            adapter_tensor: Flat GPU tensor from AdapterManager.get_adapter().

        Returns:
            Output tensor of shape (batch_size, hidden_dim).
        """
        self._require_loaded()

        r, d = self.lora_rank, self.hidden_dim
        expected = 2 * r * d
        if adapter_tensor.numel() != expected:
            raise ValueError(
                f"adapter_tensor has {adapter_tensor.numel()} elements; "
                f"expected {expected} (2 * lora_rank={r} * hidden_dim={d})."
            )

        with self._lock:
            self.stats.forward_calls += 1
            self.stats.merged_forward_calls += 1

            with torch.no_grad():
                # Base model output
                base_out = self._model(x)  # type: ignore[misc]

                # Unpack A (down) and B (up) from the flat adapter tensor
                flat = adapter_tensor.to(x.device)
                A = flat[: r * d].reshape(r, d)   # (r, d)
                B = flat[r * d :].reshape(d, r)   # (d, r)

                # LoRA correction: x (b,d) -> A (r,d)^T -> (b,r) -> B (d,r)^T -> (b,d)
                lora_out = (x @ A.t()) @ B.t()    # (batch, d)
                return base_out + self.lora_scale * lora_out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def param_count(self) -> int:
        """Total number of parameters in the base model."""
        return self.stats.param_count

    def __repr__(self) -> str:
        loaded = "loaded" if self.is_loaded() else "unloaded"
        return (
            f"BaseModelManager({loaded}, device={self.stats.device}, "
            f"params={self.stats.param_count:,}, "
            f"forward_calls={self.stats.forward_calls})"
        )