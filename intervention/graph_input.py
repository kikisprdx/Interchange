"""
pipeline/graph_input.py

Defines GraphInput — an immutable, hashable wrapper around a dict that maps
leaf node names to their input values.

GraphInput is the entry point for every forward pass through a ComputationGraph.
Because graph nodes cache their results keyed by the input object, GraphInput
objects must be effectively immutable (no in-place mutations after construction).

    from intervention.graph_input import GraphInput

    # Single input
    gi = GraphInput({"x": 3, "y": 5})

    # Batched input (every value is a sequence of N examples)
    gi = GraphInput.make_batched(
        {"x": tensor_of_shape_N_D},
        keys=["example_0", "example_1", ...],
    )
"""

import torch
from typing import Any, Dict, Optional, Sequence


class GraphInput:
    """
    Immutable dict mapping leaf-node names to input values.

    Parameters
    ----------
    values : dict
        Maps each leaf node name (str) to an input value of any type.
    cache_results : bool
        When True, downstream computation nodes cache their outputs
        against this object.
    batched : bool
        When True each value in ``values`` represents a full batch of
        N inputs stacked together.
    batch_dim : int
        The batch dimension index when values are PyTorch tensors.
        Ignored when ``batched=False``.
    keys : sequence, optional
        A unique, hashable key for each example in the batch.
        Required when ``batched=True``.
    device : torch.device, optional
        When provided, every tensor value is moved to ``device``
        at construction time.
    """

    def __init__(
        self,
        values: Dict[str, Any],
        cache_results: bool = True,
        batched: bool = False,
        batch_dim: int = 0,
        keys: Optional[Sequence] = None,
        device: Optional[torch.device] = None,
    ):
        if batched and keys is None:
            raise ValueError(
                "Must provide a `keys` sequence when constructing a batched GraphInput."
            )

        # Optionally move every tensor value to the requested device.
        if device is not None:
            values = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in values.items()
            }

        self._values = values
        self.cache_results = cache_results
        self.batched = batched
        self.batch_dim = batch_dim
        self.keys = keys

    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    def make_batched(
        cls,
        values: Dict[str, Any],
        keys: Sequence,
        cache_results: bool = True,
        batch_dim: int = 0,
        device: Optional[torch.device] = None,
    ) -> "GraphInput":
        """
        Convenience constructor for batched inputs.

        Equivalent to ``GraphInput(values, batched=True, keys=keys, ...)``.
        """
        return cls(
            values,
            cache_results=cache_results,
            batched=True,
            batch_dim=batch_dim,
            keys=keys,
            device=device,
        )

    # ------------------------------------------------------------------
    # Immutable value access
    # ------------------------------------------------------------------

    @property
    def values(self) -> Dict[str, Any]:
        """The underlying node-name to value mapping (read-only)."""
        return self._values

    @values.setter
    def values(self, _):
        raise RuntimeError("GraphInput objects are immutable — values cannot be replaced.")

    # ------------------------------------------------------------------
    # Dict-like interface
    # ------------------------------------------------------------------

    def __getitem__(self, item: str) -> Any:
        return self._values[item]

    def __contains__(self, item: str) -> bool:
        return item in self._values

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        if not self._values:
            return "GraphInput{}"
        entries = ", ".join(f"'{k}': {type(v).__name__}" for k, v in self._values.items())
        suffix = f", batched=True, batch_dim={self.batch_dim}" if self.batched else ""
        return f"GraphInput{{{entries}{suffix}}}"

    # ------------------------------------------------------------------
    # Device movement
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "GraphInput":
        """
        Return a new GraphInput with all tensor values moved to ``device``.

        Non-tensor values are kept as-is.  All metadata (``cache_results``,
        ``batched``, ``batch_dim``, ``keys``) is preserved in the new object.
        The original object is never modified.
        """
        new_values = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self._values.items()
        }
        return GraphInput(
            new_values,
            cache_results=self.cache_results,
            batched=self.batched,
            batch_dim=self.batch_dim,
            keys=self.keys,
        )


# ---------------------------------------------------------------------------
# Tests — run with:  python pipeline/graph_input.py
# ---------------------------------------------------------------------------
def _run_tests():
    # --- Basic construction and access ---
    values = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
    gi = GraphInput(values)

    assert "a" in gi
    assert "b" in gi
    assert len(gi) == 2
    assert torch.equal(gi["a"], torch.tensor([1.0, 2.0]))
    assert gi.batched is False
    assert gi.cache_results is True

    # --- Immutability: assigning to .values raises RuntimeError ---
    try:
        gi.values = {"x": torch.tensor([1.0])}
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "immutable" in str(e).lower()

    # --- to() returns a new object without modifying the original ---
    gi_cpu = gi.to(torch.device("cpu"))
    assert gi_cpu is not gi
    assert torch.equal(gi_cpu["a"], gi["a"])

    # --- to() preserves all metadata ---
    keys = ["k0", "k1"]
    gi_batched = GraphInput.make_batched(
        {"x": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        keys=keys,
        batch_dim=0,
    )
    gi_moved = gi_batched.to(torch.device("cpu"))
    assert gi_moved.batched is True
    assert gi_moved.batch_dim == 0
    assert gi_moved.keys == keys

    # --- make_batched: missing keys raises ValueError ---
    try:
        GraphInput({"x": torch.tensor([1.0])}, batched=True)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    # --- repr sanity check ---
    r = repr(gi)
    assert "GraphInput" in r and "'a'" in r

    # --- CUDA tests (skipped when no GPU available) ---
    if torch.cuda.is_available():
        gi_cuda = GraphInput(values, device=torch.device("cuda"))
        assert all(v.is_cuda for v in gi_cuda.values.values()), \
            "device= constructor param should move tensors to CUDA"

        gi_back = gi_cuda.to(torch.device("cpu"))
        assert all(not v.is_cuda for v in gi_back.values.values()), \
            "to(cpu) should move tensors back to CPU"
    else:
        print("  (CUDA not available — skipping GPU tests)")

    print("✓  All GraphInput tests passed")


if __name__ == "__main__":
    _run_tests()
