"""
pipeline/__init__.py

Public API for the pipeline package.

This package provides the full causal-intervention framework:

    Data structures
    ---------------
    Location / LOC      Describe which slice of a tensor to patch.
    GraphInput          Immutable dict mapping leaf-node names to input values.
    Intervention        Pair a base input with node patches (and optional slices).

    Graph primitives
    ----------------
    GraphNode           One node in a computation graph (wraps a forward fn).
    ComputationGraph    DAG of GraphNodes; runs base and patched forward passes.

    Automation
    ----------
    CompGraphConstructor  Build a ComputationGraph from a torch.nn.Module
                          automatically via a sample forward pass.

Quick-start
-----------
    from pipeline import ComputationGraph, GraphNode, GraphInput, Intervention, LOC

    class MyGraph(ComputationGraph):
        def __init__(self):
            @GraphNode()
            def leaf(x):
                return x * 2

            @GraphNode(leaf)
            def root(val):
                return val + 1

            super().__init__(root)

    g   = MyGraph()
    inp = GraphInput({"leaf": 5})
    print(g.compute(inp))                          # 11

    interv = Intervention(inp, {"root": 99})
    base, patched = g.intervene(interv)
    print(base, patched)                           # 11  99
"""

from intervention.location import Location, LOC
from intervention.graph_input import GraphInput
from intervention.intervention import Intervention
from intervention.graph_node import GraphNode
from intervention.graph import ComputationGraph
from intervention.constructor import CompGraphConstructor

__all__ = [
    # Location helpers
    "Location",
    "LOC",
    # Core data structures
    "GraphInput",
    "Intervention",
    # Graph primitives
    "GraphNode",
    "ComputationGraph",
    # Automation
    "CompGraphConstructor",
]


# ---------------------------------------------------------------------------
# Smoke test — run with:  python -m pipeline
# ---------------------------------------------------------------------------
def _run_tests():
    # ── 1. All public symbols are importable ──────────────────────────────
    from pipeline import (
        Location, LOC,
        GraphInput, Intervention,
        GraphNode, ComputationGraph,
        CompGraphConstructor,
    )
    print("  all imports ✓")

    # ── 2. LOC is a Location instance ─────────────────────────────────────
    assert isinstance(LOC, Location)
    assert LOC[5:10] == slice(5, 10)
    print("  LOC singleton ✓")

    # ── 3. End-to-end: build graph, run base + intervention ───────────────
    class SignGraph(ComputationGraph):
        def __init__(self):
            @GraphNode()
            def value(x):
                return x

            @GraphNode(value)
            def sign(v):
                return 1 if v > 0 else (-1 if v < 0 else 0)

            super().__init__(sign)

    g   = SignGraph()
    inp = GraphInput({"value": 3})

    assert g.compute(inp) == 1                        # positive
    g.clear_caches()

    interv = Intervention(inp, {"value": -7})
    base, patched = g.intervene(interv)
    assert base    ==  1                              # original: positive
    assert patched == -1                              # forced negative
    print("  end-to-end compute + intervene ✓")

    # ── 4. __all__ contains every exported symbol ─────────────────────────
    import pipeline
    for name in __all__:
        assert hasattr(pipeline, name), f"Missing from package: {name}"
    print("  __all__ complete ✓")

    print("\n✓  All pipeline.__init__ tests passed")


if __name__ == "__main__":
    _run_tests()
