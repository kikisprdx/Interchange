"""
pipeline/constructor.py

Defines CompGraphConstructor — automatically builds a ComputationGraph from
a torch.nn.Module by running a single sample forward pass through it.

How it works
------------
The constructor uses PyTorch forward hooks to observe how data flows between
submodules *while* the sample input is being processed:

1.  Each submodule is wrapped with a ``pre_hook`` and ``post_hook``.

2.  ``post_hook`` tags every module's output with the module's own name::

        original_output  →  (original_output, "module_name")

3.  ``pre_hook`` reads those name tags from the inputs it receives,
    so it knows which upstream modules produced each input tensor::

        inputs: [(tensor_a, "layer1"), (tensor_b, "layer2")]
                  ↑ data          ↑ where it came from

    This is how graph edges are discovered at runtime.

4.  An input that carries ``None`` as its tag (i.e. ``(tensor, None)``)
    is an external input — the leaf node.  The leaf's name in the resulting
    ``GraphInput`` is the name of the first submodule that receives it.

Constraints
-----------
*   Every submodule's output must flow **directly** into another named
    submodule.  Intermediate Python operations between module calls are
    not tracked and will break the topology.

*   Only **one** external input (leaf) is supported per graph.  Multiple
    input tensors to the same leaf module are allowed (e.g. a module
    that takes ``(query, key, value)``), but they must all be external —
    mixed external + internal inputs to the same module are not supported.

Typical usage
-------------
    from intervention.constructor import CompGraphConstructor

    model = MyTorchModel()
    sample_input = torch.randn(1, 128)

    graph, graph_input = CompGraphConstructor.construct(model, sample_input)
    # graph_input: GraphInput with the sample input stored under the leaf name
    # graph:       ComputationGraph ready for compute() and intervene()
"""

import torch
from typing import Dict, Iterable, Optional, Tuple, Union

from intervention.graph import ComputationGraph
from intervention.graph_input import GraphInput
from intervention.graph_node import GraphNode


class CompGraphConstructor:
    """
    Build a :class:`~pipeline.graph.ComputationGraph` from a
    ``torch.nn.Module`` using a single sample forward pass.

    See the module docstring for a detailed explanation of the hook mechanism.

    Parameters
    ----------
    module : torch.nn.Module
        The model whose submodules will become graph nodes.
    submodules : dict or iterable of (name, module) pairs, optional
        Explicit list of submodules to treat as graph nodes.
        Defaults to ``module.named_children()``.
    verbose : bool
        When True, print a line each time a hook fires (useful for debugging
        graph construction).  Default False.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        submodules=None,
        verbose: bool = False,
    ):
        if not isinstance(module, torch.nn.Module):
            raise TypeError(
                f"module must be a torch.nn.Module, got {type(module).__name__!r}."
            )

        self.module = module
        self.verbose = verbose

        self._name_to_node: Dict[str, GraphNode] = {}
        self._module_to_name: Dict[torch.nn.Module, str] = {}
        self._current_input: Optional[GraphInput] = None
        self._hook_handles: list = []   # kept so we can remove hooks after construction

        # Resolve submodule iterable
        if submodules is None:
            submodules = module.named_children()
        elif isinstance(submodules, dict):
            submodules = submodules.items()

        for name, submodule in submodules:
            # Create one node per submodule (edges set later via hooks)
            node = GraphNode(name=name, forward=submodule.forward)
            self._name_to_node[name] = node
            self._module_to_name[submodule] = name

            self._hook_handles.append(
                submodule.register_forward_pre_hook(self._pre_hook)
            )
            self._hook_handles.append(
                submodule.register_forward_hook(self._post_hook)
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def construct(
        cls,
        module: torch.nn.Module,
        *args,
        device: Optional[torch.device] = None,
        submodules=None,
        verbose: bool = False,
    ) -> Tuple[ComputationGraph, GraphInput]:
        """
        Build a ComputationGraph by running ``module`` on a sample input.

        Parameters
        ----------
        module : torch.nn.Module
        *args : tensors
            Sample inputs passed directly to ``module.forward()``.
        device : torch.device, optional
            Move sample inputs to this device before the forward pass.
        submodules : dict or iterable, optional
            Which submodules to treat as nodes.  Defaults to all named children.
        verbose : bool
            Print hook trace during construction.

        Returns
        -------
        (ComputationGraph, GraphInput)
            ``graph`` is ready for ``compute()`` and ``intervene()``.
            ``graph_input`` is the GraphInput built from the sample inputs —
            keep it for running experiments later.
        """
        constructor = cls(module, submodules=submodules, verbose=verbose)
        return constructor._make_graph(*args, device=device)

    # ------------------------------------------------------------------
    # Hooks  (called automatically by PyTorch during the sample forward pass)
    # ------------------------------------------------------------------

    def _pre_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple,
    ) -> tuple:
        """
        Fires before a submodule's ``forward()``.

        Reads the ``(tensor, source_name)`` tags attached to each input to
        learn which upstream nodes produced them, then sets the current node's
        ``children`` list accordingly.

        External inputs (from outside any named module) carry ``None`` as
        their tag and are used to build the leaf ``GraphInput``.

        Returns
        -------
        The unwrapped tensors that are actually passed to ``forward()``.
        """
        name = self._module_to_name[module]
        current_node = self._name_to_node[name]

        if self.verbose:
            print(f"[constructor] pre_hook  '{name}'  ({len(inputs)} input(s))")

        # Every input must be a 2-tuple (tensor, source_name_or_None)
        if not all(isinstance(x, tuple) and len(x) == 2 for x in inputs):
            raise RuntimeError(
                f"At least one input to '{name}' is not the output of a named "
                "module.  All data flowing between submodules must pass through "
                "registered submodule calls — no intermediate Python operations."
            )

        actual_inputs = tuple(t[0] for t in inputs)
        sources = [t[1] for t in inputs]

        if any(src is None for src in sources):
            # This node receives external (leaf) input
            if not all(src is None for src in sources):
                raise NotImplementedError(
                    f"Node '{name}' has mixed external and internal inputs. "
                    "This is not currently supported."
                )
            if self._current_input is not None:
                raise NotImplementedError(
                    "Only one external input (leaf) is supported per graph. "
                    f"A second leaf was detected at node '{name}'."
                )
            current_node.children = []
            self._current_input = GraphInput({name: actual_inputs})
        else:
            # Wire up edges: this node's children are the nodes that produced its inputs
            current_node.children = [self._name_to_node[src] for src in sources]

        return actual_inputs

    def _post_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple,
        output,
    ) -> tuple:
        """
        Fires after a submodule's ``forward()``.

        Tags the output with the module's name so the next module's
        ``_pre_hook`` can read where it came from.  Also populates the
        node's ``base_cache`` with the sample output.

        Returns
        -------
        ``(output, module_name)`` — the tagged output that flows onward.
        """
        name = self._module_to_name[module]
        current_node = self._name_to_node[name]

        if self.verbose:
            print(f"[constructor] post_hook '{name}'")

        # Cache the sample output so the graph can be used immediately
        if self._current_input is not None:
            current_node.base_cache[self._current_input] = output

        return output, name

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _make_graph(
        self,
        *args,
        device: Optional[torch.device] = None,
    ) -> Tuple[ComputationGraph, GraphInput]:
        """
        Run the sample forward pass and return the constructed graph.

        The sample inputs are wrapped as ``(tensor, None)`` to signal that
        they are external (leaf) inputs, then the full module is called.
        The hook machinery fires during this call and builds the graph.
        """
        if device is not None:
            wrapped = tuple((x.to(device), None) for x in args)
        else:
            wrapped = tuple((x, None) for x in args)

        if self.verbose:
            print(f"[constructor] running sample forward pass  "
                  f"({len(args)} input tensor(s))")

        # Run the full forward pass — hooks fire and build the graph topology
        _output, root_name = self.module(*wrapped)

        graph_input = self._current_input
        self._current_input = None   # reset for potential re-use

        # Remove hooks so the original model works normally after construction
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

        root_node = self._name_to_node[root_name]
        graph = ComputationGraph(root_node)

        return graph, graph_input


# ---------------------------------------------------------------------------
# Tests — run with:  python -m pipeline.constructor
# ---------------------------------------------------------------------------
def _run_tests():
    import torch
    import torch.nn as nn
    from intervention.intervention import Intervention

    torch.manual_seed(0)

    # ── Helper model: three-layer chain ─────────────────────────────────────
    #
    #   linear1 → relu → linear2
    #
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 8, bias=False)
            self.relu    = nn.ReLU()
            self.linear2 = nn.Linear(8, 2, bias=False)

        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x

    model = SimpleNet()
    sample = torch.randn(1, 4)

    # ── 1. Graph is constructed without errors ─────────────────────────────
    graph, gi = CompGraphConstructor.construct(model, sample)
    print("  construct() ran ✓")

    # ── 2. All three submodules become nodes ──────────────────────────────
    assert set(graph.nodes.keys()) == {"linear1", "relu", "linear2"}, \
        f"Unexpected nodes: {set(graph.nodes.keys())}"
    print("  all submodules registered as nodes ✓")

    # ── 3. Root is the last module ─────────────────────────────────────────
    assert graph.root.name == "linear2"
    print("  root node is linear2 ✓")

    # ── 4. Leaf is the first module ────────────────────────────────────────
    leaf_names = {n.name for n in graph.leaves}
    assert leaf_names == {"linear1"}, f"Unexpected leaves: {leaf_names}"
    print("  leaf node is linear1 ✓")

    # ── 5. Edges are correct: linear1→relu→linear2 ────────────────────────
    assert graph.nodes["relu"].children    == [graph.nodes["linear1"]]
    assert graph.nodes["linear2"].children == [graph.nodes["relu"]]
    assert graph.nodes["linear1"].children == []
    print("  graph edges are correct ✓")

    # ── 6. GraphInput was captured with the leaf node name as key ─────────
    assert "linear1" in gi
    print("  GraphInput captured ✓")

    # ── 7. graph.compute() matches model.forward() ────────────────────────
    graph.clear_caches()
    with torch.no_grad():
        expected = model(sample)
    graph_out = graph.compute(gi)
    assert torch.allclose(graph_out, expected), \
        f"Outputs differ: {graph_out} vs {expected}"
    print("  graph.compute() matches model.forward() ✓")

    # ── 8. Verbose flag prints hook trace (just check it doesn't crash) ───
    graph2, gi2 = CompGraphConstructor.construct(
        SimpleNet(), sample, verbose=True
    )
    assert graph2 is not None
    print("  verbose=True does not crash ✓")

    # ── 9. Non-Module input raises TypeError ──────────────────────────────
    try:
        CompGraphConstructor("not_a_module")
        raise AssertionError("Should have raised TypeError")
    except TypeError:
        pass
    print("  non-Module input raises TypeError ✓")

    # ── 10. Intervention on intermediate node (relu) ──────────────────────
    graph.clear_caches()
    graph.compute(gi)   # populate base caches

    # Force relu output to all-ones
    forced_relu = torch.ones(1, 8)
    interv = Intervention(gi, {"relu": forced_relu})
    base_out, patched_out = graph.intervene(interv)

    # Manually compute expected: linear2(all_ones)
    with torch.no_grad():
        expected_patched = model.linear2(forced_relu)
    assert torch.allclose(patched_out, expected_patched), \
        f"Patched output mismatch: {patched_out} vs {expected_patched}"
    print("  intervention on relu node ✓")

    print("\n✓  All CompGraphConstructor tests passed")


if __name__ == "__main__":
    _run_tests()
