"""
pipeline/graph.py

Defines ComputationGraph — a directed acyclic graph of GraphNodes that
supports both plain forward passes and causal interventions.

Typical usage
-------------
    from pipeline.graph import ComputationGraph
    from pipeline.graph_node import GraphNode
    from pipeline.graph_input import GraphInput
    from pipeline.intervention import Intervention

    class MyGraph(ComputationGraph):
        def __init__(self):
            @GraphNode()
            def leaf(x):
                return x * 2

            @GraphNode(leaf)
            def root(val):
                return val + 1

            super().__init__(root)

    g = MyGraph()

    # Plain forward pass
    inp = GraphInput({"leaf": 5})
    result = g.compute(inp)            # → 11

    # Causal intervention: force root to 99
    interv = Intervention(inp, {"root": 99})
    base, patched = g.intervene(interv)  # → (11, 99)
"""

import itertools
from collections import deque
from typing import Dict, Optional, Set, Union

import torch

from pipeline.graph_input import GraphInput
from pipeline.graph_node import GraphNode
from pipeline.intervention import Intervention
from pipeline.location import Location


class ComputationGraph:
    """
    A directed acyclic graph of :class:`~pipeline.graph_node.GraphNode` objects.

    The graph is constructed once from a root node; all reachable nodes are
    discovered automatically via DFS.  The graph then exposes two main
    operations:

    * :meth:`compute`   — plain forward pass
    * :meth:`intervene` — patched forward pass (returns base and patched result)

    Parameters
    ----------
    root : GraphNode
        The output node of the graph.
    root_output_device : torch.device, optional
        When set, the root's output tensor is moved to this device before
        being returned.  Useful when the model runs on GPU but results are
        needed on CPU.
    """

    def __init__(
        self,
        root: GraphNode,
        root_output_device: Optional[torch.device] = None,
    ):
        self.root = root
        self.root_output_device = root_output_device
        self.nodes: Dict[str, GraphNode] = {}
        self.leaves: Set[GraphNode] = set()
        self.results_cache: dict = {}        # top-level result cache
        self.cache_device: Optional[torch.device] = None

        self._validate_graph()

    # ------------------------------------------------------------------
    # Top-level result cache
    # ------------------------------------------------------------------

    def get_from_cache(self, inputs) -> Optional[object]:
        """
        Retrieve a top-level cached result (base run only, never batched).

        Returns ``None`` on a cache miss — use ``result is None`` to check,
        not ``if not result``, since zero / False are valid results.
        """
        if inputs.batched:
            return None

        result = self.results_cache.get(inputs, None)
        if (
            result is not None
            and self.cache_device is not None
            and isinstance(result, torch.Tensor)
        ):
            original_device = self._result_output_device_dict.get(inputs)
            if original_device and original_device != self.cache_device:
                return result.to(original_device)
        return result

    def save_to_cache(self, inputs, result) -> None:
        """Store a top-level result.  No-op for batched inputs."""
        if not inputs.cache_results or inputs.batched:
            return

        result_for_cache = result
        if self.cache_device is not None and isinstance(result, torch.Tensor):
            if result.device != self.cache_device:
                result_for_cache = result.to(self.cache_device)
            self._result_output_device_dict[inputs] = result.device

        self.results_cache[inputs] = result_for_cache

    def set_cache_device(self, cache_device: torch.device) -> None:
        """Pin the cache storage to a specific device (e.g. CPU) for all nodes."""
        self.cache_device = cache_device
        self._result_output_device_dict = {}
        for node in self.nodes.values():
            node.cache_device = cache_device

    # ------------------------------------------------------------------
    # Forward computation
    # ------------------------------------------------------------------

    def compute(
        self,
        inputs: GraphInput,
        store_cache: bool = True,
        iterative: bool = False,
    ):
        """
        Run a plain (un-patched) forward pass through the graph.

        Parameters
        ----------
        inputs : GraphInput
            Leaf-node values for this run.
        store_cache : bool
            Cache the root result so repeated calls with the same inputs
            are free.
        iterative : bool
            When True, use an iterative (non-recursive) topological traversal
            instead of DFS recursion.  Useful for very deep graphs that would
            otherwise overflow the call stack.

        Returns
        -------
        Root node output.
        """
        result = self.get_from_cache(inputs)
        if result is None:                          # ← was: if not result (BUG)
            self._validate_inputs(inputs)
            result = (
                self._iterative_compute(inputs)
                if iterative
                else self.root.compute(inputs)
            )

        if store_cache:
            self.save_to_cache(inputs, result)

        if isinstance(result, torch.Tensor) and self.root_output_device:
            result = result.to(self.root_output_device)
        return result

    def _iterative_compute(self, inputs: GraphInput):
        """
        Compute the graph in topological order without recursion.

        Uses Kahn's algorithm (BFS from leaves) so each node is computed
        exactly once, after all of its children (dependencies) are ready.

        This produces the same result as the recursive ``root.compute()``
        but avoids Python's call-stack limit for very deep graphs.
        """
        # Number of children not yet computed for each node.
        in_degree = {name: len(node.children) for name, node in self.nodes.items()}

        # For each node, which nodes depend on it (i.e. its parents).
        dependents: Dict[str, list] = {name: [] for name in self.nodes}
        for name, node in self.nodes.items():
            for child in node.children:
                dependents[child.name].append(name)

        # Seed with leaf nodes (no children → in_degree == 0).
        queue: deque = deque(
            name for name, deg in in_degree.items() if deg == 0
        )
        computed: Dict[str, object] = {}

        while queue:
            name = queue.popleft()
            node = self.nodes[name]

            if not node.children:
                # Leaf: pull value from inputs, unpack tuples for multi-arg fns
                values = inputs[name]
                result = (
                    node.forward(*values)
                    if isinstance(values, (tuple, list))
                    else node.forward(values)
                )
            else:
                # Non-leaf: all children are already in `computed`
                children_results = [computed[child.name] for child in node.children]
                result = node.forward(*children_results)

            computed[name] = result
            node.save_to_cache(inputs, result, to_interv=False)

            # Unlock parent nodes whose last unresolved child was just computed.
            for parent_name in dependents[name]:
                in_degree[parent_name] -= 1
                if in_degree[parent_name] == 0:
                    queue.append(parent_name)

        return computed[self.root.name]

    # ------------------------------------------------------------------
    # Intervention
    # ------------------------------------------------------------------

    def intervene(self, intervention: Intervention, store_cache: bool = True):
        """
        Run a patched forward pass and return both the base and patched results.

        Steps
        -----
        1. Validate the intervention against this graph.
        2. Run the base (un-patched) forward pass — populates all node base_caches.
           This step is required before the patched pass so that location-based
           interventions can find the base value they need to partially overwrite.
        3. Check whether the patched result is already cached.
        4. Find which nodes are "affected" (downstream of any patch).
        5. Run the patched forward pass.

        Parameters
        ----------
        intervention : Intervention
        store_cache : bool

        Returns
        -------
        (base_result, patched_result)
        """
        self._validate_interv(intervention)            # ← validate BEFORE compute

        base_res = self.compute(intervention.base)

        interv_res = self.get_from_cache(intervention)
        if interv_res is None:                         # ← was: if not interv_res (BUG)
            intervention.find_affected_nodes(self)
            interv_res = self.root.compute(intervention)
            if store_cache:
                self.save_to_cache(intervention, interv_res)

        return base_res, interv_res

    # ------------------------------------------------------------------
    # Intermediate-node access
    # ------------------------------------------------------------------

    def get_result(self, node_name: str, x: Union[GraphInput, Intervention]):
        """
        Return the output of an intermediate node for a given input.

        Runs the full forward pass (base + intervention if needed) if the
        node's result is not already cached.

        Parameters
        ----------
        node_name : str
            Name of the node whose output is requested.
        x : GraphInput or Intervention

        Returns
        -------
        The node's output value.
        """
        if node_name not in self.nodes:
            raise RuntimeError(f"Node '{node_name}' not found in graph.")

        node = self.nodes[node_name]

        if isinstance(x, GraphInput):
            return node.compute(x)

        elif isinstance(x, Intervention):
            x.find_affected_nodes(self)
            # Ensure the full forward pass has run so caches are populated.
            if x.base not in node.base_cache or x not in node.interv_cache:
                self.compute(x.base)
                self.root.compute(x)
            return node.compute(x)

        else:
            raise RuntimeError(
                f"get_result() requires a GraphInput or Intervention, "
                f"got {type(x).__name__}."
            )

    # ------------------------------------------------------------------
    # State dict (save / restore all node caches)
    # ------------------------------------------------------------------

    def get_state_dict(self) -> dict:
        """
        Snapshot all node caches into a plain dict.

        The returned dict can be saved to disk and later restored with
        :meth:`set_state_dict` to avoid rerunning expensive computations.
        """
        return {
            "base_caches": {
                name: node.base_cache
                for name, node in self.nodes.items()
                if node.cache_results
            },
            "interv_caches": {
                name: node.interv_cache
                for name, node in self.nodes.items()
                if node.cache_results
            },
            "base_output_devices": {
                name: node.base_output_devices
                for name, node in self.nodes.items()
                if node.cache_results
            },
            "interv_output_devices": {
                name: node.interv_output_devices
                for name, node in self.nodes.items()
                if node.cache_results
            },
        }

    def set_state_dict(self, d: dict) -> None:
        """
        Restore all node caches from a dict produced by :meth:`get_state_dict`.

        Unknown node names in ``d`` are silently ignored so that a state dict
        from a superset graph can be safely loaded into a smaller one.
        """
        for name, cache in d.get("base_caches", {}).items():
            if name in self.nodes:
                self.nodes[name].base_cache = cache
        for name, cache in d.get("interv_caches", {}).items():
            if name in self.nodes:
                self.nodes[name].interv_cache = cache
        for name, devices in d.get("base_output_devices", {}).items():
            if name in self.nodes:
                self.nodes[name].base_output_devices = devices
        for name, devices in d.get("interv_output_devices", {}).items():
            if name in self.nodes:
                self.nodes[name].interv_output_devices = devices

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_caches(self) -> None:
        """Clear all node caches and the top-level results cache."""
        for node in self.nodes.values():
            node.clear_caches()
        self.results_cache = {}
        if hasattr(self, "_result_output_device_dict"):
            self._result_output_device_dict = {}

    # ------------------------------------------------------------------
    # Graph analysis helpers (used by abstraction search)
    # ------------------------------------------------------------------

    def get_nodes_and_dependencies(self):
        """
        Return (node_names, dependencies) where dependencies maps each node
        name to the set of node names that directly depend on it (its parents).
        """
        node_names = list(self.nodes.keys())
        dependencies = {self.root.name: set()}

        def _fill(node):
            for child in node.children:
                if child.name in dependencies:         # ← was: if child in (BUG)
                    dependencies[child.name].add(node.name)
                else:
                    dependencies[child.name] = {node.name}
                _fill(child)

        _fill(self.root)
        return node_names, dependencies

    def get_indices(self, node_name: str):
        """
        Return all non-empty index subsets for a node's cached output shape.
        Used when searching for viable low-level intervention locations.
        """
        node = self.nodes[node_name]
        length = None
        for cached_val in node.base_cache.values():
            length = max(cached_val.shape)
            break
        if length is None:
            return []

        indices = []
        for size in range(1, length + 1):
            for subset in itertools.combinations(range(length), size):
                subset = sorted(subset)
                indices.append(Location()[subset])
        return indices

    def get_locations(self, root_locations, unwanted_low_nodes=None):
        """
        Return viable {node: index} location dicts for abstraction search.

        A node is viable if it is a descendant of ALL root_locations nodes,
        ensuring the intervention can affect every desired output.
        """
        root_nodes = [
            self.nodes[node_name]
            for loc in root_locations
            for node_name in loc
        ]

        viable_nodes = None
        for root_node in root_nodes:
            current = set()
            def _descendants(node):
                for child in node.children:
                    current.add(child.name)
                    _descendants(child)
            _descendants(root_node)

            viable_nodes = (
                current if viable_nodes is None
                else viable_nodes.intersection(current)
            )

        if not viable_nodes:
            return []

        result = []
        for node_name in viable_nodes:
            if unwanted_low_nodes and node_name in unwanted_low_nodes:
                continue
            for idx in self.get_indices(node_name):
                result.append({node_name: idx})
        return result

    # ------------------------------------------------------------------
    # Validation (private)
    # ------------------------------------------------------------------

    def _validate_graph(self) -> None:
        """
        Discover all nodes via DFS and check for name collisions.

        Raises
        ------
        RuntimeError
            If two different GraphNode objects share the same name.
        """
        def _add_node(node: GraphNode):
            if node.name in self.nodes:
                if self.nodes[node.name] is not node:
                    raise RuntimeError(
                        f"Two different nodes share the name '{node.name}'. "
                        "Every node must have a unique name."
                    )
                return   # already registered
            self.nodes[node.name] = node
            if not node.children:
                self.leaves.add(node)
            for child in node.children:
                _add_node(child)

        _add_node(self.root)

    def _validate_inputs(self, inputs: GraphInput) -> None:
        """Raise RuntimeError if any leaf node is missing from ``inputs``."""
        for leaf in self.leaves:
            if leaf.name not in inputs:
                raise RuntimeError(
                    f"No input provided for leaf node '{leaf.name}'."
                )

    def _validate_interv(self, intervention: Intervention) -> None:
        """
        Validate an intervention against this graph.

        Checks
        ------
        * All leaf nodes have inputs in ``intervention.base``.
        * The intervention is non-empty (at least one node is patched).
        * Every patched node name exists in this graph.
        """
        self._validate_inputs(intervention.base)

        if len(intervention.intervention) == 0:     # ← was: only None check (BUG)
            raise RuntimeError(
                "Must specify at least one intervention node and value."
            )

        for name in intervention.intervention.values.keys():
            if name not in self.nodes:
                raise RuntimeError(
                    f"Intervention node '{name}' not found in this graph."
                )


# ---------------------------------------------------------------------------
# Tests — run with:  python -m pipeline.graph
# ---------------------------------------------------------------------------
def _run_tests():
    from pipeline.graph_input import GraphInput
    from pipeline.graph_node import GraphNode
    from pipeline.intervention import Intervention

    # ── Helper: build a simple two-level arithmetic graph ─────────────────
    #
    #   leaf1 (a+b+c) ──┬──► child1 (*2) ──┐
    #                   └──► child2 (-leaf2)─┼──► root (+1)
    #   leaf2 ((d+e)/10)────────────────────┘
    #
    def make_graph():
        class ArithGraph(ComputationGraph):
            def __init__(self):
                @GraphNode()
                def leaf1(a, b, c):
                    return a + b + c            # (1,10,100) → 111

                @GraphNode()
                def leaf2(d, e):
                    return (d + e) / 10         # (1000,1000) → 200

                @GraphNode(leaf1)
                def child1(x):
                    return x * 2               # 111*2 = 222

                @GraphNode(leaf1, leaf2)
                def child2(x, y):
                    return x - y               # 111-200 = -89

                @GraphNode(child1, child2)
                def root(w, z):
                    return w + z + 1           # 222-89+1 = 134

                super().__init__(root)
        return ArithGraph()

    inp = GraphInput({"leaf1": (1, 10, 100), "leaf2": (1000, 1000)})

    # ── 1. Basic forward pass ──────────────────────────────────────────────
    g = make_graph()
    assert g.compute(inp) == 134
    print("  basic compute ✓")

    # ── 2. Intermediate node values ────────────────────────────────────────
    g.clear_caches()
    assert g.get_result("leaf1",  inp) == 111
    assert g.get_result("leaf2",  inp) == 200
    assert g.get_result("child1", inp) == 222
    assert g.get_result("child2", inp) == -89
    print("  get_result (base) ✓")

    # ── 3. Result caching: same input → same object returned ───────────────
    g.clear_caches()
    r1 = g.compute(inp)
    r2 = g.compute(inp)   # should hit top-level cache
    assert r1 == r2
    print("  result caching ✓")

    # ── 4. Falsy-result caching: result=0 must not be treated as cache miss ─
    class ZeroGraph(ComputationGraph):
        def __init__(self):
            @GraphNode()
            def leaf(x):
                return x
            @GraphNode(leaf)
            def root(v):
                return 0          # always returns zero (falsy!)
            super().__init__(root)

    zg = ZeroGraph()
    zi = GraphInput({"leaf": 1})
    zg.clear_caches()
    r = zg.compute(zi)
    assert r == 0
    r2 = zg.compute(zi)   # must hit cache — NOT recompute
    assert r2 == 0
    print("  falsy result (0) caching ✓")

    # ── 5. Intervention: patch leaf1 ──────────────────────────────────────
    g.clear_caches()
    i = Intervention(inp)
    i.set_intervention("leaf1", 100)
    base, after = g.intervene(i)

    assert base == 134
    # child1 = 100*2=200, child2 = 100-200=-100, root = 200-100+1=101
    assert after == 101
    assert i.affected_nodes == {"leaf1", "child1", "child2", "root"}
    print("  intervention (leaf patch) ✓")

    # ── 6. Intermediate results after intervention ──────────────────────────
    assert g.get_result("child1", inp) == 222   # base unchanged
    assert g.get_result("child1", i)   == 200   # patched path
    assert g.get_result("child2", i)   == -100
    assert g.get_result("leaf2",  i)   == 200   # unaffected
    print("  get_result (intervention) ✓")

    # ── 7. Empty intervention raises RuntimeError ──────────────────────────
    g.clear_caches()
    empty_i = Intervention(inp)
    try:
        g.intervene(empty_i)
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "intervention" in str(e).lower()
    print("  empty intervention raises ✓")

    # ── 8. Intervention on unknown node raises RuntimeError ───────────────
    g.clear_caches()
    bad_i = Intervention(inp, {"nonexistent": 99})
    try:
        g.intervene(bad_i)
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "nonexistent" in str(e)
    print("  unknown node raises ✓")

    # ── 9. Iterative compute matches recursive compute ──────────────────────
    g2 = make_graph()
    g2.clear_caches()
    r_recursive = g2.compute(inp, iterative=False)
    g2.clear_caches()
    r_iterative = g2.compute(inp, iterative=True)
    assert r_recursive == r_iterative
    print("  iterative compute matches recursive ✓")

    # ── 10. get_state_dict / set_state_dict round-trip ────────────────────
    g3 = make_graph()
    g3.clear_caches()
    g3.compute(inp)
    state = g3.get_state_dict()

    g4 = make_graph()    # fresh graph, empty caches
    g4.set_state_dict(state)
    # After restoring, node caches should be populated
    assert inp in g4.nodes["leaf1"].base_cache
    assert g4.nodes["child1"].base_cache[inp] == 222
    print("  get/set_state_dict ✓")

    # ── 11. clear_caches empties everything ─────────────────────────────────
    g3.clear_caches()
    assert g3.results_cache == {}
    assert g3.nodes["leaf1"].base_cache == {}
    print("  clear_caches ✓")

    # ── 12. Duplicate node name raises RuntimeError ────────────────────────
    try:
        class BadGraph(ComputationGraph):
            def __init__(self):
                @GraphNode()
                def leaf(x): return x

                # Different object, same name — must be rejected
                impostor = GraphNode(name="leaf", forward=lambda x: x + 1)

                @GraphNode(leaf, impostor)
                def root(a, b): return a + b
                super().__init__(root)
        BadGraph()
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "leaf" in str(e)
    print("  duplicate node name raises ✓")

    # ── 13. Missing leaf input raises RuntimeError ─────────────────────────
    g5 = make_graph()
    incomplete = GraphInput({"leaf1": (1, 10, 100)})   # leaf2 missing
    try:
        g5.compute(incomplete)
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "leaf2" in str(e)
    print("  missing leaf input raises ✓")

    # ── 14. get_nodes_and_dependencies ────────────────────────────────────
    g6 = make_graph()
    node_names, deps = g6.get_nodes_and_dependencies()
    assert "leaf1" in node_names
    assert "root" in node_names
    # leaf1's dependents should include child1 and child2
    assert "child1" in deps["leaf1"]
    assert "child2" in deps["leaf1"]
    print("  get_nodes_and_dependencies ✓")

    print("\n✓  All ComputationGraph tests passed")


if __name__ == "__main__":
    _run_tests()
