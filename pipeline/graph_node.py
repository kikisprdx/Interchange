"""
pipeline/graph_node.py

Defines GraphNode — one node in a causal computation graph.

A GraphNode wraps a Python function (its ``forward`` method) and knows about
its children (the nodes whose outputs feed into it as inputs).  It also owns
the result caches for base runs and intervention runs.

Typical usage
-------------
Nodes are most naturally defined with the decorator syntax:

    from pipeline.graph_node import GraphNode

    @GraphNode()
    def leaf(x):           # leaf: no children, takes input from GraphInput
        return x * 2

    @GraphNode(leaf)
    def root(val):         # non-leaf: takes leaf's output as argument
        return val + 1

Or constructed directly when wrapping an existing callable:

    node = GraphNode(name="embed", forward=embedding_layer.forward)
"""

from typing import Callable, List, Optional, Union

import torch

from pipeline.graph_input import GraphInput
from pipeline.intervention import Intervention
from pipeline.utils import copy_helper


class GraphNode:
    """
    One node in a causal computation graph.

    Parameters
    ----------
    *children : GraphNode
        Nodes whose outputs feed into this node as arguments.
        Empty for leaf nodes (they read directly from GraphInput).
    name : str, optional
        Node identifier.  Defaults to the decorated function's ``__name__``.
    forward : callable, optional
        The function to call when computing this node.  Normally provided
        via the decorator syntax rather than directly.
    cache_results : bool
        When True (default), cache base and intervention outputs so they
        are not recomputed on subsequent calls with the same input.
    """

    def __init__(
        self,
        *children: "GraphNode",
        name: Optional[str] = None,
        forward: Optional[Callable] = None,
        cache_results: bool = True,
    ):
        self.children = children          # uses the property setter below
        self.cache_results = cache_results
        self.name = name
        self.cache_device: Optional[torch.device] = None

        if cache_results:
            self.base_cache: dict = {}        # key: GraphInput  → value: result
            self.interv_cache: dict = {}      # key: Intervention → value: result
            self.base_output_devices: dict = {}
            self.interv_output_devices: dict = {}

        if forward is not None:
            self.forward = forward
            if name is None:
                self.name = forward.__name__

    # ------------------------------------------------------------------
    # Decorator protocol  (@GraphNode() syntax)
    # ------------------------------------------------------------------

    def __call__(self, f: Callable) -> "GraphNode":
        """
        Called immediately after ``__init__`` when used as a decorator.

        The decorated function becomes this node's ``forward`` method,
        and its ``__name__`` becomes the node's name (unless one was given
        explicitly).

            @GraphNode(child1, child2)
            def hidden(a, b):
                return a + b
        """
        self.forward = f
        if self.name is None:
            self.name = f.__name__
        return self

    # ------------------------------------------------------------------
    # Children  (property so children_dict stays in sync)
    # ------------------------------------------------------------------

    @property
    def children(self) -> List["GraphNode"]:
        return self._children

    @children.setter
    def children(self, values):
        self._children = list(values)
        self._children_dict = {c.name: c for c in self._children}

    @property
    def children_dict(self) -> dict:
        """Dict mapping child name → child GraphNode."""
        return self._children_dict

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def get_from_cache(
        self,
        inputs: Union[GraphInput, Intervention],
        from_interv: bool,
    ) -> Optional[object]:
        """
        Retrieve a previously cached result.

        Parameters
        ----------
        inputs : GraphInput or Intervention
            The key to look up.
        from_interv : bool
            True  → look in ``interv_cache`` (intervened run).
            False → look in ``base_cache``   (base run).

        Returns
        -------
        Cached result, or ``None`` if not found.
        """
        if not self.cache_results:
            return None

        assert (from_interv and isinstance(inputs, Intervention)) or \
               (not from_interv and isinstance(inputs, GraphInput)), \
            "from_interv=True requires Intervention; from_interv=False requires GraphInput"

        cache = self.interv_cache if from_interv else self.base_cache
        device_map = self.interv_output_devices if from_interv else self.base_output_devices

        if inputs.batched:
            return self._get_batched_from_cache(inputs, cache, device_map)
        else:
            return self._get_single_from_cache(inputs, cache, device_map)

    def _get_single_from_cache(self, inputs, cache, device_map):
        result = cache.get(inputs, None)
        if result is None:
            return None
        if self.cache_device is not None and isinstance(result, torch.Tensor):
            original_device = device_map.get(inputs)
            if original_device and original_device != self.cache_device:
                return result.to(original_device)
        return result

    def _get_batched_from_cache(self, inputs, cache, device_map):
        values = [cache.get(key, None) for key in inputs.keys]
        if any(v is None for v in values):
            return None

        if isinstance(values[0], torch.Tensor):
            stack_dim = 0 if values[0].dim() == 0 else inputs.batch_dim
            result = torch.stack(values, dim=stack_dim)
            if self.cache_device is not None:
                original_device = device_map.get(inputs.keys[0])
                if original_device and original_device != self.cache_device:
                    return result.to(original_device)
            return result
        else:
            return values

    def save_to_cache(self, inputs, result, to_interv: bool):
        """
        Store a computed result in the appropriate cache.

        Batched results are split so each example is stored under its own key,
        enabling per-example cache lookups later.
        """
        if not self.cache_results or not inputs.cache_results:
            return

        cache = self.interv_cache if to_interv else self.base_cache
        device_map = self.interv_output_devices if to_interv else self.base_output_devices

        if inputs.batched:
            self._save_batched_to_cache(inputs, result, cache, device_map)
        else:
            self._save_single_to_cache(inputs, result, cache, device_map)

    def _save_single_to_cache(self, inputs, result, cache, device_map):
        result_for_cache = result
        if self.cache_device is not None and isinstance(result, torch.Tensor):
            if result.device != self.cache_device:
                result_for_cache = result.to(self.cache_device)
            device_map[inputs] = result.device
        cache[inputs] = result_for_cache

    def _save_batched_to_cache(self, inputs, result, cache, device_map):
        result_for_cache = result
        if self.cache_device is not None and isinstance(result, torch.Tensor):
            if result.device != self.cache_device:
                result_for_cache = result.to(self.cache_device)
            for key in inputs.keys:
                device_map[key] = result.device

        if not isinstance(result_for_cache, torch.Tensor):
            raise RuntimeError(
                f"Batched cache only supports tensor results; got {type(result_for_cache)}."
            )

        if inputs.batch_dim == 0 or result_for_cache.dim() == 1:
            # Each row is one example — zip directly
            for key, value in zip(inputs.keys, result_for_cache):
                cache[key] = value
        else:
            # Split along batch_dim and squeeze that dimension out
            splits = result_for_cache.split(1, dim=inputs.batch_dim)
            for key, value in zip(inputs.keys, splits):
                cache[key] = value.squeeze(inputs.batch_dim)

    # ------------------------------------------------------------------
    # Forward computation
    # ------------------------------------------------------------------

    def compute(self, inputs: Union[GraphInput, Intervention]):
        """
        Compute and return this node's output.

        The same method handles both base runs (``inputs`` is a GraphInput)
        and intervention runs (``inputs`` is an Intervention).  On an
        intervention run the method checks whether this node is "affected"
        (i.e. directly patched or downstream of a patch) and routes to the
        appropriate sub-method.

        Parameters
        ----------
        inputs : GraphInput or Intervention

        Returns
        -------
        The node's output value (type depends on the forward function).
        """
        intervention, is_affected = self._resolve_intervention(inputs)
        base_inputs = intervention.base if intervention else inputs

        # ── Try cache first ──────────────────────────────────────────────
        cache_key = intervention if is_affected else base_inputs
        result = self.get_from_cache(cache_key, from_interv=is_affected)
        if result is not None:
            return result

        # ── Compute ───────────────────────────────────────────────────────
        if intervention and self.name in intervention.intervention:
            result = self._apply_patch(base_inputs, intervention)
            # Even though this node's result is already set by the patch,
            # we still compute descendants so that any location-based
            # interventions further down the graph find their base values
            # in cache when they need them.
            self._propagate_to_children(base_inputs, intervention)
        else:
            result = self._compute_forward(base_inputs, intervention)

        self.save_to_cache(cache_key, result, to_interv=is_affected)
        return result

    # ── compute() helpers ────────────────────────────────────────────────

    @staticmethod
    def _resolve_intervention(inputs):
        """
        Unpack inputs into (intervention_or_None, is_affected).

        Raises RuntimeError if an Intervention is passed before
        ``find_affected_nodes()`` has been called.
        """
        if isinstance(inputs, Intervention):
            if inputs.affected_nodes is None:
                raise RuntimeError(
                    "Must call Intervention.find_affected_nodes(graph) "
                    "before running compute()."
                )
            is_affected = inputs.name in inputs.affected_nodes if False else \
                None  # placeholder — set below

            # We need the node's own name to decide; that's done in compute()
            # itself.  Return the full object and let compute() check.
            return inputs, None   # is_affected resolved per-node in compute()
        return None, False

    def _apply_patch(self, base_inputs: GraphInput, intervention: Intervention):
        """
        Apply the intervention value to this node's output.

        * If a location is specified: clone the base output and overwrite
          just the indexed slice.
        * Otherwise: replace the entire output with the intervention value.
        """
        if self.name in intervention.location:
            if not self.cache_results:
                raise RuntimeError(
                    f"Cannot apply a location-based intervention on '{self.name}' "
                    "because cache_results=False — the base output is not stored."
                )
            base_result = self.get_from_cache(base_inputs, from_interv=False)
            if base_result is None:
                raise RuntimeError(
                    f"Base result for '{self.name}' not in cache. "
                    "Call graph.compute(base_inputs) before graph.intervene()."
                )
            result = copy_helper(base_result)
            result[intervention.location[self.name]] = intervention[self.name]
        else:
            result = intervention[self.name]
        return result

    def _compute_forward(
        self,
        base_inputs: GraphInput,
        intervention: Optional[Intervention],
    ):
        """
        Compute this node's output from its ``forward`` function.

        * Leaf nodes pull their value from ``base_inputs`` and call ``forward``.
          If the stored value is a ``tuple`` or ``list`` it is unpacked into
          positional arguments, so multi-argument leaf functions work naturally::

              @GraphNode()
              def leaf(a, b, c):     # forward(a, b, c)
                  return a + b + c

              gi = GraphInput({"leaf": (1, 10, 100)})  # stored as tuple

        * Non-leaf nodes collect their children's outputs and call ``forward``
          with those as positional arguments.
        """
        pass_through = intervention if intervention is not None else base_inputs

        if not self.children:
            # Leaf: read value from graph input
            values = base_inputs[self.name]
            if isinstance(values, (tuple, list)):
                return self.forward(*values)   # unpack multi-arg tuples
            return self.forward(values)
        else:
            # Non-leaf: gather children results then call forward
            children_results = [child.compute(pass_through) for child in self.children]
            return self.forward(*children_results)

    def _propagate_to_children(
        self,
        base_inputs: GraphInput,
        intervention: Intervention,
    ):
        """
        Compute children after a direct patch on this node.

        This is a side-effect-only step: results are discarded here.
        Its purpose is to ensure that descendants' base values end up in
        their caches, so that any location-based interventions on those
        descendants can retrieve the base value when they need it.
        """
        if self.children:
            pass_through = intervention
            for child in self.children:
                child.compute(pass_through)

    # ------------------------------------------------------------------
    # Override _resolve_intervention to be per-node (fix the placeholder)
    # ------------------------------------------------------------------

    def compute(self, inputs: Union[GraphInput, Intervention]):  # noqa: F811
        """
        Compute and return this node's output.

        Handles both plain GraphInput (base run) and Intervention (patched run).
        """
        # ── Resolve intervention context ─────────────────────────────────
        intervention = None
        is_affected = False
        if isinstance(inputs, Intervention):
            intervention = inputs
            if intervention.affected_nodes is None:
                raise RuntimeError(
                    "Must call Intervention.find_affected_nodes(graph) "
                    "before running compute()."
                )
            is_affected = self.name in intervention.affected_nodes
        base_inputs = intervention.base if intervention else inputs

        # ── Try cache first ──────────────────────────────────────────────
        cache_key = intervention if is_affected else base_inputs
        result = self.get_from_cache(cache_key, from_interv=is_affected)
        if result is not None:
            return result

        # ── Compute ───────────────────────────────────────────────────────
        if intervention and self.name in intervention.intervention:
            result = self._apply_patch(base_inputs, intervention)
            self._propagate_to_children(base_inputs, intervention)
        else:
            result = self._compute_forward(base_inputs, intervention)

        self.save_to_cache(cache_key, result, to_interv=is_affected)
        return result

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_caches(self):
        """Reset all base and intervention caches to empty dicts."""
        self.base_cache = {}
        self.interv_cache = {}
        self.base_output_devices = {}
        self.interv_output_devices = {}

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f'GraphNode("{self.name}")'


# ---------------------------------------------------------------------------
# Tests — run with:  python -m pipeline.graph_node
# ---------------------------------------------------------------------------
def _run_tests():
    import torch
    from pipeline.graph_input import GraphInput

    # ── 1. Decorator syntax: @GraphNode() ─────────────────────────────────
    @GraphNode()
    def leaf(x):
        return x * 2

    assert leaf.name == "leaf"
    assert leaf.children == []
    assert repr(leaf) == 'GraphNode("leaf")'
    print("  decorator syntax ✓")

    # ── 2. Parent-child relationship ──────────────────────────────────────
    @GraphNode(leaf)
    def root(val):
        return val + 1

    assert root.name == "root"
    assert root.children == [leaf]
    assert root.children_dict == {"leaf": leaf}
    print("  parent-child relationship ✓")

    # ── 3. Explicit name override ──────────────────────────────────────────
    @GraphNode(name="my_leaf")
    def some_fn(x):
        return x

    assert some_fn.name == "my_leaf"
    print("  explicit name ✓")

    # ── 4. Direct construction (no decorator) ─────────────────────────────
    direct = GraphNode(name="direct", forward=lambda x: x + 10)
    assert direct.name == "direct"
    print("  direct construction ✓")

    # ── 5. Leaf: single-value forward call ────────────────────────────────
    @GraphNode()
    def single_leaf(x):
        return x + 100

    gi = GraphInput({"single_leaf": 5})
    interv_none = None
    result = single_leaf._compute_forward(gi, interv_none)
    assert result == 105
    print("  single-value leaf forward ✓")

    # ── 6. Leaf: tuple-unpacking for multi-arg functions ──────────────────
    @GraphNode()
    def multi_leaf(a, b, c):
        return a + b + c

    gi = GraphInput({"multi_leaf": (1, 10, 100)})
    result = multi_leaf._compute_forward(gi, None)
    assert result == 111, f"Expected 111, got {result}"
    print("  multi-arg tuple unpacking ✓")

    # ── 7. Non-leaf: calls children and passes results to forward ─────────
    @GraphNode()
    def child_a(x):
        return x * 2

    @GraphNode(child_a)
    def parent_node(val):
        return val + 1

    gi2 = GraphInput({"child_a": 5})
    result = parent_node._compute_forward(gi2, None)
    assert result == 11   # child_a(5)=10, parent_node(10)=11
    print("  non-leaf forward ✓")

    # ── 8. save_to_cache / get_from_cache round-trip ─────────────────────
    @GraphNode()
    def cached_node(x):
        return x * 3

    gi3 = GraphInput({"cached_node": 7})
    cached_node.save_to_cache(gi3, result=21, to_interv=False)
    cached = cached_node.get_from_cache(gi3, from_interv=False)
    assert cached == 21
    print("  save_to_cache / get_from_cache ✓")

    # ── 9. Cache miss returns None ────────────────────────────────────────
    @GraphNode()
    def empty_node(x):
        return x

    gi4 = GraphInput({"empty_node": 0})
    assert empty_node.get_from_cache(gi4, from_interv=False) is None
    print("  cache miss returns None ✓")

    # ── 10. clear_caches empties all dicts ────────────────────────────────
    cached_node.clear_caches()
    assert cached_node.base_cache == {}
    assert cached_node.interv_cache == {}
    print("  clear_caches ✓")

    # ── 11. cache_results=False: get always returns None ──────────────────
    @GraphNode(cache_results=False)
    def no_cache_node(x):
        return x

    gi5 = GraphInput({"no_cache_node": 1})
    no_cache_node.save_to_cache(gi5, result=99, to_interv=False)
    assert no_cache_node.get_from_cache(gi5, from_interv=False) is None
    print("  cache_results=False ✓")

    print("\n✓  All GraphNode tests passed")


if __name__ == "__main__":
    _run_tests()
