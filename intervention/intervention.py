"""
pipeline/intervention.py

Defines Intervention — pairs a base GraphInput with a set of node patches
(intervention values) and optional locations (which slice to patch).

Together these three pieces describe one causal intervention experiment:

    base         →  the inputs to run the graph on as normal
    intervention →  which node(s) to patch and to what value(s)
    location     →  (optional) which slice of a node's output to patch

Typical usage
-------------
    from intervention.graph_input import GraphInput
    from intervention.intervention import Intervention
    from intervention.location import LOC

    base = GraphInput({"x": 3, "y": 5})

    # Patch an entire node output
    interv = Intervention(base, intervention={"hidden": new_value})

    # Patch only a slice of a node output
    interv = Intervention(
        base,
        intervention={"hidden": new_value[:64]},
        location={"hidden": LOC[:64]},
    )

    # Inline slice syntax (equivalent to the above)
    interv = Intervention(base, intervention={"hidden[:64]": new_value[:64]})
"""

import re
from typing import Dict, Optional, Sequence, Union

import torch

from intervention.location import Location
from intervention.graph_input import GraphInput


class Intervention:
    """
    Pairs a base GraphInput with node-level patches for a causal intervention.

    Parameters
    ----------
    base : GraphInput or dict
        Inputs for the normal (un-patched) forward pass.  A plain dict is
        automatically wrapped in a GraphInput.
    intervention : GraphInput or dict, optional
        Maps node names → values to patch into the graph.
        Key names may embed a location using bracket syntax, e.g.
        ``"hidden[5:10]"`` patches only indices 5–10 of that node.
    location : dict, optional
        Maps node names → index/slice (a value accepted by
        :meth:`~pipeline.location.Location.process`).  If a node appears in
        both ``intervention`` and ``location``, only that slice is replaced.
    cache_results : bool
        Cache the results of this intervention during computation.
    cache_base_results : bool
        Cache the base (un-patched) run results.  Only matters when ``base``
        is given as a plain dict (GraphInput objects carry their own flag).
    batched : bool
        True when ``base`` is a batched GraphInput and ``intervention``
        values are batched tensors.
    batch_dim : int
        Batch dimension index for PyTorch tensors (default 0).
    keys : sequence, optional
        Unique hashable key for each example in the batch.
        Required when ``batched=True``.
    device : torch.device, optional
        When provided, all intervention tensor values are moved to this device.
    """

    def __init__(
        self,
        base: Union[Dict, GraphInput],
        intervention: Union[Dict, GraphInput] = None,
        location: Dict = None,
        cache_results: bool = True,
        cache_base_results: bool = False,
        batched: bool = False,
        batch_dim: int = 0,
        keys: Optional[Sequence] = None,
        device: Optional[torch.device] = None,
    ):
        # ── Assign ALL attributes BEFORE calling _setup(), because _setup()
        #    reads self.cache_base_results when wrapping a dict base. ────────
        self.cache_results = cache_results
        self.cache_base_results = cache_base_results
        self.batched = batched
        self.batch_dim = batch_dim
        self.keys = keys
        self.affected_nodes: Optional[set] = None  # computed lazily

        if batched:
            if not (isinstance(base, GraphInput) and base.batched):
                raise ValueError(
                    "batched=True requires a batched GraphInput as the base."
                )
            if not keys:
                raise ValueError(
                    "Must provide a `keys` sequence when batched=True."
                )

        # Normalise None → empty dict so _setup always receives a dict.
        intervention = {} if intervention is None else intervention
        location = {} if location is None else location

        self._setup(base, intervention, location)

        # Optionally move intervention tensors to a device after construction.
        if device is not None:
            self._move_intervention_to_device(device)

    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    def make_batched(
        cls,
        base: GraphInput,
        keys: Sequence,
        intervention: Union[Dict, GraphInput] = None,
        location: Dict = None,
        cache_results: bool = False,
        cache_base_results: bool = False,
        batch_dim: int = 0,
        device: Optional[torch.device] = None,
    ) -> "Intervention":
        """
        Convenience constructor for batched interventions.

        Equivalent to ``Intervention(..., batched=True, keys=keys)``.
        """
        return cls(
            base=base,
            intervention=intervention,
            location=location,
            cache_results=cache_results,
            cache_base_results=cache_base_results,
            batched=True,
            batch_dim=batch_dim,
            keys=keys,
            device=device,
        )

    # ------------------------------------------------------------------
    # Internal initializer  (shared by __init__ and property setters)
    # ------------------------------------------------------------------

    def _setup(
        self,
        base=None,
        intervention=None,
        location=None,
    ):
        """
        Internal initializer called from ``__init__`` and property setters.

        Sentinel semantics for each argument:

        * ``None``  → keep the existing value unchanged
        * ``{}``    → clear / reset to empty
        * any dict  → replace with that dict

        Location annotations embedded in intervention key names
        (e.g. ``"hidden[5:10]"``) are extracted and merged into ``location``.
        """
        # ── base ────────────────────────────────────────────────────────────
        if base is not None:
            if isinstance(base, dict):
                base = GraphInput(base, cache_results=self.cache_base_results)
            self._base = base

        # ── location ────────────────────────────────────────────────────────
        if location is not None:
            # Normalise each value through Location.process (accepts strings,
            # slices, ints, tuples, Ellipsis …).
            location = {
                name: Location.process(loc)
                for name, loc in location.items()
            }
        else:
            location = self._location  # keep existing (safe after first _setup)

        # ── intervention ────────────────────────────────────────────────────
        if intervention is not None:
            if isinstance(intervention, GraphInput):
                intervention = dict(intervention.values)  # unwrap to plain dict
            else:
                intervention = dict(intervention)         # defensive copy

            # Parse inline locations like "hidden[5:10]" → node "hidden",
            # location slice(5, 10).
            location = _extract_inline_locations(intervention, location)

            self._intervention = GraphInput(intervention)

        # ── persist & validate ───────────────────────────────────────────────
        self._location = location

        for loc_name in self._location:
            if loc_name not in self._intervention:
                raise ValueError(
                    f"Location key '{loc_name}' has no matching intervention value."
                )

        # Changing the intervention invalidates any cached affected-node set.
        self.affected_nodes = None

    def _move_intervention_to_device(self, device: torch.device):
        """Replace intervention GraphInput with one whose tensors are on ``device``."""
        new_values = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self._intervention.values.items()
        }
        self._intervention = GraphInput(new_values)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def base(self) -> GraphInput:
        """The base inputs (un-patched forward pass)."""
        return self._base

    @property
    def intervention(self) -> GraphInput:
        """GraphInput mapping node names → patch values."""
        return self._intervention

    @intervention.setter
    def intervention(self, values):
        """Replace all intervention values; clears existing locations."""
        self._setup(intervention=values, location={})

    @property
    def location(self) -> Dict:
        """Dict mapping node names → index / slice to patch."""
        return self._location

    @location.setter
    def location(self, values):
        """Replace all location values."""
        self._setup(location=values)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def set_intervention(self, name: str, value):
        """
        Add or update a single intervention value.

        Unlike assigning to ``self.intervention`` (which clears locations),
        this call preserves all existing locations.
        """
        current = dict(self._intervention.values)   # copy — don't mutate GraphInput
        current[name] = value
        self._setup(intervention=current, location=None)  # None → keep locations

    def set_location(self, name: str, value):
        """Add or update a single location. Existing interventions are preserved."""
        current = dict(self._location)
        current[name] = value
        self._setup(location=current)

    # ------------------------------------------------------------------
    # Dict-like access into intervention values
    # ------------------------------------------------------------------

    def __getitem__(self, name: str):
        return self._intervention.values[name]

    def __setitem__(self, name: str, value):
        self.set_intervention(name, value)

    # ------------------------------------------------------------------
    # Causal graph: find nodes affected by this intervention
    # ------------------------------------------------------------------

    def find_affected_nodes(self, graph) -> set:
        """
        Find all nodes in ``graph`` whose output changes under this intervention.

        A node is "affected" if it is directly patched OR if any of its
        *descendants* are patched (because its output depends on theirs).

        The result is cached in ``self.affected_nodes``; repeated calls are free.
        The cache is automatically cleared whenever ``_setup()`` is called.

        Parameters
        ----------
        graph : ComputationGraph

        Returns
        -------
        set of str
            Names of all nodes whose output differs from the base run.
        """
        if self.affected_nodes is not None:
            return self.affected_nodes

        if not self._intervention or len(self._intervention) == 0:
            self.affected_nodes = set()
            return set()

        affected: set = set()

        def _mark_affected(node) -> bool:
            """DFS — returns True if this node or any child is patched."""
            # Do NOT use any() here — it short-circuits and skips unvisited
            # children, leaving them unmarked even when they are affected.
            # Every child must be visited regardless of what earlier ones return.
            node_is_affected = False
            for child in node.children:
                if _mark_affected(child):
                    node_is_affected = True
            node_is_affected = node_is_affected or (node.name in self._intervention)

            if node_is_affected:
                affected.add(node.name)
            return node_is_affected

        _mark_affected(graph.root)
        self.affected_nodes = affected
        return affected


# ---------------------------------------------------------------------------
# Module-level helper (not part of the public API)
# ---------------------------------------------------------------------------

def _extract_inline_locations(intervention: dict, location: dict) -> dict:
    """
    Parse location annotations embedded in intervention key names.

    ``{"hidden[5:10]": value}``  →  key renamed to ``"hidden"``,
    ``location["hidden"] = slice(5, 10)`` added.

    Modifies ``intervention`` in-place and returns the updated ``location`` dict.
    """
    loc_pattern = re.compile(r"\[.*?]")
    to_rename = {}

    for name in list(intervention.keys()):
        match = loc_pattern.search(name)
        if match:
            true_name = name.split("[")[0]
            loc_str = match.group().strip("[]")
            location[true_name] = Location.parse_str(loc_str)
            to_rename[name] = true_name

    for old_name, new_name in to_rename.items():
        intervention[new_name] = intervention.pop(old_name)

    return location


# ---------------------------------------------------------------------------
# Tests — run with:  python -m pipeline.intervention
# ---------------------------------------------------------------------------
def _run_tests():
    import torch
    from intervention.location import LOC

    base_values = {str(x): torch.randn(10) for x in range(3)}
    base_input = GraphInput(base_values)

    # ── 1. Base only: empty intervention and location ────────────────────
    i = Intervention(base_input)
    assert isinstance(i.location, dict) and len(i.location) == 0
    assert isinstance(i.intervention, GraphInput) and len(i.intervention) == 0
    print("  base-only construction ✓")

    # ── 2. With a plain dict intervention ────────────────────────────────
    node_val = torch.randn(10)
    i = Intervention(base_input, intervention={"node": node_val})
    assert "node" in i.intervention
    assert i.location == {}
    print("  intervention dict ✓")

    # ── 3. Inline location syntax  "node[5:10]" ──────────────────────────
    i = Intervention(base_input, intervention={"node[5:10]": torch.randn(5)})
    assert "node" in i.intervention
    assert i.location["node"] == LOC[5:10]
    print("  inline location syntax ✓")

    # ── 4. Inline location with spaces  "node[5 :10]" ────────────────────
    i = Intervention(base_input, intervention={"node[5 :10]": torch.randn(5)})
    assert i.location["node"] == LOC[5:10]
    print("  inline location with spaces ✓")

    # ── 5. Separate location dict ─────────────────────────────────────────
    i = Intervention(base_input,
                     intervention={"node": torch.randn(5)},
                     location={"node": LOC[5:10]})
    assert i.location["node"] == LOC[5:10]
    print("  separate location dict ✓")

    # ── 6. Multi-dim inline location  "node[0,...,:]" ─────────────────────
    i = Intervention(base_input,
                     intervention={"node[0,...,:]": torch.randn(10)})
    assert i.location["node"] == LOC[0, ..., :]
    print("  multi-dim inline location ✓")

    # ── 7. intervention setter replaces values AND clears location ─────────
    i = Intervention(base_input, intervention={"node": torch.randn(10)})
    old_interv = i.intervention
    i.intervention = {"node": torch.randn(10)}
    assert i.intervention is not old_interv   # new GraphInput object
    assert i.location == {}                   # cleared
    print("  intervention setter ✓")

    # ── 8. location setter ────────────────────────────────────────────────
    i = Intervention(base_input, intervention={"node": torch.randn(5)})
    i.location = {"node": LOC[5:10]}
    assert i.location["node"] == LOC[5:10]
    print("  location setter ✓")

    # ── 9. set_intervention preserves existing locations ──────────────────
    i = Intervention(base_input)
    i.set_intervention("node", torch.randn(10))
    assert len(i.intervention["node"]) == 10
    first_interv = i.intervention

    i.set_intervention("node2", torch.randn(10))
    assert i.intervention is not first_interv  # new immutable GraphInput
    assert len(i.intervention["node"]) == 10
    assert len(i.intervention["node2"]) == 10
    print("  set_intervention ✓")

    # ── 10. set_intervention must not mutate the old GraphInput ───────────
    i = Intervention(base_input, intervention={"node": torch.randn(10)})
    old_gi = i.intervention
    i.set_intervention("node2", torch.randn(10))
    assert "node2" not in old_gi          # old object unchanged
    print("  set_intervention immutability ✓")

    # ── 11. __setitem__ / __getitem__ ─────────────────────────────────────
    i = Intervention(base_input)
    i["node"] = torch.randn(10)
    assert len(i["node"]) == 10
    print("  __setitem__ / __getitem__ ✓")

    # ── 12. affected_nodes cache is cleared on intervention change ─────────
    i = Intervention(base_input, intervention={"node": torch.randn(10)})
    i.affected_nodes = {"node"}           # simulate cached result
    i.set_intervention("node2", torch.randn(10))
    assert i.affected_nodes is None       # cache must be cleared
    print("  affected_nodes cache invalidation ✓")

    # ── 13. dict base is auto-wrapped in GraphInput ────────────────────────
    i = Intervention({"x": 1, "y": 2}, cache_base_results=True)
    assert isinstance(i.base, GraphInput)
    assert i.base.cache_results is True
    print("  dict base auto-wrapping ✓")

    # ── 14. make_batched classmethod ──────────────────────────────────────
    batched_base = GraphInput.make_batched(
        {"x": torch.randn(4, 10)},
        keys=["k0", "k1", "k2", "k3"],
    )
    bi = Intervention.make_batched(
        batched_base,
        keys=["k0", "k1", "k2", "k3"],
        intervention={"hidden": torch.randn(4, 10)},
    )
    assert bi.batched is True
    assert bi.keys == ["k0", "k1", "k2", "k3"]
    print("  make_batched ✓")

    # ── 15. CUDA: device= moves intervention tensors (GPU only) ───────────
    if torch.cuda.is_available():
        i = Intervention(
            base_input.to(torch.device("cuda")),
            intervention={"node": torch.randn(10)},
            device=torch.device("cuda"),
        )
        assert all(v.is_cuda for v in i.intervention.values.values())
        print("  device= (CUDA) ✓")
    else:
        print("  (CUDA not available — skipping GPU test)")

    print("\n✓  All Intervention tests passed")


if __name__ == "__main__":
    _run_tests()
