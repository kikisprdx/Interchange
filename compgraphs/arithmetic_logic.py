"""High-level causal graph for the arithmetic sign-classification task.

The low-level BERT graph represents the neural computation.  This file
represents the high-level symbolic computation that we want to test BERT
against with interchange interventions.

Expected high-level input format
--------------------------------
The arithmetic dataset should return a raw tensor for every example:

    raw_arithmetic_input = torch.tensor([x, op_id, y])

where op_id is 0 for '+' and 1 for '-'.  A fourth column containing the
precomputed result is allowed but ignored, because the graph recomputes the
result from x, op, and y.

The graph output labels follow the dataset convention:

    positive -> 0
    negative -> 1
    zero     -> 2
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch

from compgraphs.abstractable import AbstractableCompGraph
from intervention import ComputationGraph, GraphNode

PLUS = 0
MINUS = 1
POSITIVE = 0
NEGATIVE = 1
ZERO = 2

# The graph is written in the same parent -> children style as mqnli_logic.py.
# Edges point from a node to the nodes it depends on.
compgraph_structure: Dict[str, List[str]] = {
    "input": [],
    "x_value": ["input"],
    "op": ["input"],
    "y_value": ["input"],
    "result": ["x_value", "op", "y_value"],
    "sign": ["result"],
    "root": ["sign"],
}


def _as_batch_matrix(raw_input: torch.Tensor) -> torch.Tensor:
    """Normalize raw arithmetic input to shape [batch, features].

    The interchange runner provides [batch, 3].  This helper also accepts a
    single example [3] and converts it to [1, 3].
    """
    if raw_input.dim() == 1:
        raw_input = raw_input.unsqueeze(0)
    if raw_input.dim() != 2 or raw_input.shape[1] < 3:
        raise ValueError(
            "Arithmetic high-level input must have shape [batch, >=3] "
            "with columns [x, op_id, y]."
        )
    return raw_input.long()


class Arithmetic_Logic_CompGraph(ComputationGraph):
    """Full symbolic causal graph for arithmetic.

    The full graph exposes these high-level variables:

        input -> x_value
        input -> op
        input -> y_value
        x_value, op, y_value -> result
        result -> sign -> root

    The node names are intentionally explicit because these are the names used
    in the intervention mapping, for example x_value -> bert_layer_4[:, 1, :].
    """

    def __init__(self, root_output_device: Optional[torch.device] = None):
        @GraphNode()
        def input(x: torch.Tensor) -> torch.Tensor:
            return _as_batch_matrix(x)

        # The input node should not cache because it just normalizes input and
        # is cheap; it also avoids retaining unnecessary raw tensors.
        input.cache_results = False

        @GraphNode(input)
        def x_value(x: torch.Tensor) -> torch.Tensor:
            return x[:, 0]

        @GraphNode(input)
        def op(x: torch.Tensor) -> torch.Tensor:
            return x[:, 1]

        @GraphNode(input)
        def y_value(x: torch.Tensor) -> torch.Tensor:
            return x[:, 2]

        @GraphNode(x_value, op, y_value)
        def result(x: torch.Tensor, operator: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.where(operator == PLUS, x + y, x - y)

        @GraphNode(result)
        def sign(z: torch.Tensor) -> torch.Tensor:
            return torch.where(
                z > 0,
                torch.full_like(z, POSITIVE),
                torch.where(z < 0, torch.full_like(z, NEGATIVE), torch.full_like(z, ZERO)),
            )

        @GraphNode(sign)
        def root(label: torch.Tensor) -> torch.Tensor:
            return label

        super().__init__(root, root_output_device=root_output_device)


class Abstr_Arithmetic_Logic_CompGraph(AbstractableCompGraph):
    """Abstractable arithmetic logic graph.

    This mirrors Abstr_MQNLI_Logic_CompGraph: it keeps only `input`, `root`, and
    the requested high-level intermediate node(s).  Example:

        Abstr_Arithmetic_Logic_CompGraph(["x_value"])

    creates an abstract graph equivalent to:

        input -> x_value -> root

    where everything between x_value and root is recomputed implicitly by the
    generated forward function.
    """

    def __init__(
        self,
        intermediate_nodes: List[str],
        root_output_device: Optional[torch.device] = None,
    ):
        full_model = Arithmetic_Logic_CompGraph(root_output_device=root_output_device)

        forward_functions: Dict[str, Callable[..., Any]] = {
            node_name: node.forward for node_name, node in full_model.nodes.items()
        }

        super().__init__(
            full_graph=compgraph_structure,
            root_node_name="root",
            abstract_nodes=intermediate_nodes,
            forward_functions=forward_functions,
            root_output_device=root_output_device,
        )

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")
