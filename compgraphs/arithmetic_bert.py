from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

import torch

from compgraphs.abstractable import AbstractableCompGraph
from intervention import LOC, ComputationGraph, GraphNode


class Arithmetic_Bert_CompGraph(ComputationGraph):
    """A computation graph for the BERT model on the Arithmetic task.

    This class constructs the DAG representing the forward pass of a BERT model
    trained on simple arithmetic expressions ([CLS] x op y [SEP]), allowing for
    interventions at various stages.
    """

    def __init__(
        self, bert_model: Any, root_output_device: Optional[torch.device] = None
    ):
        """Initialize the Arithmetic Bert computation graph.

        :param bert_model: The BERT model instance (fine-tuned on arithmetic data).
        :param root_output_device: Optional device for the root output.
        """
        # We assume the model follows the same structure as the MQNLI BERT model
        self.model = bert_model
        bert = self.model.bert

        # 1. Define Input Node
        node_input = GraphNode(name="input", forward=self._input_forward)

        # 2. Define MetaInfo Node
        node_metainfo = GraphNode(
            node_input, name="metainfo", forward=self._metainfo_forward
        )

        # 3. Define Embeddings Node
        node_embed = GraphNode(node_input, name="embed", forward=self._embed_forward)

        # 4. Define BERT Layers
        hidden_node = node_embed
        for i, layer_module in enumerate(bert.encoder.layer):
            hidden_node = GraphNode(
                hidden_node,
                node_metainfo,
                name=f"bert_layer_{i}",
                forward=self._generate_bert_layer_forward(layer_module, i),
            )

        # 5. Define Pooler Node
        node_pool = GraphNode(hidden_node, name="pool", forward=self._pool_forward)

        # 6. Define Logits Node
        node_logits = GraphNode(node_pool, name="logits", forward=self._logits_forward)

        # 7. Define Root (Argmax) Node
        # The arithmetic task labels are positive (0), negative (1), zero (2)
        node_root = GraphNode(node_logits, name="root", forward=self._root_forward)

        super(Arithmetic_Bert_CompGraph, self).__init__(
            node_root, root_output_device=root_output_device
        )

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return self.model.device

    def _input_forward(self, x: Any) -> Any:
        """Identity forward function for the input node."""
        return x

    def _metainfo_forward(self, input_tuple: tuple) -> Dict[str, Any]:
        """Generate extended attention masks and other metadata.

        The Arithmetic dataset returns (input_ids, token_type_ids, attention_mask, label).
        """
        bert = self.model.bert
        input_ids, _, attention_mask = input_tuple[:3]
        input_shape = input_ids.shape
        device = input_ids.device

        extended_attention_mask = bert.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        return {
            "attention_mask": extended_attention_mask,
            "head_mask": [None] * 12,
            "encoder_hidden_states": None,
            "encoder_extended_attention_mask": None,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": False,
        }

    def _embed_forward(self, input_tuple: tuple) -> torch.Tensor:
        """Compute BERT embeddings."""
        input_ids, token_type_ids = input_tuple[:2]
        return self.model.bert.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )

    def _generate_bert_layer_forward(
        self, layer_module: torch.nn.Module, layer_idx: int
    ) -> Callable:
        """Generate the forward function for a specific BERT layer."""

        def _bert_layer_forward(
            hidden_states: torch.Tensor, metainfo: Dict[str, Any]
        ) -> torch.Tensor:
            head_mask = metainfo.get("head_mask")
            layer_head_mask = head_mask[layer_idx] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=metainfo.get("attention_mask"),
                head_mask=layer_head_mask,
                encoder_hidden_states=metainfo.get("encoder_hidden_states"),
                encoder_attention_mask=metainfo.get("encoder_extended_attention_mask"),
                output_attentions=metainfo.get("output_attentions"),
            )
            return layer_outputs[0]

        return _bert_layer_forward

    def _pool_forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply BERT pooler."""
        return self.model.bert.pooler(h)

    def _logits_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute final logits."""
        return self.model.logits(x)

    def _root_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the prediction using argmax."""
        return torch.argmax(x, dim=1)


class Abstr_Arithmetic_Bert_CompGraph(AbstractableCompGraph):
    """An abstractable version of the Arithmetic BERT computation graph."""

    def __init__(
        self,
        base_compgraph: Arithmetic_Bert_CompGraph,
        intermediate_nodes: List[str],
        interv_info: Any = None,
        root_output_device: Optional[torch.device] = None,
    ):
        """Initialize the abstractable Arithmetic BERT graph.

        :param base_compgraph: The full Arithmetic BERT graph.
        :param intermediate_nodes: Nodes to keep in the abstract graph.
        :param interv_info: Optional intervention metadata (e.g., target locations).
        :param root_output_device: Optional device for the root output.
        """
        self.base = base_compgraph
        self.interv_info = interv_info

        full_graph = {
            node_name: [child.name for child in node.children]
            for node_name, node in base_compgraph.nodes.items()
        }

        forward_functions = {
            node_name: node.forward for node_name, node in base_compgraph.nodes.items()
        }

        super(Abstr_Arithmetic_Bert_CompGraph, self).__init__(
            full_graph=full_graph,
            root_node_name="root",
            abstract_nodes=intermediate_nodes,
            forward_functions=forward_functions,
            root_output_device=root_output_device,
        )

    @property
    def device(self) -> torch.device:
        """Get the device the base model is on."""
        return self.base.device

    def get_indices(self, node: str) -> List[Any]:
        """Get the indices for intervention for a given node.

        Usually used to target specific tokens (e.g., x, op, or y).
        """
        if re.match(r".*bert_layer_[0-9]*", node):
            # Target specific locations if provided in interv_info
            if self.interv_info and "target_locs" in self.interv_info:
                return [LOC[:, i, :] for i in self.interv_info["target_locs"]]
            return [LOC[:, :, :]]  # Default to whole tensor
        else:
            raise ValueError(f"Cannot get indices for node {node}")
