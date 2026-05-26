from __future__ import annotations
from intervention import ComputationGraph
from intervention import GraphNode

from typing import Any, Dict, List, Callable, Set, Optional, Union
import torch


class AbstractableCompGraph(ComputationGraph):
    """An abstractable computation graph structure.
    
    This class allows for the generation of an abstracted computation graph from a full graph
    specification, keeping only designated intermediate nodes and abstracting away others.
    """

    def __init__(
        self,
        full_graph: Dict[str, List[str]],
        root_node_name: str,
        abstract_nodes: List[str],
        forward_functions: Dict[str, Callable],
        topological_order: Optional[List[str]] = None,
        root_output_device: Optional[torch.device] = None
    ):
        """Initialize an AbstractableCompGraph.

        :param full_graph: A dict describing the structure of a computation
            graph, mapping name of each parent node to list of names of children.
        :param root_node_name: The name of the root node.
        :param abstract_nodes: Names of intermediate nodes that we would like
            to keep, while abstracting away all other nodes.
        :param forward_functions: Forward function for each node in full_graph.
        :param topological_order: Topological ordering of nodes.
        :param root_output_device: Optional device for the root output.
        """
        self.full_graph = full_graph
        self.input_node_names = {k for k, v in full_graph.items() if len(v) == 0}
        self.root_node_name = root_node_name
        self.forward_functions = forward_functions
        self.topological_order = topological_order or self.find_topological_order(full_graph, root_node_name)

        self.validate_full_graph()

        root = self.generate_abstract_graph(abstract_nodes)
        super(AbstractableCompGraph, self).__init__(root, root_output_device=root_output_device)

    def validate_full_graph(self):
        """Validate that the provided full graph is consistent."""
        all_child_nodes = set(n for nodes in self.full_graph.values() for n in nodes)
        all_parent_nodes = set(self.full_graph.keys())
        
        missing_nodes = all_child_nodes - all_parent_nodes
        if missing_nodes:
            raise RuntimeError(f"All nodes in the underlying graph should be in the keys of the dict. Missing: {missing_nodes}")

        # Each non-leaf node should have a forward function
        missing_forward = (all_child_nodes - self.input_node_names) - set(self.forward_functions.keys())
        if missing_forward:
            raise RuntimeError(f"These nodes are missing an associated forward function: {missing_forward}")

    @staticmethod
    def find_topological_order(full_graph: Dict[str, List[str]], root: str) -> List[str]:
        """Find a topological ordering of nodes in the graph starting from root."""
        ordering: List[str] = []
        visited: Set[str] = set()

        def recursive_call(node: str):
            visited.add(node)
            children = full_graph.get(node, [])
            for i in range(len(children) - 1, -1, -1):
                child = children[i]
                if child not in visited:
                    recursive_call(child)
            ordering.append(node)

        recursive_call(root)
        return ordering[::-1]

    def generate_abstract_graph(self, abstract_nodes: List[str]) -> GraphNode:
        """Generate the abstracted computation graph."""
        relevant_nodes = self.get_node_names(abstract_nodes)
        relevant_node_set = set(relevant_nodes)

        # Define input leaf nodes
        node_dict = {name: self._generate_input_node(name) for name in self.input_node_names}

        for node_name in relevant_nodes:
            if node_name in self.input_node_names:
                continue

            curr_children = self.get_children(node_name, relevant_node_set)
            args = [node_dict[child] for child in curr_children]
            forward = self.generate_forward_function(node_name, curr_children)
            node_dict[node_name] = GraphNode(*args, name=node_name, forward=forward)

        return node_dict[self.root_node_name]

    def get_node_names(self, abstract_nodes: List[str]) -> List[str]:
        """Get topologically ordered list of node names in final compgraph, given intermediate nodes."""
        nodes_to_keep = set(abstract_nodes)
        nodes_to_keep.add(self.root_node_name)
        nodes_to_keep.update(self.input_node_names)

        return [name for name in reversed(self.topological_order) if name in nodes_to_keep]

    def get_children(self, abstract_node: str, abstract_node_set: Set[str]) -> List[str]:
        """Get immediate children in abstracted graph given an abstract node."""
        res: List[str] = []
        stack = [abstract_node]
        visited: Set[str] = set()

        while stack:
            curr_node = stack.pop()
            visited.add(curr_node)
            
            if curr_node != abstract_node and curr_node in abstract_node_set:
                res.append(curr_node)
            else:
                children = self.full_graph.get(curr_node, [])
                for i in range(len(children) - 1, -1, -1):
                    child = children[i]
                    if child not in visited:
                        stack.append(child)
        return res

    def _generate_input_node(self, name: str) -> GraphNode:
        """Helper to create a leaf input node."""
        def _input_forward_fxn(x):
            return x
        return GraphNode(name=name, forward=_input_forward_fxn, cache_results=False)

    def generate_forward_function(self, abstracted_node: str, children: List[str]) -> Callable:
        """Generate a forward function to construct an abstract node."""
        child_name_to_idx = {name: i for i, name in enumerate(children)}

        def _forward(*args):
            if len(args) != len(children):
                raise ValueError(f"Got {len(args)} arguments to forward fxn of {abstracted_node}, expected {len(children)}")

            def _implicit_call(node_name: str) -> Any:
                if node_name in child_name_to_idx:
                    return args[child_name_to_idx[node_name]]
                
                # Recursive call for intermediate nodes that were abstracted away
                child_args = [_implicit_call(child) for child in self.full_graph[node_name]]
                current_f = self.forward_functions[node_name]
                return current_f(*child_args)

            return _implicit_call(abstracted_node)

        return _forward
