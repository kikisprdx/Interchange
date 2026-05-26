import pytest
from compgraphs.abstractable import AbstractableCompGraph
from intervention.graph_node import GraphNode

# 1. Test Topological Ordering
graph_1 = {"A": ["B", "C"],
           "B": ["E"],
           "C": ["D"],
           "D": ["E"],
           "E": []}
expected_1 = ["A", "B", "C", "D", "E"]

graph_2 = {"A": ["B", "C", "D"],
           "B": ["E"],
           "C": ["E", "F"],
           "D": ["F"],
           "E": ["G", "H"],
           "F": ["H"],
           "G": [],
           "H": []}

expected_2 = ["A", "B", "C", "E", "G", "D", "F", "H"]


test_set = [(graph_1, "A", expected_1),
            (graph_2, "A", expected_2)]

@pytest.mark.parametrize("full_graph,root,expected", test_set)
def test_find_topological_order(full_graph, root, expected):
    order = AbstractableCompGraph.find_topological_order(full_graph, root)
    assert order == expected, f"expected {expected}, got {order}"

# 2. Test Abstract Graph Generation
def test_abstract_graph_generation():
    # Simple diamond graph: A -> B, C; B -> D; C -> D; D (leaf)
    full_graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D"],
        "D": []
    }
    
    def forward_A(b, c): return b + c
    def forward_B(d): return d * 2
    def forward_C(d): return d + 5
    
    forward_functions = {
        "A": forward_A,
        "B": forward_B,
        "C": forward_C
    }
    
    # We want to abstract away B and C, so A should depend directly on D
    abstract_nodes = ["A", "D"] # root and leaf always included by get_node_names internally
    
    abstr_graph = AbstractableCompGraph(
        full_graph=full_graph,
        root_node_name="A",
        abstract_nodes=["A"], # D and A are implicitly included
        forward_functions=forward_functions
    )
    
    # Structure check
    assert "A" in abstr_graph.nodes
    assert "D" in abstr_graph.nodes["A"].children_dict
    assert len(abstr_graph.nodes["A"].children) == 1
    
    # Computation check
    # In full graph: D=1 -> B=2, C=6 -> A=8
    # In abstract graph: A(D) should return 8
    from intervention.graph_input import GraphInput
    result = abstr_graph.compute(GraphInput({"D": 1}))
    assert result == 8

if __name__ == "__main__":
    pytest.main([__file__])
