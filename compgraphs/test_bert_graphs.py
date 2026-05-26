import torch
import pytest
from unittest.mock import MagicMock
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Callable

# Import the refactored classes
from compgraphs.mqnli_bert import MQNLI_Bert_CompGraph, Abstr_MQNLI_Bert_CompGraph
from compgraphs.arithmetic_bert import Arithmetic_Bert_CompGraph, Abstr_Arithmetic_Bert_CompGraph
from intervention.graph_input import GraphInput

# --- Mocking Infrastructure ---

class MockBertModel(torch.nn.Module):
    def __init__(self, task="mqnli", num_layers=2):
        super().__init__()
        self.task = task
        self.device = torch.device("cpu")
        self.bert = MagicMock()
        
        # Mocking layers to return a tuple containing the input hidden states (to preserve batch size)
        def mock_layer_forward(hidden_states, *args, **kwargs):
            return (hidden_states, None)
        
        self.bert.encoder.layer = []
        for _ in range(num_layers):
            layer = MagicMock()
            layer.side_effect = mock_layer_forward
            self.bert.encoder.layer.append(layer)
            
        # Mocking sub-modules and their forward passes
        def mock_embeddings(input_ids, **kwargs):
            return torch.randn(input_ids.shape[0], 5, 768)
        self.bert.embeddings.side_effect = mock_embeddings
        
        self.bert.pooler = MagicMock(side_effect=lambda h: torch.randn(h.shape[0], 768))
        self.bert.get_extended_attention_mask = MagicMock(side_effect=lambda mask, shape, device: torch.ones(shape[0], 1, 1, shape[1]))
        self.logits = MagicMock(side_effect=lambda x: torch.randn(x.shape[0], 3))

class MockArithmeticDataset(Dataset):
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
    def __len__(self): return self.num_samples
    def __getitem__(self, i):
        # (ids, token_types, masks, label)
        return (torch.randint(0, 100, (5,)), 
                torch.zeros(5, dtype=torch.long), 
                torch.ones(5), 
                0)

# --- Fixtures ---

@pytest.fixture
def mock_model():
    return MockBertModel(num_layers=4)

@pytest.fixture
def mqnli_graph(mock_model):
    return MQNLI_Bert_CompGraph(mock_model)

@pytest.fixture
def arithmetic_graph(mock_model):
    return Arithmetic_Bert_CompGraph(mock_model)

# --- Tests for MQNLI BERT ---

def test_mqnli_bert_structure(mqnli_graph):
    """Verify the DAG structure of MQNLI_Bert_CompGraph."""
    nodes = mqnli_graph.nodes
    assert "root" in nodes
    assert "logits" in nodes["root"].children_dict
    assert "pool" in nodes["logits"].children_dict
    assert "bert_layer_3" in nodes["pool"].children_dict
    assert "bert_layer_0" in nodes["bert_layer_1"].children_dict
    assert "embed" in nodes["bert_layer_0"].children_dict
    assert "metainfo" in nodes["bert_layer_0"].children_dict
    assert "input" in nodes["embed"].children_dict

def test_mqnli_bert_forward_batch(mqnli_graph):
    """Verify forward pass with a batch of inputs."""
    batch_size = 4
    input_ids = torch.randint(0, 100, (batch_size, 10))
    token_type_ids = torch.zeros(batch_size, 10, dtype=torch.long)
    attention_mask = torch.ones(batch_size, 10)
    
    graph_input = GraphInput({"input": (input_ids, token_type_ids, attention_mask)})
    result = mqnli_graph.compute(graph_input)
    
    assert isinstance(result, torch.Tensor)
    assert result.shape == (batch_size,)

def test_mqnli_bert_abstractable(mqnli_graph):
    """Test abstraction of the MQNLI BERT graph."""
    abstract_nodes = ["bert_layer_1", "pool"]
    abstr_graph = Abstr_MQNLI_Bert_CompGraph(mqnli_graph, abstract_nodes)
    
    assert "root" in abstr_graph.nodes
    assert "pool" in abstr_graph.nodes["root"].children_dict
    assert "bert_layer_1" in abstr_graph.nodes["pool"].children_dict

# --- Tests for Arithmetic BERT ---

def test_arithmetic_bert_structure(arithmetic_graph):
    """Verify the DAG structure of Arithmetic_Bert_CompGraph."""
    nodes = arithmetic_graph.nodes
    assert "root" in nodes
    assert "bert_layer_0" in nodes.keys()
    assert "embed" in nodes["bert_layer_0"].children_dict

def test_arithmetic_bert_forward_dataset(arithmetic_graph):
    """Verify forward pass using a mock arithmetic dataset."""
    dataset = MockArithmeticDataset(num_samples=8)
    dataloader = DataLoader(dataset, batch_size=4)
    
    for input_tuple in dataloader:
        graph_input = GraphInput({"input": input_tuple})
        result = arithmetic_graph.compute(graph_input)
        assert result.shape == (4,)
        break

def test_arithmetic_bert_indices(arithmetic_graph):
    """Verify get_indices logic in the abstractable version."""
    abstr_graph = Abstr_Arithmetic_Bert_CompGraph(
        arithmetic_graph, 
        ["bert_layer_0"], 
        interv_info={"target_locs": [1, 3]}
    )
    
    indices = abstr_graph.get_indices("bert_layer_0")
    assert len(indices) == 2
    from intervention import LOC
    assert indices[0] == (slice(None), 1, slice(None))

if __name__ == "__main__":
    pytest.main([__file__])
