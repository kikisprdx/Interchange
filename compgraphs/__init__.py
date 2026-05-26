from compgraphs.mqnli_bert import MQNLI_Bert_CompGraph, Abstr_MQNLI_Bert_CompGraph
from compgraphs.arithmetic_bert import Arithmetic_Bert_CompGraph, Abstr_Arithmetic_Bert_CompGraph

_name_to_compgraph_class = {
    "bert": MQNLI_Bert_CompGraph,
    "arithmetic_bert": Arithmetic_Bert_CompGraph,
}

def get_compgraph_class_by_name(name: str):
    return _name_to_compgraph_class[name]


_name_to_abstr_compgraph_class = {
    "bert": Abstr_MQNLI_Bert_CompGraph,
    "arithmetic_bert": Abstr_Arithmetic_Bert_CompGraph,
}

def get_abstr_compgraph_class_by_name(name: str):
    return _name_to_abstr_compgraph_class[name]
