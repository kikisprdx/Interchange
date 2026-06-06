from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch

from compgraphs.abstractable import AbstractableCompGraph
from pipeline import ComputationGraph, GraphNode

INDEP, EQUIV, ENTAIL, REV_ENTAIL, CONTRADICT, ALTER, COVER = range(7)
IDX_Q_S, IDX_A_S, IDX_N_S, IDX_NEG, IDX_ADV, IDX_V, IDX_Q_O, IDX_A_O, IDX_N_O = range(9)
SOME, EVERY, NO, NOTEVERY = range(4)


def _as_raw_matrix(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.long)
    if x.dim() == 1:
        if x.numel() != 18:
            raise ValueError(f"Expected raw MQNLI input length 18, got {x.numel()}.")
        x = x.unsqueeze(0)
    if x.dim() != 2:
        raise ValueError(f"Expected raw MQNLI input with shape [batch, 18], got {tuple(x.shape)}.")
    if x.shape[1] == 18:
        return x.long()
    if x.shape[0] == 18:
        return x.T.contiguous().long()
    raise ValueError(f"Expected raw MQNLI input shape [batch,18] or [18,batch], got {tuple(x.shape)}.")


def get_relation_composition() -> torch.Tensor:
    composition_dict = {}
    rels = [EQUIV, ENTAIL, REV_ENTAIL, CONTRADICT, COVER, ALTER, INDEP]
    for r1 in rels:
        for r2 in rels:
            composition_dict[(r1, r2)] = INDEP
    for r in rels:
        composition_dict[(EQUIV, r)] = r
        composition_dict[(r, EQUIV)] = r
    composition_dict[(ENTAIL, ENTAIL)] = ENTAIL
    composition_dict[(ENTAIL, CONTRADICT)] = ALTER
    composition_dict[(ENTAIL, ALTER)] = ALTER
    composition_dict[(REV_ENTAIL, REV_ENTAIL)] = REV_ENTAIL
    composition_dict[(REV_ENTAIL, CONTRADICT)] = COVER
    composition_dict[(REV_ENTAIL, COVER)] = COVER
    composition_dict[(CONTRADICT, ENTAIL)] = COVER
    composition_dict[(CONTRADICT, REV_ENTAIL)] = ALTER
    composition_dict[(CONTRADICT, CONTRADICT)] = EQUIV
    composition_dict[(CONTRADICT, COVER)] = REV_ENTAIL
    composition_dict[(CONTRADICT, ALTER)] = ENTAIL
    composition_dict[(ALTER, REV_ENTAIL)] = ALTER
    composition_dict[(ALTER, CONTRADICT)] = ENTAIL
    composition_dict[(ALTER, COVER)] = ENTAIL
    composition_dict[(COVER, ENTAIL)] = COVER
    composition_dict[(COVER, CONTRADICT)] = REV_ENTAIL
    composition_dict[(COVER, ALTER)] = REV_ENTAIL

    res = torch.zeros(7, 7, dtype=torch.long)
    for (r1, r2), v in composition_dict.items():
        res[r1, r2] = v
    return res


relation_composition = get_relation_composition()


def strong_composition(signature1: Dict[int, int], signature2: Dict[int, int], relation1: int, relation2: int) -> int:
    composition1 = relation_composition[signature1[relation1], signature2[relation2]].item()
    composition2 = relation_composition[signature2[relation2], signature1[relation1]].item()
    if composition1 == INDEP:
        return composition2
    return composition1


def get_negation_signatures() -> torch.Tensor:
    res = torch.zeros(4, 7, dtype=torch.long)
    res[0] = torch.tensor([INDEP, EQUIV, ENTAIL, REV_ENTAIL, CONTRADICT, ALTER, COVER], dtype=torch.long)
    res[3] = torch.tensor([INDEP, EQUIV, REV_ENTAIL, ENTAIL, CONTRADICT, COVER, ALTER], dtype=torch.long)
    for rel in range(7):
        res[1, rel] = relation_composition[rel, CONTRADICT]
    for rel in range(7):
        res[2, rel] = res[1, res[3, rel]]
    return res


negation_signatures = get_negation_signatures()


def get_quantifier_signatures() -> torch.Tensor:
    sigs_dict: Dict[Any, Dict[Any, int]] = {}
    symmetric_relation = {
        EQUIV: EQUIV,
        ENTAIL: REV_ENTAIL,
        REV_ENTAIL: ENTAIL,
        CONTRADICT: CONTRADICT,
        COVER: COVER,
        ALTER: ALTER,
        INDEP: INDEP,
    }

    sigs_dict[(SOME, SOME)] = (
        {EQUIV: EQUIV, ENTAIL: ENTAIL, REV_ENTAIL: REV_ENTAIL, INDEP: INDEP},
        {EQUIV: EQUIV, ENTAIL: ENTAIL, REV_ENTAIL: REV_ENTAIL, CONTRADICT: COVER, COVER: COVER, ALTER: INDEP, INDEP: INDEP},
    )
    sigs_dict[(EVERY, EVERY)] = (
        {EQUIV: EQUIV, ENTAIL: REV_ENTAIL, REV_ENTAIL: ENTAIL, INDEP: INDEP},
        {EQUIV: EQUIV, ENTAIL: ENTAIL, REV_ENTAIL: REV_ENTAIL, CONTRADICT: ALTER, COVER: INDEP, ALTER: ALTER, INDEP: INDEP},
    )
    for key in list(sigs_dict.keys()):
        signature1, signature2 = sigs_dict[key]
        new_signature = {}
        for key1 in signature1:
            for key2 in signature2:
                new_signature[(key1, key2)] = strong_composition(signature1, signature2, key1, key2)
        sigs_dict[key] = new_signature

    new_signature = {}
    relations = [EQUIV, ENTAIL, REV_ENTAIL, CONTRADICT, COVER, ALTER, INDEP]
    for relation1 in [EQUIV, ENTAIL, REV_ENTAIL, INDEP]:
        for relation2 in relations:
            if (relation2 == EQUIV or relation2 == REV_ENTAIL) and relation1 != INDEP:
                new_signature[(relation1, relation2)] = REV_ENTAIL
            else:
                new_signature[(relation1, relation2)] = INDEP
    sigs_dict[(SOME, EVERY)] = new_signature
    sigs_dict[(SOME, EVERY)][(ENTAIL, CONTRADICT)] = ALTER
    sigs_dict[(SOME, EVERY)][(ENTAIL, ALTER)] = ALTER
    sigs_dict[(SOME, EVERY)][(EQUIV, ALTER)] = ALTER
    sigs_dict[(SOME, EVERY)][(EQUIV, CONTRADICT)] = CONTRADICT
    sigs_dict[(SOME, EVERY)][(EQUIV, COVER)] = COVER
    sigs_dict[(SOME, EVERY)][(REV_ENTAIL, COVER)] = COVER
    sigs_dict[(SOME, EVERY)][(REV_ENTAIL, CONTRADICT)] = COVER

    new_signature = {}
    for key in sigs_dict[(SOME, EVERY)]:
        new_signature[(symmetric_relation[key[0]], symmetric_relation[key[1]])] = symmetric_relation[sigs_dict[(SOME, EVERY)][key]]
    sigs_dict[(EVERY, SOME)] = new_signature

    res = torch.zeros(4 * 4, 4 * 7, dtype=torch.long)
    for neg_1 in [0, 1]:
        for neg_2 in [0, 1]:
            for (q1, q2), d in sigs_dict.items():
                for (r1, r2), v in d.items():
                    neg_sig = negation_signatures[neg_1 * 2 + neg_2]
                    res[4 * (neg_1 * 2 + q1) + (neg_2 * 2 + q2), 7 * r1 + r2] = neg_sig[v]
    return res


quantifier_signatures = get_quantifier_signatures()
output_remapping = torch.tensor([0, 1, 1, 0, 2, 2, 0], dtype=torch.long)

compgraph_structure = {
    "input": [],
    "get_p": ["input"],
    "get_h": ["input"],
    "obj_noun": ["get_p", "get_h"],
    "obj_adj": ["get_p", "get_h"],
    "obj": ["obj_adj", "obj_noun"],
    "vp_q": ["get_p", "get_h"],
    "v_verb": ["get_p", "get_h"],
    "v_adv": ["get_p", "get_h"],
    "v_bar": ["v_adv", "v_verb"],
    "vp": ["v_bar", "vp_q", "obj"],
    "neg": ["get_p", "get_h"],
    "negp": ["neg", "vp"],
    "subj_noun": ["get_p", "get_h"],
    "subj_adj": ["get_p", "get_h"],
    "subj": ["subj_adj", "subj_noun"],
    "sentence_q": ["get_p", "get_h"],
    "root": ["sentence_q", "subj", "negp"],
}


class MQNLI_Logic_CompGraph(ComputationGraph):
    def __init__(self, data: Any, root_output_device: Optional[torch.device] = None):
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.keyword_dict = {w: self.word_to_id[w] for w in ["emptystring", "no", "some", "every", "notevery", "doesnot"]}

        @GraphNode()
        def input(x: torch.Tensor) -> torch.Tensor:
            return _as_raw_matrix(x)

        input.cache_results = False

        @GraphNode(input)
        def get_p(x: torch.Tensor) -> torch.Tensor:
            return x[:, :9]

        get_p.cache_results = False

        @GraphNode(input)
        def get_h(x: torch.Tensor) -> torch.Tensor:
            return x[:, 9:]

        get_h.cache_results = False

        @GraphNode(get_p, get_h)
        def obj_noun(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            return (p[:, IDX_N_O] == h[:, IDX_N_O]).type(torch.long)

        @GraphNode(get_p, get_h)
        def obj_adj(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            return self._intersective_projection(p[:, IDX_A_O], h[:, IDX_A_O])

        @GraphNode(obj_adj, obj_noun)
        def obj(a: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            return torch.gather(a, 1, n.unsqueeze(1)).view(n.shape[0])

        @GraphNode(get_p, get_h)
        def vp_q(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            return self._merge_quantifiers(p[:, IDX_Q_O], h[:, IDX_Q_O])

        @GraphNode(get_p, get_h)
        def v_verb(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            return (p[:, IDX_V] == h[:, IDX_V]).type(torch.long)

        @GraphNode(get_p, get_h)
        def v_adv(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            return self._intersective_projection(p[:, IDX_ADV], h[:, IDX_ADV])

        @GraphNode(v_adv, v_verb)
        def v_bar(a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return torch.gather(a, 1, v.unsqueeze(1)).view(v.shape[0])

        @GraphNode(v_bar, vp_q, obj)
        def vp(v: torch.Tensor, q: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
            idxs = (o * 7 + v).unsqueeze(1)
            return torch.gather(q, 1, idxs).view(v.shape[0])

        @GraphNode(get_p, get_h)
        def neg(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            return self._merge_negation(p[:, IDX_NEG], h[:, IDX_NEG])

        @GraphNode(neg, vp)
        def negp(n: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return torch.gather(n, 1, v.unsqueeze(1)).view(v.shape[0])

        @GraphNode(get_p, get_h)
        def subj_noun(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            return (p[:, IDX_N_S] == h[:, IDX_N_S]).type(torch.long)

        @GraphNode(get_p, get_h)
        def subj_adj(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            return self._intersective_projection(p[:, IDX_A_S], h[:, IDX_A_S])

        @GraphNode(subj_adj, subj_noun)
        def subj(a: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            return torch.gather(a, 1, n.unsqueeze(1)).view(n.shape[0])

        @GraphNode(get_p, get_h)
        def sentence_q(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            return self._merge_quantifiers(p[:, IDX_Q_S], h[:, IDX_Q_S])

        @GraphNode(sentence_q, subj, negp)
        def root(q: torch.Tensor, s: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            idxs = (s * 7 + n).unsqueeze(1)
            res = torch.gather(q, 1, idxs).view(n.shape[0])
            return output_remapping.to(res.device)[res]

        super().__init__(root, root_output_device=root_output_device)

    def __getattr__(self, item: str) -> int:
        if item in self.keyword_dict:
            return self.keyword_dict[item]
        raise AttributeError(item)

    def _intersective_projection(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        eq = p == h
        p_is_empty = p == self.emptystring
        h_is_empty = h == self.emptystring
        forward_entail = (~p_is_empty & h_is_empty)
        backward_entail = (p_is_empty & ~h_is_empty)
        res = torch.zeros(p.size(0), 2, dtype=torch.long, device=p.device)
        res[eq] = torch.tensor([INDEP, EQUIV], dtype=torch.long, device=p.device)
        res[forward_entail] = torch.tensor([INDEP, ENTAIL], dtype=torch.long, device=p.device)
        res[backward_entail] = torch.tensor([INDEP, REV_ENTAIL], dtype=torch.long, device=p.device)
        return res

    def _merge_quantifiers(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        p_idx_tensor = torch.zeros(p.size(0), dtype=torch.long, device=p.device)
        h_idx_tensor = torch.zeros(h.size(0), dtype=torch.long, device=p.device)
        for q_idx, q_token in enumerate([self.some, self.every, self.no, self.notevery]):
            p_idx_tensor[p == q_token] = q_idx
            h_idx_tensor[h == q_token] = q_idx
        idx_tensor = p_idx_tensor * 4 + h_idx_tensor
        return quantifier_signatures.to(p.device).index_select(0, idx_tensor)

    def _merge_negation(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        p_idx_tensor = torch.zeros(p.size(0), dtype=torch.long, device=p.device)
        h_idx_tensor = torch.zeros(h.size(0), dtype=torch.long, device=p.device)
        for q_idx, q_token in enumerate([self.emptystring, self.doesnot]):
            p_idx_tensor[p == q_token] = q_idx
            h_idx_tensor[h == q_token] = q_idx
        idx_tensor = p_idx_tensor * 2 + h_idx_tensor
        return negation_signatures.to(p.device).index_select(0, idx_tensor)


class Abstr_MQNLI_Logic_CompGraph(AbstractableCompGraph):
    def __init__(self, data: Any, intermediate_nodes: List[str], root_output_device: Optional[torch.device] = None):
        full_model = MQNLI_Logic_CompGraph(data, root_output_device=root_output_device)
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
