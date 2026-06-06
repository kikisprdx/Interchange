
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import torch

from causal_abstraction.mqnli_interchange import (
    MQNLIInterchangeCSVWriter,
    MQNLIInterchangeRunner,
    make_mqnli_mappings,
)
from compgraphs.mqnli_bert import MQNLI_Bert_CompGraph, Abstr_MQNLI_Bert_CompGraph
from compgraphs.mqnli_logic import Abstr_MQNLI_Logic_CompGraph
from modeling.arithmetic_bert import ArithmeticBertModule

DEFAULT_HIGH_NODES = [
    "sentence_q", "subj_adj", "subj_noun", "neg", "v_adv", "v_verb", "vp_q",
    "obj_adj", "obj_noun", "obj", "vp", "v_bar", "negp", "subj",
]

# BERT token positions for [CLS] + 12 premise + [SEP] + 12 hypothesis + [SEP].
MQNLI_BERT_TOKEN_LOCS = {
    "sentence_q": [0, 1, 2, 13, 14, 15, 26],
    "subj_adj":  [0, 3, 13, 16, 26],
    "subj_noun": [0, 4, 13, 17, 26],
    "neg":       [0, 5, 6, 13, 18, 19, 26],
    "v_adv":     [0, 7, 13, 20, 26],
    "v_verb":    [0, 8, 13, 21, 26],
    "vp_q":      [0, 9, 10, 13, 22, 23, 26],
    "obj_adj":   [0, 11, 13, 24, 26],
    "obj_noun":  [0, 12, 13, 25, 26],
    "obj":       [0, 11, 12, 13, 24, 25, 26],
    "vp":        [0, 8, 9, 10, 13, 21, 22, 23, 26],
    "v_bar":     [0, 7, 8, 13, 20, 21, 26],
    "negp":      [0, 5, 6, 13, 18, 19, 26],
    "subj":      [0, 3, 4, 13, 16, 17, 26],
}


def _torch_load(path: str, map_location: torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_data_and_dataset(data_path: str, split: str = "dev"):
    data = _torch_load(data_path, map_location=torch.device("cpu"))
    if hasattr(data, split):
        dataset = getattr(data, split)
    else:
        dataset = data
    return data, dataset


def max_input_id_in_dataset(dataset: Any) -> int:
    if hasattr(dataset, "tensors"):
        return int(dataset.tensors[0].max().item())
    max_id = -1
    n = min(len(dataset), 5000)
    for i in range(n):
        max_id = max(max_id, int(dataset[i][0].max().item()))
    return max_id


def infer_vocab_size(data: Any, dataset: Any, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> int:
    sizes = [max_input_id_in_dataset(dataset) + 1]
    if hasattr(data, "tokenizer"):
        sizes.append(len(data.tokenizer))
    if state_dict is not None:
        for key in ["bert.embeddings.word_embeddings.weight", "module.bert.embeddings.word_embeddings.weight"]:
            if key in state_dict:
                sizes.append(int(state_dict[key].shape[0]))
    return max(sizes)


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        if new_key.startswith("classifier."):
            new_key = "logits." + new_key[len("classifier."):]
        cleaned[new_key] = value
    return cleaned


def load_mqnli_bert_model(model_path: str, data: Any, dataset: Any, device: torch.device) -> torch.nn.Module:
    obj = _torch_load(model_path, map_location=device)
    if isinstance(obj, torch.nn.Module):
        model = obj.module if hasattr(obj, "module") and isinstance(obj.module, torch.nn.Module) else obj
        model.to(device)
        model.eval()
        return model

    if isinstance(obj, dict) and isinstance(obj.get("model"), torch.nn.Module):
        model = obj["model"]
        model.to(device)
        model.eval()
        return model

    if isinstance(obj, dict) and isinstance(obj.get("module"), torch.nn.Module):
        model = obj["module"]
        model.to(device)
        model.eval()
        return model

    if isinstance(obj, dict) and "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        state_dict = obj
    else:
        raise ValueError("model_path must contain a full model or a state_dict/checkpoint dictionary.")

    state_dict = clean_state_dict_keys(state_dict)
    vocab_size = infer_vocab_size(data, dataset, state_dict)
    model = ArithmeticBertModule(num_labels=3)
    model.bert.resize_token_embeddings(vocab_size)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("Warning: missing keys while loading MQNLI BERT:", missing[:20])
    if unexpected:
        print("Warning: unexpected keys while loading MQNLI BERT:", unexpected[:20])
    model.task = "mqnli"
    model.to(device)
    model.eval()
    return model


def num_bert_layers(model: torch.nn.Module) -> int:
    return len(model.bert.encoder.layer)


def parse_layers(layer_args: Optional[Sequence[int]], model: torch.nn.Module, include_final_layer: bool) -> List[int]:
    if layer_args:
        return list(layer_args)
    n = num_bert_layers(model)
    return list(range(n if include_final_layer else n - 1))


def output_path_from_args(args: argparse.Namespace, high_nodes: Sequence[str], layers: Sequence[int]) -> str:
    if args.output_csv:
        return args.output_csv
    os.makedirs(args.res_save_dir or "results", exist_ok=True)
    time_str = datetime.now().strftime("%m%d-%H%M%S")
    high_node = json.loads(args.abstraction)[0] if args.abstraction else "-".join(high_nodes)
    layer_str = "-".join(str(x) for x in layers[:5]) + ("-etc" if len(layers) > 5 else "")
    run_id = f"id{args.id}-" if args.id is not None else ""
    return os.path.join(args.res_save_dir or "results", f"mqnli-interventions-{run_id}{high_node}-layers-{layer_str}-{time_str}.csv")


def update_manager_csv(csv_path: str, row_id: int, updates: Dict[str, Any]) -> None:
    if not csv_path or row_id is None:
        return
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for key in updates:
        if key not in fieldnames:
            fieldnames.append(key)
    for row in rows:
        if int(row.get("id", -1)) == int(row_id):
            row.update({k: str(v) for k, v in updates.items()})
            break
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    data, dataset = load_data_and_dataset(args.data_path, split=args.split)
    model = load_mqnli_bert_model(args.model_path, data, dataset, device=device)

    if args.abstraction:
        high_node, low_nodes = json.loads(args.abstraction)
        high_nodes = [high_node]
        layers = [int(low_node.rsplit("_", 1)[-1]) for low_node in low_nodes]
    else:
        high_nodes = list(args.high_nodes or DEFAULT_HIGH_NODES)
        layers = parse_layers(args.layers, model, args.include_final_layer)

    output_csv = output_path_from_args(args, high_nodes, layers)
    base_low = MQNLI_Bert_CompGraph(model)

    writer = MQNLIInterchangeCSVWriter(output_csv)
    total_rows = 0
    mapping_id = 0
    try:
        for high_node in high_nodes:
            target_locs = list(args.target_locs) if args.target_locs else MQNLI_BERT_TOKEN_LOCS[high_node]
            all_low_nodes = [f"bert_layer_{layer}" for layer in layers]
            low_model = Abstr_MQNLI_Bert_CompGraph(
                base_low,
                all_low_nodes,
                interv_info={"target_locs": target_locs},
                root_output_device=torch.device("cpu"),
            )
            low_model.set_cache_device(torch.device("cpu"))
            high_model = Abstr_MQNLI_Logic_CompGraph(data, [high_node], root_output_device=torch.device("cpu"))
            runner = MQNLIInterchangeRunner(
                low_model=low_model,
                high_model=high_model,
                dataset=dataset,
                device=device,
                num_inputs=args.num_inputs,
                batch_size=args.batch_size,
            )
            for layer in layers:
                low_node = f"bert_layer_{layer}"
                for mapping in make_mqnli_mappings(high_node, low_node, low_model):
                    rows = runner.run_mapping(mapping_id, mapping, writer)
                    total_rows += rows
                    mapping_id += 1
    finally:
        writer.close()

    result = {
        "save_path": output_csv,
        "num_inputs": args.num_inputs,
        "num_mappings": mapping_id,
        "total_rows": total_rows,
    }
    update_manager_csv(args.csv_path, args.id, {**result, "status": 2})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--res_save_dir", default="results")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_inputs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--high_nodes", nargs="+", default=DEFAULT_HIGH_NODES)
    parser.add_argument("--layers", type=int, nargs="*")
    parser.add_argument("--include_final_layer", action="store_true")
    parser.add_argument("--target_locs", type=int, nargs="*")

    # Manager-compatible args.
    parser.add_argument("--id", type=int)
    parser.add_argument("--csv_path", default="")
    parser.add_argument("--abstraction", default="")
    parser.add_argument("--model_type", default="bert")
    parser.add_argument("--graph_alpha", default="")
    parser.add_argument("--interchange_batch_size", type=int, default=None)
    parser.add_argument("--loc_mapping_type", default="")
    parser.add_argument("--save_intermediate_results", default="True")
    args = parser.parse_args()
    if args.interchange_batch_size is not None:
        args.batch_size = args.interchange_batch_size
    result = run(args)
    print(result)


if __name__ == "__main__":
    main()
