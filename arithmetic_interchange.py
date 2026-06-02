"""Run arithmetic interchange interventions and save raw results to CSV.

This script assumes:
1. the arithmetic dataset has already been preprocessed and can be loaded with
   torch.load(...);
2. the fine-tuned arithmetic BERT model can be loaded with torch.load(...);
3. each dataset item has this format:

       (input_ids, token_type_ids, attention_mask, raw_arithmetic_input, label)

   where raw_arithmetic_input is [x, op_id, y], op_id=0 for '+', op_id=1 for '-'.

Example direct run:

    python arithmetic_interchange.py \
      --data_path data/arithmetic_preprocessed.pt \
      --model_path models/arithmetic_bert.pt \
      --output_csv results/arithmetic_interventions.csv \
      --high_nodes x_value y_value op result sign \
      --num_inputs 100

Example manager-style run:

    python arithmetic_interchange.py \
      --id 3 \
      --csv_path experiments.csv \
      --data_path data/arithmetic_preprocessed.pt \
      --model_path models/arithmetic_bert.pt \
      --abstraction '["x_value", ["bert_layer_4"]]' \
      --res_save_dir results/expt-3 \
      --num_inputs 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from causal_abstraction.arithmetic_interchange import (
    ArithmeticInterchangeRunner,
    make_arithmetic_mappings,
)
from compgraphs.arithmetic_bert import Arithmetic_Bert_CompGraph, Abstr_Arithmetic_Bert_CompGraph
from compgraphs.arithmetic_logic import Abstr_Arithmetic_Logic_CompGraph

DEFAULT_HIGH_NODES = ["x_value", "y_value", "op", "result", "sign"]
DEFAULT_TOKEN_LOCS = [0, 1, 2, 3, 4]  # [CLS], x, op, y, [SEP]


def _torch_load(path: str, map_location: torch.device) -> Any:
    """torch.load wrapper compatible with newer PyTorch defaults."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_dataset(data_path: str, split: str = "dev") -> Any:
    data = _torch_load(data_path, map_location=torch.device("cpu"))
    if hasattr(data, split):
        return getattr(data, split)
    # Allow loading a Dataset object directly.
    return data


def load_finetuned_bert_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load a fine-tuned model object.

    This supports the common cases where torch.load returns the whole model or
    a dictionary containing the model under 'model' or 'module'.  If your
    checkpoint only contains a state_dict, instantiate your model in your own
    loader and replace this function.
    """
    obj = _torch_load(model_path, map_location=device)

    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict) and isinstance(obj.get("model"), torch.nn.Module):
        model = obj["model"]
    elif isinstance(obj, dict) and isinstance(obj.get("module"), torch.nn.Module):
        model = obj["module"]
    else:
        raise ValueError(
            "Could not load a complete BERT model from model_path. This script "
            "expects torch.load(model_path) to return an nn.Module or a dict with "
            "a 'model'/'module' nn.Module. If your checkpoint only stores a "
            "state_dict, instantiate the model first and then load the state_dict."
        )

    if hasattr(model, "module") and isinstance(model.module, torch.nn.Module):
        model = model.module

    model.to(device)
    model.eval()
    return model


def num_bert_layers(model: torch.nn.Module) -> int:
    return len(model.bert.encoder.layer)


def parse_layers(layer_args: Optional[Sequence[int]], model: torch.nn.Module, include_final_layer: bool) -> List[int]:
    if layer_args:
        return list(layer_args)
    n = num_bert_layers(model)
    # Keep the paper's convention by default: do not test the final layer.
    return list(range(n if include_final_layer else n - 1))


def output_path_from_args(args: argparse.Namespace, high_nodes: Sequence[str], layers: Sequence[int]) -> str:
    if args.output_csv:
        return args.output_csv
    os.makedirs(args.res_save_dir or "results", exist_ok=True)
    time_str = datetime.now().strftime("%m%d-%H%M%S")
    if args.abstraction:
        high_node = json.loads(args.abstraction)[0]
    else:
        high_node = "-".join(high_nodes)
    layer_str = "-".join(str(x) for x in layers[:5]) + ("-etc" if len(layers) > 5 else "")
    run_id = f"id{args.id}-" if args.id is not None else ""
    return os.path.join(args.res_save_dir or "results", f"arithmetic-interventions-{run_id}{high_node}-layers-{layer_str}-{time_str}.csv")


def build_mappings(
    model: torch.nn.Module,
    high_nodes: Sequence[str],
    layers: Sequence[int],
    target_locs: Sequence[int],
) -> List[Any]:
    """Build all requested high-node/layer/token-location mappings."""
    # A temporary low graph is enough to ask the abstract low graph for LOC objects.
    base_low = Arithmetic_Bert_CompGraph(model)
    mappings = []
    for high_node in high_nodes:
        for layer in layers:
            low_node = f"bert_layer_{layer}"
            low_model_for_indices = Abstr_Arithmetic_Bert_CompGraph(
                base_low,
                [low_node],
                interv_info={"target_locs": list(target_locs)},
                root_output_device=torch.device("cpu"),
            )
            mappings.extend(make_arithmetic_mappings(high_node, low_node, low_model_for_indices))
    return mappings


def update_manager_csv(csv_path: str, row_id: int, updates: Dict[str, Any]) -> None:
    """Minimal manager update so this script can be launched from a CSV queue."""
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
    dataset = load_dataset(args.data_path, split=args.split)
    model = load_finetuned_bert_model(args.model_path, device=device)

    if args.abstraction:
        high_node, low_nodes = json.loads(args.abstraction)
        high_nodes = [high_node]
        layers = [int(low_node.rsplit("_", 1)[-1]) for low_node in low_nodes]
    else:
        high_nodes = list(args.high_nodes or DEFAULT_HIGH_NODES)
        layers = parse_layers(args.layers, model, args.include_final_layer)

    target_locs = list(args.target_locs or DEFAULT_TOKEN_LOCS)
    output_csv = output_path_from_args(args, high_nodes, layers)

    # Use one base low graph and one abstract low graph per run.  The abstract
    # graph exposes all selected layers as intervention nodes.  Mapping objects
    # then choose one layer and one token location at a time.
    base_low = Arithmetic_Bert_CompGraph(model)
    all_low_nodes = [f"bert_layer_{layer}" for layer in layers]
    low_model = Abstr_Arithmetic_Bert_CompGraph(
        base_low,
        all_low_nodes,
        interv_info={"target_locs": target_locs},
        root_output_device=torch.device("cpu"),
    )
    low_model.set_cache_device(torch.device("cpu"))

    # High model is recreated per high node in the runner loop because the
    # abstract graph should expose only the high node being tested.
    all_results = []
    all_total_rows = 0
    start_mapping_id = 0
    for high_node in high_nodes:
        high_model = Abstr_Arithmetic_Logic_CompGraph([high_node], root_output_device=torch.device("cpu"))
        high_node_mappings = []
        for layer in layers:
            low_node = f"bert_layer_{layer}"
            high_node_mappings.extend(make_arithmetic_mappings(high_node, low_node, low_model))

        runner = ArithmeticInterchangeRunner(
            low_model=low_model,
            high_model=high_model,
            dataset=dataset,
            device=device,
            num_inputs=args.num_inputs,
            batch_size=args.batch_size,
        )

        # Offset mapping IDs if multiple high nodes are appended into one file.
        # The writer currently starts at 0 per call, so for simplicity use one
        # runner call when one high node is requested, and a temporary file append
        # for multiple nodes is avoided by calling all mappings at once below.
        all_results.append((runner, high_node_mappings))

    # Run all mappings with one writer by using the first runner and replacing
    # high_model for each high-node group.  This keeps the CSV header unique.
    from causal_abstraction.arithmetic_interchange import ArithmeticInterchangeCSVWriter
    writer = ArithmeticInterchangeCSVWriter(output_csv)
    try:
        mapping_id = 0
        for runner, mappings in all_results:
            for mapping in mappings:
                rows = runner.run_mapping(mapping_id, mapping, writer)
                all_total_rows += rows
                mapping_id += 1
    finally:
        writer.close()

    result = {
        "save_path": output_csv,
        "num_inputs": args.num_inputs,
        "num_mappings": mapping_id,
        "total_rows": all_total_rows,
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
    parser.add_argument("--target_locs", type=int, nargs="*", default=DEFAULT_TOKEN_LOCS)

    # Manager-compatible args.
    parser.add_argument("--id", type=int)
    parser.add_argument("--csv_path", default="")
    parser.add_argument("--abstraction", default="")
    parser.add_argument("--model_type", default="arithmetic_bert")
    parser.add_argument("--graph_alpha", default="")
    parser.add_argument("--interchange_batch_size", type=int, default=None)
    parser.add_argument("--loc_mapping_type", default="")
    parser.add_argument("--save_intermediate_results", default="True")

    args = parser.parse_args()
    if args.interchange_batch_size is not None:
        args.batch_size = args.interchange_batch_size

    res = run(args)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
