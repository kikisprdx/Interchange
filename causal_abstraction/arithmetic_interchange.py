"""Batched arithmetic interchange experiment.

This module is deliberately close to the original
`causal_abstraction/interchange.py`, but it is specialized to the arithmetic
BERT task and writes the raw intervention results directly to CSV.

It does not run clique analysis.  The saved CSV is the input for any later
analysis step.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset

import intervention
from intervention.utils import serialize
from intervention.location import Location


def _to_bool_int(value: bool) -> int:
    return 1 if bool(value) else 0


def location_to_str(loc: Any) -> str:
    """Convert an intervention Location object/index tuple into stable text."""
    if isinstance(loc, tuple):
        parts = []
        for x in loc:
            if isinstance(x, slice):
                parts.append(Location.slice_to_str(x))
            else:
                parts.append(str(x))
        return "[" + ",".join(parts) + "]"
    if isinstance(loc, slice):
        return Location.slice_to_str(loc)
    return str(loc)


@dataclass(frozen=True)
class ArithmeticMapping:
    """One tested high-level/low-level alignment.

    Example: high_node='x_value', low_node='bert_layer_4',
    low_location=LOC[:, 1, :].
    """

    high_node: str
    low_node: str
    low_location: Any

    def as_dict(self) -> Dict[str, Any]:
        return {
            "high_node": self.high_node,
            "low_node": self.low_node,
            "low_location": location_to_str(self.low_location),
        }


class ArithmeticBaseCache:
    """Stores base outputs and hidden states for the selected examples.

    The original paper code builds an IterableDataset that materializes all
    base/source pairs.  For 100 examples this is not necessary.  Here we first
    cache the N=100 normal forward passes, then generate N*N intervention pairs
    in small batches and write each batch to CSV.
    """

    def __init__(self, num_low_fields: int):
        self.low_fields: List[List[torch.Tensor]] = [[] for _ in range(num_low_fields)]
        self.raw_inputs: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.low_outputs: List[int] = []
        self.high_outputs: List[int] = []
        self.low_hidden_values: List[torch.Tensor] = []
        self.high_hidden_values: List[torch.Tensor] = []

    @property
    def num_examples(self) -> int:
        return len(self.low_outputs)

    def append_batch(
        self,
        low_input_tuple_cpu: Sequence[torch.Tensor],
        raw_inputs_cpu: torch.Tensor,
        labels_cpu: torch.Tensor,
        low_outputs_cpu: torch.Tensor,
        high_outputs_cpu: torch.Tensor,
        low_hidden_cpu: torch.Tensor,
        high_hidden_cpu: torch.Tensor,
    ) -> None:
        for field_idx, field_tensor in enumerate(low_input_tuple_cpu):
            # Store each example separately.  This keeps later base/source
            # indexing simple and avoids relying on Dataset internals.
            self.low_fields[field_idx].extend([x.detach().cpu() for x in field_tensor])

        self.raw_inputs.extend([x.detach().cpu() for x in raw_inputs_cpu])
        self.labels.extend([int(x) for x in labels_cpu.tolist()])
        self.low_outputs.extend([int(x) for x in low_outputs_cpu.tolist()])
        self.high_outputs.extend([int(x) for x in high_outputs_cpu.tolist()])
        self.low_hidden_values.extend([x.detach().cpu() for x in low_hidden_cpu])
        self.high_hidden_values.extend([x.detach().cpu() for x in high_hidden_cpu])

    def low_batch(self, indices: torch.Tensor, device: torch.device) -> List[torch.Tensor]:
        return [torch.stack([field[int(i)] for i in indices]).to(device) for field in self.low_fields]

    def raw_batch(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.raw_inputs[int(i)] for i in indices])

    def low_hidden_batch(self, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        return torch.stack([self.low_hidden_values[int(i)] for i in indices]).to(device)

    def high_hidden_batch(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.high_hidden_values[int(i)] for i in indices])

    def raw_row(self, index: int) -> Tuple[int, int, int, int, int]:
        """Return x, op, y, result, sign for CSV output."""
        raw = self.raw_inputs[index].long().tolist()
        x, op_id, y = int(raw[0]), int(raw[1]), int(raw[2])
        result = x + y if op_id == 0 else x - y
        sign = 0 if result > 0 else 1 if result < 0 else 2
        return x, op_id, y, result, sign


def make_arithmetic_mappings(
    high_node: str,
    low_node: str,
    low_model: Any,
) -> List[ArithmeticMapping]:
    """Create all low-location mappings for one high node and one low node."""
    return [ArithmeticMapping(high_node, low_node, loc) for loc in low_model.get_indices(low_node)]


class ArithmeticInterchangeCSVWriter:
    """Small wrapper around csv.DictWriter with a fixed schema."""

    fieldnames = [
        "mapping_id",
        "high_node",
        "low_node",
        "low_location",
        "base_i",
        "source_i",
        "base_x",
        "base_op",
        "base_y",
        "base_result",
        "base_sign",
        "source_x",
        "source_op",
        "source_y",
        "source_result",
        "source_sign",
        "gold_label",
        "high_base_res",
        "low_base_res",
        "high_interv_res",
        "low_interv_res",
        "high_changed",
        "base_correct_vs_high",
        "interchange_success",
        "impactful_success",
    ]

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.row_count = 0

    def write_rows(self, rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            self.writer.writerow(row)
            self.row_count += 1

    def close(self) -> None:
        self.file.close()


class ArithmeticInterchangeRunner:
    """Runs interchange interventions for arithmetic and saves raw rows to CSV.

    Parameters
    ----------
    low_model:
        An abstracted low-level BERT graph, for example
        Abstr_Arithmetic_Bert_CompGraph(full_bert_graph, ["bert_layer_4"], ...).
    high_model:
        An abstracted high-level arithmetic logic graph, for example
        Abstr_Arithmetic_Logic_CompGraph(["x_value"]).
    dataset:
        A dataset whose items must be:
        (input_ids, token_type_ids, attention_mask, raw_arithmetic_input, label).
        The raw arithmetic input must be [x, op_id, y], with op_id 0 for '+',
        1 for '-'.
    device:
        Device for the neural BERT graph.  The high-level model runs on CPU.
    """

    def __init__(
        self,
        low_model: Any,
        high_model: Any,
        dataset: Any,
        device: torch.device,
        num_inputs: int = 100,
        batch_size: int = 64,
    ):
        self.low_model = low_model
        self.high_model = high_model
        self.dataset = dataset
        self.device = device
        self.num_inputs = int(num_inputs)
        self.batch_size = int(batch_size)

        if len(dataset[0]) < 5:
            raise ValueError(
                "Arithmetic interchange expects dataset items of the form "
                "(input_ids, token_type_ids, attention_mask, raw_arithmetic_input, label). "
                "Add raw_arithmetic_input=[x, op_id, y] during preprocessing."
            )

    def _cache_base_runs(self, mapping: ArithmeticMapping) -> ArithmeticBaseCache:
        """Compute normal high/low outputs and hidden values for N examples."""
        n = min(self.num_inputs, len(self.dataset))
        subset = Subset(self.dataset, list(range(n)))
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)

        num_fields = len(subset[0])
        cache = ArithmeticBaseCache(num_low_fields=num_fields)

        for input_tuple in loader:
            raw_inputs = input_tuple[-2].long().cpu()
            labels = input_tuple[-1].long().cpu()

            # High-level graph receives only [x, op_id, y].
            high_keys = [serialize(x) for x in raw_inputs]
            high_graph_input = intervention.GraphInput.batched(
                {"input": raw_inputs}, high_keys, batch_dim=0
            )
            high_output = self.high_model.compute(high_graph_input)
            high_hidden = self.high_model.get_result(mapping.high_node, high_graph_input)

            low_input_for_graph = [x.to(self.device) for x in input_tuple]
            low_keys = [serialize(x) for x in input_tuple[0]]
            low_graph_input = intervention.GraphInput.batched(
                {"input": low_input_for_graph}, low_keys, batch_dim=0
            )
            low_output = self.low_model.compute(low_graph_input)
            low_hidden_full = self.low_model.get_result(mapping.low_node, low_graph_input)
            low_hidden = low_hidden_full[mapping.low_location]

            cache.append_batch(
                low_input_tuple_cpu=[x.detach().cpu() for x in input_tuple],
                raw_inputs_cpu=raw_inputs,
                labels_cpu=labels,
                low_outputs_cpu=low_output.detach().cpu(),
                high_outputs_cpu=high_output.detach().cpu(),
                low_hidden_cpu=low_hidden.detach().cpu(),
                high_hidden_cpu=high_hidden.detach().cpu(),
            )

        return cache

    def _pair_indices(self, n: int) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        pairs = list(product(range(n), repeat=2))
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            base_idx = torch.tensor([p[0] for p in chunk], dtype=torch.long)
            source_idx = torch.tensor([p[1] for p in chunk], dtype=torch.long)
            yield base_idx, source_idx

    def run_mapping(
        self,
        mapping_id: int,
        mapping: ArithmeticMapping,
        writer: ArithmeticInterchangeCSVWriter,
    ) -> int:
        """Run all 100x100 base/source interventions for one mapping."""
        cache = self._cache_base_runs(mapping)
        n = cache.num_examples
        written_before = writer.row_count

        for base_idx, source_idx in self._pair_indices(n):
            # ----- high-level intervention -----
            high_base_raw = cache.raw_batch(base_idx)
            high_interv_value = cache.high_hidden_batch(source_idx)

            high_base_key = [serialize(x) for x in high_base_raw]
            high_base = intervention.GraphInput.batched(
                {"input": high_base_raw}, high_base_key, cache_results=False, batch_dim=0
            )
            high_interv_key = [
                (serialize(x), serialize(v)) for x, v in zip(high_base_raw, high_interv_value)
            ]
            high_intervention = intervention.Intervention.batched(
                high_base,
                high_interv_key,
                intervention={mapping.high_node: high_interv_value},
                batch_dim=0,
            )
            high_base_res, high_interv_res = self.high_model.intervene(
                high_intervention, store_cache=False
            )

            # ----- low-level BERT intervention -----
            low_base_tuple = cache.low_batch(base_idx, self.device)
            low_interv_value = cache.low_hidden_batch(source_idx, self.device)

            low_base_key = [serialize(x) for x in low_base_tuple[0].detach().cpu()]
            low_base = intervention.GraphInput.batched(
                {"input": low_base_tuple}, low_base_key, cache_results=False, batch_dim=0
            )
            low_interv_key = [
                (serialize(x.detach().cpu()), serialize(v.detach().cpu()))
                for x, v in zip(low_base_tuple[0], low_interv_value)
            ]
            low_intervention = intervention.Intervention.batched(
                low_base,
                low_interv_key,
                intervention={mapping.low_node: low_interv_value},
                location={mapping.low_node: mapping.low_location},
                batch_dim=0,
            )
            low_base_res, low_interv_res = self.low_model.intervene(
                low_intervention, store_cache=False
            )

            high_base_res = high_base_res.detach().cpu()
            high_interv_res = high_interv_res.detach().cpu()
            low_base_res = low_base_res.detach().cpu()
            low_interv_res = low_interv_res.detach().cpu()

            rows = []
            for row_pos in range(len(base_idx)):
                b = int(base_idx[row_pos])
                s = int(source_idx[row_pos])
                base_x, base_op, base_y, base_result, base_sign = cache.raw_row(b)
                source_x, source_op, source_y, source_result, source_sign = cache.raw_row(s)

                hb = int(high_base_res[row_pos])
                lb = int(low_base_res[row_pos])
                hi = int(high_interv_res[row_pos])
                li = int(low_interv_res[row_pos])
                high_changed = hb != hi
                base_correct_vs_high = lb == hb
                interchange_success = li == hi

                rows.append(
                    {
                        **mapping.as_dict(),
                        "mapping_id": mapping_id,
                        "base_i": b,
                        "source_i": s,
                        "base_x": base_x,
                        "base_op": base_op,
                        "base_y": base_y,
                        "base_result": base_result,
                        "base_sign": base_sign,
                        "source_x": source_x,
                        "source_op": source_op,
                        "source_y": source_y,
                        "source_result": source_result,
                        "source_sign": source_sign,
                        "gold_label": cache.labels[b],
                        "high_base_res": hb,
                        "low_base_res": lb,
                        "high_interv_res": hi,
                        "low_interv_res": li,
                        "high_changed": _to_bool_int(high_changed),
                        "base_correct_vs_high": _to_bool_int(base_correct_vs_high),
                        "interchange_success": _to_bool_int(interchange_success),
                        "impactful_success": _to_bool_int(high_changed and base_correct_vs_high and interchange_success),
                    }
                )
            writer.write_rows(rows)

        # Clear graph caches between mappings to avoid memory growth.
        self.low_model.clear_caches()
        self.high_model.clear_caches()
        return writer.row_count - written_before

    def run_to_csv(self, mappings: Sequence[ArithmeticMapping], output_csv: str) -> Dict[str, Any]:
        writer = ArithmeticInterchangeCSVWriter(output_csv)
        mapping_counts: List[int] = []
        try:
            for mapping_id, mapping in enumerate(mappings):
                count = self.run_mapping(mapping_id, mapping, writer)
                mapping_counts.append(count)
        finally:
            writer.close()

        return {
            "save_path": output_csv,
            "num_mappings": len(mappings),
            "mapping_row_counts": mapping_counts,
            "total_rows": sum(mapping_counts),
        }
