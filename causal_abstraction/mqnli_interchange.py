from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset

import pipeline as intervention
from pipeline.location import Location
from pipeline.utils import serialize


def _to_bool_int(value: bool) -> int:
    return 1 if bool(value) else 0


def tensor_to_jsonable(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        if x.dim() == 0:
            return int(x.item())
        return x.detach().cpu().long().tolist()
    return x


def location_to_str(loc: Any) -> str:
    if isinstance(loc, tuple):
        parts = []
        for x in loc:
            if isinstance(x, slice):
                parts.append(Location._slice_to_str(x))
            else:
                parts.append(str(x))
        return "[" + ",".join(parts) + "]"
    if isinstance(loc, slice):
        return Location.slice_to_str(loc)
    return str(loc)


@dataclass(frozen=True)
class MQNLIMapping:
    high_node: str
    low_node: str
    low_location: Any

    def as_dict(self) -> Dict[str, Any]:
        return {
            "high_node": self.high_node,
            "low_node": self.low_node,
            "low_location": location_to_str(self.low_location),
        }


class MQNLIBaseCache:
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
            self.low_fields[field_idx].extend([x.detach().cpu() if isinstance(x, torch.Tensor) else torch.tensor(x) for x in field_tensor])
        self.raw_inputs.extend([x.detach().cpu().long() for x in raw_inputs_cpu])
        self.labels.extend([int(x) for x in labels_cpu.tolist()])
        self.low_outputs.extend([int(x) for x in low_outputs_cpu.tolist()])
        self.high_outputs.extend([int(x) for x in high_outputs_cpu.tolist()])
        self.low_hidden_values.extend([x.detach().cpu() for x in low_hidden_cpu])
        self.high_hidden_values.extend([x.detach().cpu() for x in high_hidden_cpu])

    def low_batch(self, indices: torch.Tensor, device: torch.device) -> List[torch.Tensor]:
        return [torch.stack([field[int(i)] for i in indices]).to(device) for field in self.low_fields]

    def raw_batch(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.raw_inputs[int(i)] for i in indices]).long()

    def low_hidden_batch(self, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        return torch.stack([self.low_hidden_values[int(i)] for i in indices]).to(device)

    def high_hidden_batch(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.high_hidden_values[int(i)] for i in indices])

    def raw_row(self, index: int) -> List[int]:
        return [int(x) for x in self.raw_inputs[index].long().tolist()]


def make_mqnli_mappings(high_node: str, low_node: str, low_model: Any) -> List[MQNLIMapping]:
    return [MQNLIMapping(high_node, low_node, loc) for loc in low_model.get_indices(low_node)]


class MQNLIInterchangeCSVWriter:
    fieldnames = [
        "mapping_id",
        "high_node",
        "low_node",
        "low_location",
        "base_i",
        "source_i",
        "base_raw_original_x",
        "source_raw_original_x",
        "gold_label",
        "source_gold_label",
        "source_high_value",
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


class MQNLIInterchangeRunner:
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
                "MQNLI interchange expects dataset items of the form "
                "(input_ids, token_type_ids, attention_mask, raw_original_x, label)."
            )

    def _labels_to_tensor(self, labels: Any) -> torch.Tensor:
        if isinstance(labels, torch.Tensor):
            return labels.long().cpu()
        return torch.tensor(labels, dtype=torch.long)

    def _cache_base_runs(self, mapping: MQNLIMapping) -> MQNLIBaseCache:
        n = min(self.num_inputs, len(self.dataset))
        subset = Subset(self.dataset, list(range(n)))
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)
        cache = MQNLIBaseCache(num_low_fields=len(subset[0]))

        for input_tuple in loader:
            raw_inputs = input_tuple[-2].long().cpu()
            labels = self._labels_to_tensor(input_tuple[-1])

            high_keys = [serialize(x) for x in raw_inputs]
            high_graph_input = intervention.GraphInput.make_batched(
                {"input": raw_inputs}, high_keys, batch_dim=0
            )
            high_output = self.high_model.compute(high_graph_input)
            high_hidden = self.high_model.get_result(mapping.high_node, high_graph_input)

            low_input_for_graph = [x.to(self.device) if isinstance(x, torch.Tensor) else torch.tensor(x).to(self.device) for x in input_tuple]
            low_keys = [serialize(x) for x in input_tuple[0]]
            low_graph_input = intervention.GraphInput.make_batched(
                {"input": low_input_for_graph}, low_keys, batch_dim=0
            )
            low_output = self.low_model.compute(low_graph_input)
            low_hidden_full = self.low_model.get_result(mapping.low_node, low_graph_input)
            low_hidden = low_hidden_full[mapping.low_location]

            cache.append_batch(
                low_input_tuple_cpu=[x.detach().cpu() if isinstance(x, torch.Tensor) else torch.tensor(x) for x in input_tuple],
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

    def run_mapping(self, mapping_id: int, mapping: MQNLIMapping, writer: MQNLIInterchangeCSVWriter) -> int:
        cache = self._cache_base_runs(mapping)
        n = cache.num_examples
        written_before = writer.row_count

        for base_idx, source_idx in self._pair_indices(n):
            high_base_raw = cache.raw_batch(base_idx)
            high_interv_value = cache.high_hidden_batch(source_idx)

            high_base_key = [serialize(x) for x in high_base_raw]
            high_base = intervention.GraphInput.make_batched(
                {"input": high_base_raw}, high_base_key, cache_results=False, batch_dim=0
            )
            high_interv_key = [(serialize(x), serialize(v)) for x, v in zip(high_base_raw, high_interv_value)]
            high_intervention = intervention.Intervention.make_batched(
                high_base,
                high_interv_key,
                intervention={mapping.high_node: high_interv_value},
                batch_dim=0,
            )
            high_base_res, high_interv_res = self.high_model.intervene(high_intervention, store_cache=False)

            low_base_tuple = cache.low_batch(base_idx, self.device)
            low_interv_value = cache.low_hidden_batch(source_idx, self.device)
            low_base_key = [serialize(x) for x in low_base_tuple[0].detach().cpu()]
            low_base = intervention.GraphInput.make_batched(
                {"input": low_base_tuple}, low_base_key, cache_results=False, batch_dim=0
            )
            low_interv_key = [
                (serialize(x.detach().cpu()), serialize(v.detach().cpu()))
                for x, v in zip(low_base_tuple[0], low_interv_value)
            ]
            low_intervention = intervention.Intervention.make_batched(
                low_base,
                low_interv_key,
                intervention={mapping.low_node: low_interv_value},
                location={mapping.low_node: mapping.low_location},
                batch_dim=0,
            )
            low_base_res, low_interv_res = self.low_model.intervene(low_intervention, store_cache=False)

            high_base_res = high_base_res.detach().cpu()
            high_interv_res = high_interv_res.detach().cpu()
            low_base_res = low_base_res.detach().cpu()
            low_interv_res = low_interv_res.detach().cpu()

            rows = []
            for row_pos in range(len(base_idx)):
                b = int(base_idx[row_pos])
                s = int(source_idx[row_pos])
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
                        "base_raw_original_x": json.dumps(cache.raw_row(b)),
                        "source_raw_original_x": json.dumps(cache.raw_row(s)),
                        "gold_label": cache.labels[b],
                        "source_gold_label": cache.labels[s],
                        "source_high_value": json.dumps(tensor_to_jsonable(cache.high_hidden_values[s])),
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

        self.low_model.clear_caches()
        self.high_model.clear_caches()
        return writer.row_count - written_before

    def run_to_csv(self, mappings: Sequence[MQNLIMapping], output_csv: str) -> Dict[str, Any]:
        writer = MQNLIInterchangeCSVWriter(output_csv)
        mapping_counts: List[int] = []
        try:
            for mapping_id, mapping in enumerate(mappings):
                mapping_counts.append(self.run_mapping(mapping_id, mapping, writer))
        finally:
            writer.close()
        return {
            "save_path": output_csv,
            "num_mappings": len(mappings),
            "mapping_row_counts": mapping_counts,
            "total_rows": sum(mapping_counts),
        }
