"""CSV manager for MQNLI BERT interchange jobs.

Identical pattern to arithmetic_interchange_manager.py.
One job = one high_node x one BERT layer.
target_locs are omitted — mqnli_interchange.py resolves them
per high_node from MQNLI_BERT_TOKEN_LOCS.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

STATUS_READY = 0
STATUS_DONE = 2

HIGH_NODES = [
    "sentence_q", "subj_adj", "subj_noun", "neg", "v_adv", "v_verb",
    "vp_q", "obj_adj", "obj_noun", "obj", "vp", "v_bar", "negp", "subj",
]

DEFAULT_OPTS = {
    "data_path": "",
    "model_path": "",
    "model_type": "bert",
    "res_save_dir": "",
    "abstraction": "",
    "num_inputs": 100,
    "interchange_batch_size": 64,
    "device": "cuda",
}


class MQNLICSVExperimentManager:
    def __init__(self, csv_path: str, default_opts: Optional[Dict[str, Any]] = None):
        self.csv_path = csv_path
        self.default_opts = default_opts or DEFAULT_OPTS.copy()
        if not os.path.exists(csv_path):
            cols = ["id", "status"] + [k for k in self.default_opts if k not in {"id", "status"}]
            self._write(cols, [])

    def _read(self):
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader.fieldnames or []), list(reader)

    def _write(self, cols: List[str], rows: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)

    def _ensure_cols(self, cols, rows, keys):
        for key in keys:
            if key not in cols:
                cols.append(key)
                for row in rows:
                    row[key] = ""
        return cols

    def insert(self, opts: Dict[str, Any]) -> int:
        cols, rows = self._read()
        row_id = len(rows) + 1
        full = {**self.default_opts, **opts}
        full["id"] = row_id
        full["status"] = full.get("status", STATUS_READY)
        cols = self._ensure_cols(cols, rows, full.keys())
        row = {c: "" for c in cols}
        row.update({k: str(v) for k, v in full.items()})
        rows.append(row)
        self._write(cols, rows)
        return row_id

    def update(self, row_id: int, opts: Dict[str, Any]) -> None:
        cols, rows = self._read()
        cols = self._ensure_cols(cols, rows, opts.keys())
        for row in rows:
            if int(row["id"]) == int(row_id):
                row.update({k: str(v) for k, v in opts.items()})
                break
        self._write(cols, rows)

    def fetch(self, n: Optional[int] = None, status: int = STATUS_READY):
        _, rows = self._read()
        result = [r for r in rows if int(r.get("status") or 0) == int(status)]
        return result[:n] if n is not None else result

    def query(self, status: Optional[int] = None, limit: Optional[int] = None):
        _, rows = self._read()
        if status is not None:
            rows = [r for r in rows if int(r.get("status") or 0) == int(status)]
        if limit is not None:
            rows = rows[:limit]
        return rows


def setup(csv_path: str, model_path: str, data_path: str) -> None:
    opts = DEFAULT_OPTS.copy()
    opts["model_path"] = model_path
    opts["data_path"] = data_path
    MQNLICSVExperimentManager(csv_path, opts)


def add(
    csv_path: str,
    model_path: str,
    data_path: str,
    res_dir: str,
    layers: List[int],
    high_nodes: List[str],
    num_inputs: int,
) -> None:
    manager = MQNLICSVExperimentManager(csv_path)
    time_str = datetime.now().strftime("%m%d-%H%M%S")
    high_nodes = high_nodes or HIGH_NODES
    layers = layers or list(range(11))
    for high_node in high_nodes:
        for layer in layers:
            row_id = manager.insert(
                {
                    "data_path": data_path,
                    "model_path": model_path,
                    "res_save_dir": os.path.join(res_dir, f"expt-{high_node}-layer{layer}-{time_str}"),
                    "abstraction": json.dumps([high_node, [f"bert_layer_{layer}"]]),
                    "num_inputs": num_inputs,
                }
            )
            print(f"added row {row_id}: {high_node} -> bert_layer_{layer}")


def run(csv_path: str, script: str, n: Optional[int]) -> None:
    manager = MQNLICSVExperimentManager(csv_path)
    for row in manager.fetch(n=n, status=STATUS_READY):
        cmd = [
            *shlex.split(script),
            "--id", row["id"],
            "--csv_path", csv_path,
            "--data_path", row["data_path"],
            "--model_path", row["model_path"],
            "--res_save_dir", row["res_save_dir"],
            "--abstraction", row["abstraction"],
            "--num_inputs", row.get("num_inputs", "100"),
            "--interchange_batch_size", row.get("interchange_batch_size", "64"),
            "--device", row.get("device", "cuda"),
        ]
        print("running:", " ".join(shlex.quote(x) for x in cmd))
        subprocess.run(cmd, check=True)


def query(csv_path: str, status: Optional[int], limit: Optional[int]) -> None:
    manager = MQNLICSVExperimentManager(csv_path)
    rows = manager.query(status=status, limit=limit)
    cols = ["id", "status", "abstraction", "num_inputs", "save_path", "total_rows"]
    print(", ".join(cols))
    print("-" * 80)
    for row in rows:
        print({c: row.get(c, "") for c in cols})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("setup")
    p.add_argument("-d", "--csv_path", required=True)
    p.add_argument("-m", "--model_path", required=True)
    p.add_argument("-i", "--data_path", required=True)

    p = sub.add_parser("add")
    p.add_argument("-d", "--csv_path", required=True)
    p.add_argument("-m", "--model_path", required=True)
    p.add_argument("-i", "--data_path", required=True)
    p.add_argument("-o", "--res_dir", required=True)
    p.add_argument("--layers", type=int, nargs="*", default=list(range(11)))
    p.add_argument("--high_nodes", nargs="*", default=HIGH_NODES)
    p.add_argument("--num_inputs", type=int, default=100)

    p = sub.add_parser("run")
    p.add_argument("-d", "--csv_path", required=True)
    p.add_argument("-i", "--script", default="python mqnli_interchange.py")
    p.add_argument("-n", type=int, default=None)

    p = sub.add_parser("query")
    p.add_argument("-d", "--csv_path", required=True)
    p.add_argument("-s", "--status", type=int, default=None)
    p.add_argument("-n", "--limit", type=int, default=None)

    args = parser.parse_args()
    if args.cmd == "setup":
        setup(args.csv_path, args.model_path, args.data_path)
    elif args.cmd == "add":
        add(args.csv_path, args.model_path, args.data_path, args.res_dir,
            args.layers, args.high_nodes, args.num_inputs)
    elif args.cmd == "run":
        run(args.csv_path, args.script, args.n)
    elif args.cmd == "query":
        query(args.csv_path, args.status, args.limit)


if __name__ == "__main__":
    main()
