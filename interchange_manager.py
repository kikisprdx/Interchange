import argparse
import csv
import os
import shlex
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

from experiment_interchange_interface import (
    STATUS_GRAPH_READY,
    STATUS_INTERCHANGE_DONE,
    STATUS_READY,
    ExperimentManagerInterface,
)

HIGH_NODES = [
    "sentence_q",
    "subj_adj",
    "subj_noun",
    "neg",
    "v_adv",
    "v_verb",
    "vp_q",
    "obj_adj",
    "obj_noun",
    "obj",
    "vp",
    "v_bar",
    "negp",
    "subj",
]

INTERCHANGE_DEFAULT_OPTS = {
    "data_path": "",
    "model_path": "",
    "model_type": "",
    "res_save_dir": "",
    "abstraction": "",
    "num_inputs": 500,
    "graph_alpha": 100,
    "interchange_batch_size": 128,
    "loc_mapping_type": "",
    "save_intermediate_results": True,
}


class CSVExperimentManager(ExperimentManagerInterface):
    def __init__(self, csv_path: str, default_opts: Optional[Dict] = None):
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
            if default_opts is None:
                raise ValueError("Must provide default_opts when creating new CSV")
            cols = ["id", "status"] + [
                k for k in default_opts if k not in ("id", "status")
            ]
            self._write(cols, [])

    def _read(self) -> tuple[List[str], List[Dict]]:
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames or [])
            rows = list(reader)
        return cols, rows

    def _write(self, cols: List[str], rows: List[Dict]) -> None:
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)

    def _ensure_cols(self, cols: List[str], rows: List[Dict], new_keys) -> List[str]:
        for k in new_keys:
            if k not in cols:
                cols.append(k)
                for r in rows:
                    r.setdefault(k, "")
        return cols

    def insert(self, opts: Dict) -> int:
        cols, rows = self._read()
        new_id = len(rows) + 1
        cols = self._ensure_cols(cols, rows, opts.keys())
        row = {c: "" for c in cols}
        row.update({k: str(v) for k, v in opts.items()})
        row["id"] = str(new_id)
        row.setdefault("status", str(STATUS_READY))
        rows.append(row)
        self._write(cols, rows)
        return new_id

    def update(self, opts: Dict, id: int) -> None:
        cols, rows = self._read()
        cols = self._ensure_cols(cols, rows, opts.keys())
        for row in rows:
            if int(row["id"]) == id:
                row.update({k: str(v) for k, v in opts.items()})
                break
        self._write(cols, rows)

    def fetch(self, n: Optional[int] = None, status: int = STATUS_READY) -> List[Dict]:
        _, rows = self._read()
        result = [r for r in rows if int(r.get("status", 0)) == status]
        return result[:n] if n is not None else result

    def query(
        self, cols=None, status=None, abstraction=None, id=None, limit=None
    ) -> List[Dict]:
        _, rows = self._read()
        if status is not None:
            rows = [r for r in rows if int(r.get("status", 0)) == status]
        if id is not None:
            rows = [r for r in rows if int(r["id"]) == id]
        if abstraction is not None:
            rows = [r for r in rows if abstraction in r.get("abstraction", "")]
        if limit is not None:
            rows = rows[:limit]
        if cols is not None:
            rows = [{c: r.get(c, "") for c in cols} for r in rows]
        return rows

    def get_script(self, opts: Dict, launch_script: str) -> str:
        parts = [launch_script]
        for k, v in opts.items():
            parts += [f"--{k}", repr(v)]
        parts += ["--csv_path", self.csv_path]
        return " ".join(parts)

    def dispatch(self, opts: Dict, launch_script: str) -> None:
        script = self.get_script(opts, launch_script)
        cmds = shlex.split(script)
        print("---- running:", cmds)
        subprocess.run(cmds)

    def run(
        self,
        launch_script: str,
        n: Optional[int] = None,
        ready_status: int = STATUS_READY,
    ) -> None:
        for opts in self.fetch(n, status=ready_status):
            self.dispatch(opts, launch_script)


# --- CLI functions ---


def setup(csv_path, model_path, data_path):
    opts = INTERCHANGE_DEFAULT_OPTS.copy()
    opts["data_path"] = data_path
    opts["model_path"] = model_path
    CSVExperimentManager(csv_path, opts)


def add(csv_path, model_type, model_path, res_dir, num_inputs, loc_mapping_type):
    import torch

    from modeling import get_module_class_by_name
    from modeling.utils import load_model

    model_class = get_module_class_by_name(model_type)
    manager = CSVExperimentManager(csv_path)
    device = torch.device("cpu")
    module, _ = load_model(model_class, model_path, device=device)

    if model_type == "lstm":
        num_layers = module.num_lstm_layers - 1
        layer_name = "lstm"
        interchange_batch_size = 1000
    elif model_type == "bert":
        num_layers = len(module.bert.encoder.layer) - 1
        layer_name = "bert_layer"
        interchange_batch_size = 500
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    time_str = datetime.now().strftime("%m%d-%H%M%S")
    for high_node in HIGH_NODES:
        for layer in range(num_layers):
            for n in num_inputs:
                abstraction = f'["{high_node}",["{layer_name}_{layer}"]]'
                insert_dict = {
                    "abstraction": abstraction,
                    "num_inputs": n,
                    "model_type": model_type,
                    "interchange_batch_size": interchange_batch_size,
                }
                if loc_mapping_type:
                    insert_dict["loc_mapping_type"] = loc_mapping_type
                row_id = manager.insert(insert_dict)
                res_save_dir = os.path.join(res_dir, f"expt-{row_id}-{time_str}")
                manager.update(
                    {"model_path": model_path, "res_save_dir": res_save_dir}, row_id
                )


def run(csv_path, script, n, ready_status):
    manager = CSVExperimentManager(csv_path, INTERCHANGE_DEFAULT_OPTS)
    if os.path.exists(script):
        with open(script) as f:
            script = f.read().strip()
    manager.run(launch_script=script, n=n, ready_status=ready_status)


def add_graph(csv_path, ids, alpha, all_rows):
    manager = CSVExperimentManager(csv_path)
    if all_rows:
        for row in manager.query(status=STATUS_INTERCHANGE_DONE):
            manager.update(
                {"graph_alpha": alpha, "status": STATUS_GRAPH_READY}, int(row["id"])
            )
    elif ids:
        for row_id in ids:
            manager.update({"graph_alpha": alpha, "status": STATUS_GRAPH_READY}, row_id)


def analyze_graph(csv_path, script, n, ready_status):
    manager = CSVExperimentManager(csv_path)
    if os.path.exists(script):
        with open(script) as f:
            script = f.read().strip()
    manager.run(launch_script=script, n=n, ready_status=ready_status)


def query(csv_path, id=None, status=None, abstraction=None, limit=None):
    manager = CSVExperimentManager(csv_path)
    cols = ["id", "res_save_dir", "abstraction", "num_inputs", "status"]
    rows = manager.query(
        cols=cols, status=status, abstraction=abstraction, id=id, limit=limit
    )
    if not rows:
        print("No data found")
        return
    header = ", ".join(cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(row)
        print("-------")


def update_status(csv_path, ids, id_range, status):
    manager = CSVExperimentManager(csv_path)
    if id_range:
        ids = list(range(id_range[0], id_range[1] + 1))
    for row_id in ids:
        manager.update({"status": status}, row_id)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")

    setup_p = subparsers.add_parser("setup")
    setup_p.add_argument("-d", "--csv_path", required=True)
    setup_p.add_argument("-m", "--model_path", type=str)
    setup_p.add_argument("-i", "--data_path", type=str)

    add_p = subparsers.add_parser("add")
    add_p.add_argument("-d", "--csv_path", required=True)
    add_p.add_argument("-t", "--model_type", required=True)
    add_p.add_argument("-m", "--model_path", required=True)
    add_p.add_argument("-o", "--res_dir", required=True)
    add_p.add_argument("-n", "--num_inputs", type=int, nargs="+")
    add_p.add_argument("-l", "--loc_mapping_type", type=str, default="")

    add_graph_p = subparsers.add_parser("add_graph")
    add_graph_p.add_argument("-d", "--csv_path", required=True)
    add_graph_p.add_argument("-i", "--ids", type=int, nargs="*")
    add_graph_p.add_argument("-a", "--alpha", type=int, required=True)
    add_graph_p.add_argument("--all_rows", action="store_true")

    run_p = subparsers.add_parser("run")
    run_p.add_argument("-d", "--csv_path", required=True)
    run_p.add_argument("-i", "--script", default="python interchange.py")
    run_p.add_argument("-n", "--n", type=int, default=None)
    run_p.add_argument("-r", "--ready_status", type=int, default=STATUS_READY)

    analyze_p = subparsers.add_parser("analyze_graph")
    analyze_p.add_argument("-d", "--csv_path", required=True)
    analyze_p.add_argument("-i", "--script", default="python graph_analysis.py")
    analyze_p.add_argument("-n", "--n", type=int, default=None)
    analyze_p.add_argument("-r", "--ready_status", type=int, default=STATUS_GRAPH_READY)

    query_p = subparsers.add_parser("query")
    query_p.add_argument("-d", "--csv_path", required=True)
    query_p.add_argument("-i", "--id", type=int)
    query_p.add_argument("-s", "--status", type=int)
    query_p.add_argument("-a", "--abstraction", type=str)
    query_p.add_argument("-n", "--limit", type=int)

    update_p = subparsers.add_parser("update_status")
    update_p.add_argument("-d", "--csv_path", required=True)
    update_p.add_argument("-i", "--ids", type=int, nargs="*")
    update_p.add_argument("-r", "--id_range", type=int, nargs=2)
    update_p.add_argument("-s", "--status", type=int, required=True)

    kwargs = vars(parser.parse_args())
    globals()[kwargs.pop("subparser")](**kwargs)


if __name__ == "__main__":
    main()
