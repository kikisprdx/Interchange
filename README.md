# Interchange

causal abstraction experiments on transformer models via interchange interventions. patches hidden states from one input into another and checks whether low-level and high-level model outputs align — testing whether a given layer/location implements a specific high-level computation.

## requirements

python 3.12, and the `intervention` package (separate repo, must be installed or on `PYTHONPATH`).

```
pip install -e .
```

## repo structure

```
compgraphs/         computation graph definitions
  abstractable.py   AbstractableCompGraph base class
  mqnli_bert.py     full + abstract compgraph for BERT on MQNLI
  arithmetic_bert.py full + abstract compgraph for BERT on arithmetic
datasets/
  arithmetic.py     data generation + Dataset class for arithmetic task
experiment_interchange_interface.py  ABCs for the experiment backend
interchange_manager.py               CLI job queue manager (CSV-backed)
main.py             entry point
run_arithmetic.py   arithmetic dataset smoke test
```

## concepts

**computation graph** — a DAG wrapping a model's forward pass. each node corresponds to an intermediate representation (embedding, bert layer N, pooler, logits, etc.). built via the `intervention` library's `ComputationGraph` / `GraphNode` API.

**abstractable graph** — `AbstractableCompGraph` takes a full graph spec and a list of `abstract_nodes` to keep, collapsing everything else into composed forward functions. used to define the "low-level" graph that interchange interventions run on.

**interchange intervention** — patch the hidden state at a chosen node from an *intervention* input into a *base* input's forward pass. record whether the base output changes to match the intervention output. this tests whether that node causally mediates the high-level behaviour.

**3-bit outcome encoding** — each (base, interv) pair produces a result `N` in 0–7:
- bit 2: `high_effect_eq` — high-level model output changed
- bit 1: `base_eq` — low base output == high base output
- bit 0: `interv_eq` — low interv output == high interv output

stored as `res_0_count` … `res_7_count` in the job queue.

**clique analysis** — builds a graph over input examples where an edge means the two examples are mutually interchangeable at a given mapping. max clique size = largest subset the mapping consistently works on.

## tasks

### arithmetic

simple arithmetic expressions: `[CLS] x op y [SEP]` where `x, y ∈ 0–9`, `op ∈ {+, -}`. label is `positive / negative / zero`. 200 total examples, 70/15/15 train/dev/test split.

### MQNLI

monotonicity reasoning NLI task. BERT model with 14 high-level nodes (`sentence_q`, `subj`, `neg`, `vp`, etc.) tested against BERT layers.

## job queue (interchange_manager.py)

CSV-backed experiment queue. each row is one job with fields from `INTERCHANGE_DEFAULT_OPTS` plus results written back on completion.

**status values:**

| value | meaning |
|-------|---------|
| 0 | ready |
| 1 | running |
| 2 | interchange done |
| 3 | queued for graph analysis |
| 4 | graph analysis done |

**subcommands:**

```bash
# initialise a new queue
python interchange_manager.py setup -d experiments.csv -m model.pt -i data/

# populate with jobs (iterates high nodes × layers × num_inputs)
python interchange_manager.py add -d experiments.csv -t bert -m model.pt -o results/ -n 500

# run interchange experiments (dispatches worker scripts)
python interchange_manager.py run -d experiments.csv -i python interchange.py

# mark completed interchange jobs as ready for graph analysis
python interchange_manager.py add_graph -d experiments.csv -a 100 --all_rows

# run graph/clique analysis
python interchange_manager.py analyze_graph -d experiments.csv -i python graph_analysis.py

# query queue state
python interchange_manager.py query -d experiments.csv -s 0

# manually update status
python interchange_manager.py update_status -d experiments.csv -i 3 4 5 -s 0
```

## implementing the experiment backend

`experiment_interchange_interface.py` defines two ABCs your worker scripts must implement:

- `ExperimentManagerInterface` — `insert`, `update`, `fetch`, `query` against whatever storage you use
- `ExperimentInterface` — implement `experiment(opts) -> dict`; call `run(opts, manager)` to execute and write results back automatically

`opts` keys passed by the manager: `id`, `csv_path`, `abstraction`, `model_path`, `model_type`, `data_path`, `num_inputs`, `graph_alpha`, `interchange_batch_size`, `res_save_dir`, `save_intermediate_results`, `loc_mapping_type`.

return dict must include: `save_path`, `res_0_count`…`res_7_count`, `max_clique_sizes`, `avg_clique_sizes`, `sum_clique_sizes`, `clique_counts` (last four are JSON strings, one value per mapping).

## entry point

```bash
python main.py arithmetic   # generates data + loads dataset, prints split sizes + sample
```
