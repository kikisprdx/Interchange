"""Pytest suite for `managers/arithmetic.py` CSVExperimentManager."""

import os
import pytest
import tempfile

from interchange_manager import CSVExperimentManager
from experiment_interchange_interface import STATUS_READY, STATUS_INTERCHANGE_DONE, STATUS_GRAPH_READY


DEFAULT_OPTS = {
    "abstraction": "",
    "model_path": "",
    "num_inputs": 500,
    "graph_alpha": 100,
}


@pytest.fixture
def csv_path(tmp_path):
    """Return a temp path for the experiments CSV."""
    return str(tmp_path / "experiments.csv")


@pytest.fixture
def manager(csv_path):
    """Return a fresh CSVExperimentManager backed by the temp CSV."""
    return CSVExperimentManager(csv_path, DEFAULT_OPTS)


def test_creates_file_with_headers(csv_path):
    CSVExperimentManager(csv_path, DEFAULT_OPTS)
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
    assert "id" in header
    assert "status" in header


def test_raises_without_opts_on_new_file(csv_path):
    with pytest.raises(ValueError):
        CSVExperimentManager(csv_path)


def test_insert_returns_sequential_ids(manager):
    id1 = manager.insert({"abstraction": "a"})
    id2 = manager.insert({"abstraction": "b"})
    assert id1 == 1
    assert id2 == 2


def test_insert_defaults_status_to_ready(manager):
    row_id = manager.insert({"abstraction": "a"})
    rows = manager.query(id=row_id)
    # Status stored as string in CSV
    assert int(rows[0]["status"]) == STATUS_READY


def test_update_patches_field(manager):
    row_id = manager.insert({"abstraction": "a"})
    manager.update({"status": STATUS_INTERCHANGE_DONE}, row_id)
    rows = manager.query(id=row_id)
    # Status stored as string in CSV
    assert int(rows[0]["status"]) == STATUS_INTERCHANGE_DONE


def test_update_adds_new_column(manager):
    row_id = manager.insert({"abstraction": "a"})
    manager.update({"new_field": "hello"}, row_id)
    rows = manager.query(id=row_id)
    assert rows[0]["new_field"] == "hello"


def test_update_new_col_empty_on_other_rows(manager):
    id1 = manager.insert({"abstraction": "a"})
    id2 = manager.insert({"abstraction": "b"})
    manager.update({"extra": "only_on_first"}, id1)
    rows = manager.query(id=id2)
    assert rows[0].get("extra", "") == ""


def test_fetch_filters_by_status(manager):
    manager.insert({"abstraction": "a"})
    manager.insert({"abstraction": "b"})
    id3 = manager.insert({"abstraction": "c"})
    manager.update({"status": STATUS_INTERCHANGE_DONE}, id3)

    ready = manager.fetch(status=STATUS_READY)
    done = manager.fetch(status=STATUS_INTERCHANGE_DONE)
    assert len(ready) == 2
    assert len(done) == 1


def test_fetch_respects_n(manager):
    for i in range(5):
        manager.insert({"abstraction": str(i)})
    result = manager.fetch(n=3, status=STATUS_READY)
    assert len(result) == 3


def test_query_abstraction_substring(manager):
    manager.insert({"abstraction": '["sentence_q", ["bert_layer_0"]]'})
    manager.insert({"abstraction": '["subj", ["bert_layer_1"]]'})
    rows = manager.query(abstraction="sentence_q")
    assert len(rows) == 1
    assert "sentence_q" in rows[0]["abstraction"]


def test_query_limit(manager):
    for i in range(10):
        manager.insert({"abstraction": str(i)})
    rows = manager.query(limit=4)
    assert len(rows) == 4


def test_query_cols_projection(manager):
    manager.insert({"abstraction": "a", "num_inputs": 100})
    rows = manager.query(cols=["id", "abstraction"])
    assert set(rows[0].keys()) == {"id", "abstraction"}
