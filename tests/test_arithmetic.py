import os
import pytest
import torch
import tempfile

from datasets.arithmetic import generate_data, ArithmeticData, ArithmeticDataset, label_dict, SEQ_LEN

CLS_ID = 101
SEP_ID = 102


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "arithmetic.csv")
    original = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("root"))
    os.makedirs("data", exist_ok=True)
    import shutil
    shutil.copy(path, "data/arithmetic.csv") if os.path.exists(path) else None
    os.chdir(original)
    return path


@pytest.fixture(scope="module")
def generated_csv(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    csv_file = str(d / "arithmetic.csv")
    orig_dir = os.getcwd()
    os.chdir(str(d))
    os.makedirs("data", exist_ok=True)
    generate_data()
    os.chdir(orig_dir)
    return str(d / "data" / "arithmetic.csv")


def test_generate_data_row_count(generated_csv):
    with open(generated_csv) as f:
        rows = f.readlines()
    assert len(rows) == 401 * 401 * 2


def test_generate_data_outcome_correctness(generated_csv):
    with open(generated_csv) as f:
        lines = f.readlines()
    for line in lines[:200]:
        expr, result, outcome = line.strip().split(", ")
        x, op, y = expr.split()
        x, y = int(x), int(y)
        z = x + y if op == '+' else x - y
        expected = 'positive' if z > 0 else ('negative' if z < 0 else 'zero')
        assert outcome == expected


def test_generate_data_no_duplicates(generated_csv):
    with open(generated_csv) as f:
        lines = f.readlines()
    assert len(lines) == len(set(lines))


@pytest.fixture(scope="module")
def vocab_path(tmp_path_factory):
    from transformers import BertTokenizer
    d = str(tmp_path_factory.mktemp("vocab"))
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer.save_vocabulary(d)[0]


@pytest.fixture(scope="module")
def arithmetic_data(generated_csv, vocab_path):
    return ArithmeticData(generated_csv, vocab_path)


def test_split_ratios(arithmetic_data):
    total = len(arithmetic_data.train) + len(arithmetic_data.dev) + len(arithmetic_data.test)
    assert abs(len(arithmetic_data.train) / total - 0.70) < 0.01
    assert abs(len(arithmetic_data.dev) / total - 0.15) < 0.01


def test_dataset_item_length(arithmetic_data):
    item = arithmetic_data.train[0]
    assert len(item) == 4


def test_input_ids_length(arithmetic_data):
    input_ids, token_type_ids, attn_mask, label = arithmetic_data.train[0]
    assert input_ids.shape == torch.Size([SEQ_LEN])


def test_cls_sep_tokens(arithmetic_data):
    for i in range(10):
        input_ids, _, _, _ = arithmetic_data.train[i]
        assert input_ids[0].item() == CLS_ID
        assert input_ids[-1].item() == SEP_ID


def test_labels_in_range(arithmetic_data):
    valid_labels = set(label_dict.values())
    for i in range(50):
        _, _, _, label = arithmetic_data.train[i]
        assert label in valid_labels


def test_attention_mask_all_ones(arithmetic_data):
    _, _, attn_mask, _ = arithmetic_data.train[0]
    assert attn_mask.shape == torch.Size([SEQ_LEN])
    assert all(v == 1.0 for v in attn_mask.tolist())
