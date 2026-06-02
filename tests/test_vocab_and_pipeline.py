"""
Sanity checks for the vocab fix and training pipeline.

Run from the Interchange/ directory:
    python tests/test_vocab_and_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import BertTokenizer

UNK_ID = 100  # BERT's [UNK] token id


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def test_vocab_fix():
    print("\n=== Vocab Fix ===")

    base = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens_to_add = [str(i) for i in range(-200, 201) if str(i) not in base.vocab]
    base.add_tokens(tokens_to_add)

    # negative numbers should now resolve to non-UNK ids
    for n in ["-1", "-50", "-200"]:
        id_ = base.convert_tokens_to_ids(n)
        check(f'"{n}" is not [UNK]', id_ != UNK_ID, f"id={id_}")

    # positive numbers were already in vocab
    for n in ["0", "47", "200"]:
        id_ = base.convert_tokens_to_ids(n)
        check(f'"{n}" already in base vocab', id_ != UNK_ID, f"id={id_}")

    check("vocab size grew by 200", len(base) == 30722, f"size={len(base)}")


def test_id_consistency():
    print("\n=== ID Consistency: ArithmeticData vs ArithmeticBertModule ===")

    # simulate what train_arithmetic_bert.py does
    import tempfile
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens_to_add = [str(i) for i in range(-200, 201) if str(i) not in tokenizer.vocab]
    tokenizer.add_tokens(tokens_to_add)

    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = tokenizer.save_vocabulary(tmpdir)[0]
        with open(vocab_path, 'a', encoding='utf-8') as f:
            for token in tokens_to_add:
                f.write(token + '\n')

        # simulate what ArithmeticData does
        data_tokenizer = BertTokenizer(vocab_path)

        # simulate what ArithmeticBertModule does
        from modeling.arithmetic_bert import ArithmeticBertModule
        model = ArithmeticBertModule(num_labels=3)

        for n in ["-200", "-1", "-50"]:
            data_id = data_tokenizer.convert_tokens_to_ids(n)
            model_id = model.tokenizer.convert_tokens_to_ids(n)
            check(f'"{n}" same ID in data and model', data_id == model_id,
                  f"data={data_id}  model={model_id}")
            check(f'"{n}" not UNK in data', data_id != UNK_ID, f"id={data_id}")


def test_forward_pass():
    print("\n=== Forward Pass With Dataset-style Input ===")

    from modeling.arithmetic_bert import ArithmeticBertModule
    model = ArithmeticBertModule(num_labels=3)
    model.eval()

    # mimic exactly what ArithmeticDataset.__getitem__ returns for "-50 + 47"
    tokenizer = model.tokenizer
    toks = ["[CLS]", "-50", "+", "47", "[SEP]"]
    ids = tokenizer.convert_tokens_to_ids(toks)

    check("no [UNK] tokens in sample", UNK_ID not in ids, f"ids={ids}")

    input_ids      = torch.tensor([ids], dtype=torch.long)
    token_type_ids = torch.zeros(1, 5, dtype=torch.long)
    attention_mask = torch.ones(1, 5, dtype=torch.float)

    with torch.no_grad():
        logits, hidden_states = model(input_ids, attention_mask, token_type_ids)

    check("logits shape is (1, 3)",       logits.shape == (1, 3),  f"shape={logits.shape}")
    check("hidden_states has 13 layers",  len(hidden_states) == 13, f"layers={len(hidden_states)}")
    check("embedding size matches vocab", model.bert.embeddings.word_embeddings.weight.shape[0] == len(model.tokenizer),
          f"embedding={model.bert.embeddings.word_embeddings.weight.shape[0]}  tokenizer={len(model.tokenizer)}")


def test_dataset_pipeline():
    print("\n=== Dataset Pipeline (requires data/arithmetic.csv) ===")

    data_file = "data/arithmetic.csv"
    if not os.path.exists(data_file):
        print(f"[SKIP] {data_file} not found — run generate_data() first")
        return

    import tempfile
    from datasets.arithmetic import ArithmeticData

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens_to_add = [str(i) for i in range(-200, 201) if str(i) not in tokenizer.vocab]
    tokenizer.add_tokens(tokens_to_add)

    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = tokenizer.save_vocabulary(tmpdir)[0]
        with open(vocab_path, 'a', encoding='utf-8') as f:
            for token in tokens_to_add:
                f.write(token + '\n')

        data = ArithmeticData(data_file, vocab_path)
        batch = data.train[0]
        input_ids = batch[0]

        check("batch has 4 elements",       len(batch) == 4)
        check("input_ids length is 5",      len(input_ids) == 5,    f"len={len(input_ids)}")
        check("no UNK in first example",    UNK_ID not in input_ids.tolist(),
              f"ids={input_ids.tolist()}")
        check("label is 0, 1, or 2",        batch[3] in [0, 1, 2],  f"label={batch[3]}")


if __name__ == "__main__":
    test_vocab_fix()
    test_id_consistency()
    test_forward_pass()
    test_dataset_pipeline()
    print("\nDone.")
