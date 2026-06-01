"""
Unit tests for ArithmeticBertModule.

Checks:
    1. Model loads without crashing
    2. forward() returns the right output shapes
    3. hidden_states has exactly 13 layers
    4. Each hidden state has the right shape (batch, seq_len, 768)
    5. Tokenizer works on a real arithmetic expression
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from modeling.arithmetic_bert import ArithmeticBertModule


def test_output_shapes():
    print("Loading model... (downloads weights on first run, ~400MB)")
    model = ArithmeticBertModule(num_labels=3)
    model.eval()

    # tokenize two example expressions as a batch
    expressions = ["3 + 2", "5 - 9"]
    inputs = model.tokenizer(
        expressions,
        return_tensors="pt",
        padding=True,   # pad to same length within batch
    )

    print(f"Input token ids shape: {inputs['input_ids'].shape}")
    # expect (2, 5) — batch of 2, sequence length 5: [CLS] X op Y [SEP]

    with torch.no_grad():
        logits, hidden_states = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    # --- check logits ---
    assert logits.shape == (2, 3), \
        f"Expected logits shape (2, 3), got {logits.shape}"
    print(f"logits shape: {logits.shape}  ✓")

    # --- check hidden states ---
    assert len(hidden_states) == 13, \
        f"Expected 13 hidden state tensors, got {len(hidden_states)}"
    print(f"number of hidden state layers: {len(hidden_states)}  ✓")
    # (layer 0 = embeddings, layers 1-12 = transformer layers)

    seq_len = inputs["input_ids"].shape[1]
    for i, hs in enumerate(hidden_states):
        assert hs.shape == (2, seq_len, 768), \
            f"Layer {i}: expected shape (2, {seq_len}, 768), got {hs.shape}"
    print(f"all hidden state shapes: (2, {seq_len}, 768)  ✓")

    # --- check predictions are valid class indices ---
    preds = logits.argmax(dim=-1)
    assert all(0 <= p < 3 for p in preds.tolist()), \
        f"Predictions out of range: {preds}"
    print(f"predictions (raw, untrained): {preds.tolist()}  ✓")

    print("\nAll checks passed.")


def test_config():
    model = ArithmeticBertModule(num_labels=3, dropout=0.1)
    cfg = model.config()
    assert cfg["num_labels"] == 3
    assert cfg["hidden_size"] == 768
    assert cfg["bert_model"] == "bert-base-uncased"
    print(f"config: {cfg}  ✓")


if __name__ == "__main__":
    test_output_shapes()
    test_config()
