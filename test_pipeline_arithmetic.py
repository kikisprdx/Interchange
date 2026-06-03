"""
test_pipeline_arithmetic.py

Integration tests and a short batched training loop for everything under
pipeline/, driven by mock arithmetic data generated in-memory (no file I/O).

Run:
    cd /scratch/ysurange/Interchange
    python test_pipeline_arithmetic.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from reproduction_datasets.arithmetic import ArithmeticDataset, label_dict, vocab, numbers, ops, outcomes
from pipeline.graph       import ComputationGraph
from pipeline.graph_input import GraphInput
from pipeline.graph_node  import GraphNode
from pipeline.intervention import Intervention
from pipeline.location    import LOC
from pipeline.utils       import copy_helper, serialize, deserialize


# ---------------------------------------------------------------------------
# Constants for the toy model
# ---------------------------------------------------------------------------

VOCAB_SIZE  = len(vocab)    # 13 tokens
EMBED_DIM   = 8
HIDDEN_DIM  = 16
N_CLASSES   = 3             # positive / negative / zero


# ---------------------------------------------------------------------------
# Mock data — same format as datasets/arithmetic.py, but in-memory
# ---------------------------------------------------------------------------

def make_lines():
    """Return all 200 arithmetic expression lines without writing to disk."""
    lines = []
    for x in numbers:
        for y in numbers:
            for op in ops:
                z = x + y if op == "+" else x - y
                outcome = outcomes[0] if z > 0 else (outcomes[1] if z < 0 else outcomes[2])
                lines.append(f"{x} {op} {y}, {z}, {outcome}")
    return lines


# ---------------------------------------------------------------------------
# Toy arithmetic model wired as a ComputationGraph
#
#   tokens (leaf) → embedding → hidden → logits (root)
# ---------------------------------------------------------------------------

def build_graph():
    """
    Build a fresh ComputationGraph + return all trainable parameters.

    Graph topology:
        tokens  (leaf)   – pass-through, shape (B, 3)
        embedding        – Embedding → flatten  → (B, 3*EMBED_DIM)
        hidden           – Linear + ReLU        → (B, HIDDEN_DIM)
        logits   (root)  – Linear               → (B, N_CLASSES)
    """
    embed_layer  = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
    hidden_mod   = nn.Linear(EMBED_DIM * 3, HIDDEN_DIM)
    out_mod      = nn.Linear(HIDDEN_DIM, N_CLASSES)

    @GraphNode()
    def tokens(x):
        return x

    @GraphNode(tokens)
    def embedding(x):
        return embed_layer(x).flatten(start_dim=1)

    @GraphNode(embedding)
    def hidden(x):
        return torch.relu(hidden_mod(x))

    @GraphNode(hidden)
    def logits(x):
        return out_mod(x)

    graph  = ComputationGraph(logits)
    params = (list(embed_layer.parameters())
              + list(hidden_mod.parameters())
              + list(out_mod.parameters()))
    return graph, params


# ---------------------------------------------------------------------------
# Test 1 – GraphInput
# ---------------------------------------------------------------------------

def test_graph_input():
    print("\n[1] GraphInput")
    w2i = {w: i for i, w in enumerate(vocab)}

    # Single example
    x  = torch.tensor([w2i["3"], w2i["+"], w2i["5"]])
    gi = GraphInput({"tokens": x})
    assert "tokens" in gi
    assert len(gi) == 1
    assert torch.equal(gi["tokens"], x)
    print("  single GraphInput ✓")

    # Batched
    batch      = torch.stack([x, x])          # shape (2, 3)
    gi_batched = GraphInput({"tokens": batch}, batched=True, keys=["ex0", "ex1"])
    assert gi_batched.batched
    assert gi_batched.keys == ["ex0", "ex1"]
    assert torch.equal(gi_batched["tokens"], batch)
    print("  batched GraphInput ✓")

    # Immutability: the .values property must be read-only
    try:
        gi.values = {}
        raise AssertionError("Expected error on .values assignment")
    except (AttributeError, RuntimeError):
        pass
    print("  immutability ✓")


# ---------------------------------------------------------------------------
# Test 2 – ComputationGraph: forward pass and caching
# ---------------------------------------------------------------------------

def test_graph_forward():
    print("\n[2] ComputationGraph forward pass")
    w2i   = {w: i for i, w in enumerate(vocab)}
    graph, _ = build_graph()

    x  = torch.tensor([[w2i["3"], w2i["+"], w2i["5"]]])
    gi = GraphInput({"tokens": x})

    out = graph.compute(gi)
    assert out.shape == (1, N_CLASSES), f"Expected (1, {N_CLASSES}), got {out.shape}"
    print(f"  output shape: {tuple(out.shape)} ✓")

    # Second call must hit the top-level cache (same tensor object)
    out2 = graph.compute(gi)
    assert torch.equal(out, out2)
    print("  result caching ✓")

    # Iterative (Kahn's) traversal must match recursive
    graph.clear_caches()
    r_rec = graph.compute(gi, iterative=False)
    graph.clear_caches()
    r_it  = graph.compute(gi, iterative=True)
    assert torch.equal(r_rec, r_it)
    print("  iterative == recursive ✓")


# ---------------------------------------------------------------------------
# Test 3 – get_result: intermediate node outputs
# ---------------------------------------------------------------------------

def test_get_result():
    print("\n[3] get_result (intermediate nodes)")
    w2i   = {w: i for i, w in enumerate(vocab)}
    graph, _ = build_graph()

    x  = torch.tensor([[w2i["2"], w2i["-"], w2i["7"]]])
    gi = GraphInput({"tokens": x})

    emb = graph.get_result("embedding", gi)
    assert emb.shape == (1, EMBED_DIM * 3), f"embedding shape: {emb.shape}"
    print(f"  embedding shape: {tuple(emb.shape)} ✓")

    hid = graph.get_result("hidden", gi)
    assert hid.shape == (1, HIDDEN_DIM), f"hidden shape: {hid.shape}"
    print(f"  hidden shape:    {tuple(hid.shape)} ✓")

    out = graph.get_result("logits", gi)
    assert out.shape == (1, N_CLASSES)
    print(f"  logits shape:    {tuple(out.shape)} ✓")


# ---------------------------------------------------------------------------
# Test 4 – Intervention: full-node and location-based patches
# ---------------------------------------------------------------------------

def test_intervention():
    print("\n[4] Intervention")
    w2i   = {w: i for i, w in enumerate(vocab)}
    graph, _ = build_graph()

    x  = torch.tensor([[w2i["1"], w2i["+"], w2i["2"]]])
    gi = GraphInput({"tokens": x})

    # 4a. Full-node patch: zero out the hidden layer → output must change
    patch  = torch.zeros(1, HIDDEN_DIM)
    interv = Intervention(gi, {"hidden": patch})
    base_out, patched_out = graph.intervene(interv)

    assert base_out.shape   == (1, N_CLASSES)
    assert patched_out.shape == (1, N_CLASSES)
    assert not torch.equal(base_out, patched_out), "Full-node intervention had no effect"
    assert interv.affected_nodes == {"hidden", "logits"}
    print("  full-node patch changes output ✓")
    print(f"  affected nodes: {sorted(interv.affected_nodes)} ✓")

    # 4b. Location-based patch: overwrite only first 4 hidden dims
    graph.clear_caches()
    partial      = torch.zeros(1, 4)
    loc_interv   = Intervention(gi, {"hidden": partial}, location={"hidden": LOC[:, :4]})
    base2, patched2 = graph.intervene(loc_interv)
    assert patched2.shape == (1, N_CLASSES)
    print("  location-based patch ✓")

    # 4c. Patch leaf node → all downstream nodes are affected
    graph.clear_caches()
    leaf_patch  = torch.tensor([[w2i["0"], w2i["-"], w2i["0"]]])
    leaf_interv = Intervention(gi, {"tokens": leaf_patch})
    base3, patched3 = graph.intervene(leaf_interv)
    assert leaf_interv.affected_nodes == {"tokens", "embedding", "hidden", "logits"}
    print("  leaf patch affects full graph ✓")

    # 4d. Empty intervention must raise
    graph.clear_caches()
    try:
        graph.intervene(Intervention(gi))
        raise AssertionError("Expected RuntimeError for empty intervention")
    except RuntimeError:
        pass
    print("  empty intervention raises RuntimeError ✓")


# ---------------------------------------------------------------------------
# Test 5 – utils: copy_helper, serialize / deserialize
# ---------------------------------------------------------------------------

def test_utils():
    print("\n[5] utils")

    # copy_helper: detached clone for tensors
    t      = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    cloned = copy_helper(t)
    assert torch.equal(t, cloned) and cloned is not t
    assert not cloned.requires_grad
    cloned[0] = 99.0
    assert t[0].item() == 1.0, "copy_helper must not alias the original"
    print("  copy_helper ✓")

    # serialize / deserialize round-trip for 1-D and 2-D tensors
    for original in [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    ]:
        reconstructed = deserialize(serialize(original))
        assert torch.equal(original, reconstructed), (
            f"Round-trip failed for shape {original.shape}"
        )
    print("  serialize / deserialize round-trip ✓")


# ---------------------------------------------------------------------------
# Test 6 – ArithmeticDataset from mock lines
# ---------------------------------------------------------------------------

def test_arithmetic_dataset():
    print("\n[6] ArithmeticDataset (mock lines)")
    w2i   = {w: i for i, w in enumerate(vocab)}
    lines = make_lines()
    ds    = ArithmeticDataset(lines, w2i)

    assert len(ds) == len(lines) == 200, f"Expected 200 examples, got {len(ds)}"
    x, y = ds[0]
    assert x.shape == (3,),         f"x shape: {x.shape}"
    assert x.dtype == torch.long
    assert y in (0, 1, 2),          f"label out of range: {y}"
    print(f"  dataset size: {len(ds)}, sample x={x.tolist()}, y={y} ✓")

    # Verify every label is reachable
    all_labels = {ds[i][1] for i in range(len(ds))}
    assert all_labels == {0, 1, 2}, f"Missing labels: {all_labels}"
    print("  all three labels present ✓")


# ---------------------------------------------------------------------------
# Test 7 – Short batched training loop
# ---------------------------------------------------------------------------

def test_training():
    print("\n[7] Short batched training loop")
    w2i   = {w: i for i, w in enumerate(vocab)}
    lines = make_lines()
    ds    = ArithmeticDataset(lines, w2i)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    graph, params = build_graph()
    optimizer = torch.optim.Adam(params, lr=5e-3)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 10
    epoch_losses = []

    for epoch in range(EPOCHS):
        running = 0.0
        for x_batch, y_batch in loader:
            # Disable result caching during training to avoid memory buildup
            # and stale-cache interference between optimiser steps.
            gi = GraphInput({"tokens": x_batch}, cache_results=False)

            optimizer.zero_grad()
            output = graph.compute(gi, store_cache=False)
            loss   = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            running += loss.item()
            graph.clear_caches()   # housekeeping — no-op when cache_results=False

        avg = running / len(loader)
        epoch_losses.append(avg)
        print(f"  epoch {epoch + 1:2d}/{EPOCHS}  loss = {avg:.4f}")

    first_half_avg = sum(epoch_losses[:5])  / 5
    second_half_avg = sum(epoch_losses[5:]) / 5
    assert second_half_avg < first_half_avg, (
        f"Loss did not decrease: first-half avg={first_half_avg:.4f}, "
        f"second-half avg={second_half_avg:.4f}"
    )
    print(f"  loss decreased ({first_half_avg:.4f} → {second_half_avg:.4f}) ✓")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_graph_input()
    test_graph_forward()
    test_get_result()
    test_intervention()
    test_utils()
    test_arithmetic_dataset()
    test_training()
    print("\n✓  All tests passed")
