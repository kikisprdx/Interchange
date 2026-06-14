# Interchange

Causal abstraction analysis (Geiger et al., 2021) tests whether a neural network's internal representations align with an interpretable causal model by performing interchange interventions. By swapping activations between inputs mid-forward-pass and checking whether the output matches the high-level model's prediction the method improves conventional probing technques to causal verification.

This repo reproduces the original MQNLI experiments and extends the method to a synthetic arithmetic dataset. The arithmetic task is formally defined as a three-node SCM: operands X, Y ∈ [-200, 200] and Op ∈ {+, -} determine an intermediate result Z, which maps to a sign label. Where MQNLI tests compositional logical inference over natural language, arithmetic provides a fully enumerable, deterministic causal structure with a single mediating variable.

MQNLI reproduction reached 74% accuracy (vs. 88.5% in the original), yielding only shallow syntactic structures. Arithmetic on the other hand reached 99% accuracy but the three-class head collapsed the result and sign into identical interchange targets, making their separation ambigious.

## Setup

```bash
poetry install
```

Requires Python 3.12.

## Usage

**1. Generate arithmetic data**

```bash
python main.py generate arithmetic
```

**2. Generate MQNLI data**

```bash
python main.py generate mqnli
```

**3. Train arithmetic BERT**

```bash
python main.py train arithmetic
```

Saves to `checkpoints/bert_arithmetic_finetuned.pt`.

**4. Train MQNLI BERT**

```bash
python main.py train mqnli
```

Saves to `checkpoints/bert_mqnli_finetuned.pt`.

**5. Run arithmetic interchange**

```bash
python main.py interchange arithmetic setup -d results/experiments.csv -m checkpoints/bert_arithmetic_finetuned.pt -i data/arithmetic_preprocessed.pt
python main.py interchange arithmetic add -d results/experiments.csv -m checkpoints/bert_arithmetic_finetuned.pt -i data/arithmetic_preprocessed.pt
python main.py interchange arithmetic run -d results/experiments.csv
```

**6. Run MQNLI interchange**

```bash
python main.py interchange mqnli setup -d results/experiments_mqnli.csv -m checkpoints/bert_mqnli_finetuned.pt -i data/mqnli_preprocessed.pt
python main.py interchange mqnli add -d results/experiments_mqnli.csv -m checkpoints/bert_mqnli_finetuned.pt -i data/mqnli_preprocessed.pt
python main.py interchange mqnli run -d results/experiments_mqnli.csv
```

Results are written to `results/`.
