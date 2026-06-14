# Interchange

Reproduction of Geiger et al. (2021) causal abstraction / interchange intervention on BERT, extended to an arithmetic dataset.

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
