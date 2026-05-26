import random
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

numbers = range(-200, 201)
ops = ['+', '-']
outcomes = ['positive', 'negative', 'zero']

label_dict = {"positive": 0, "negative": 1, "zero": 2}
SEQ_LEN = 5  # [CLS] x op y [SEP]

def generate_data():
    with open("data/arithmetic.csv", "w") as f:
        for x in numbers:
            for y in numbers:
                for op in ops:
                    z = x + y if op=='+' else x - y
                    if z > 0:
                        out = outcomes[0]
                    elif z < 0:
                        out = outcomes[1]
                    else:
                        out = outcomes[2]
                    row = f"{str(x)} {op} {str(y)}, {z}, {out}\n"
                    f.write(row)

class ArithmeticData:
    def __init__(self, data_file, tokenizer_vocab_path):
        self.tokenizer = BertTokenizer(tokenizer_vocab_path)
        self.output_classes = 3

        with open(data_file) as f:
            lines = f.read().splitlines()

        random.shuffle(lines)

        n = len(lines)
        n_train = int(n * 0.7)
        n_dev = int(n * 0.15)

        self.train = ArithmeticDataset(lines[:n_train], self.tokenizer)
        self.dev   = ArithmeticDataset(lines[n_train:n_train + n_dev], self.tokenizer)
        self.test  = ArithmeticDataset(lines[n_train + n_dev:], self.tokenizer)

class ArithmeticDataset(Dataset):
    def __init__(self, lines, tokenizer):
        self.raw_x = []
        self.raw_y = []
        for line in lines:
            expr, _, outcome = line.split(", ")
            x, op, y = expr.split()
            toks = ["[CLS]", x, op, y, "[SEP]"]
            ids = tokenizer.convert_tokens_to_ids(toks)
            self.raw_x.append(ids)
            self.raw_y.append(label_dict[outcome])
        self.num_examples = len(self.raw_x)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, i):
        return (torch.tensor(self.raw_x[i], dtype=torch.long),
                torch.tensor([0] * SEQ_LEN, dtype=torch.long),
                torch.tensor([1.] * SEQ_LEN, dtype=torch.float),
                self.raw_y[i])


if __name__ == "__main__":
    generate_data()