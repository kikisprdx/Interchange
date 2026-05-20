import torch 
import numpy as np
import random
from torch.utils.data import Dataset

numbers = range(0,10)
ops = ['+', '-']
outcomes = ['positive', 'negative', 'zero']

label_dict = {"positive": 0, "negative": 1, "zero": 2}
vocab = ["[PAD]", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-"]

def generate_data():
    with open("data/arithmetic.txt", "w") as f:
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
    def __init__(self, data_file):
        train_ratio = 0.7
        dev_ratio = 0.15
        self.word_to_id = {w: i for i, w in enumerate(vocab)}
        self.id_to_word = {i: w for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.output_classes = 3

        with open(data_file) as f:
            lines = f.read().splitlines()

        random.shuffle(lines)

        n = len(lines)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)

        self.train = ArithmeticDataset(lines[:n_train], self.word_to_id)
        self.dev   = ArithmeticDataset(lines[n_train:n_train + n_dev], self.word_to_id)
        self.test  = ArithmeticDataset(lines[n_train + n_dev:], self.word_to_id)

class ArithmeticDataset(Dataset):
    def __init__(self, lines, word_to_id):
        self.raw_x = []
        self.raw_y = []
        for line in lines:
            expr, _, outcome = line.split(", ")
            x, op, y = expr.split()
            ids = np.array([word_to_id[x], word_to_id[op], word_to_id[y]])
            self.raw_x.append(ids)
            self.raw_y.append(label_dict[outcome])
        self.num_examples = len(self.raw_x) 

    def __len__(self):
        return self.num_examples

    def __getitem__(self, i):
        return (torch.tensor(self.raw_x[i], dtype=torch.long),
                self.raw_y[i])


if __name__ == "__main__":
    generate_data()