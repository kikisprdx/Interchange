import os
from transformers import BertTokenizer
from reproduction_datasets.arithmetic import generate_data, ArithmeticData


DATA_FILE = "data/arithmetic.csv"


def main(n=100_000):
    if not os.path.exists(DATA_FILE):
        os.makedirs("data", exist_ok=True)
        generate_data()
        print(f"generated {DATA_FILE}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    os.makedirs("/tmp/arithmetic_vocab", exist_ok=True)
    vocab_path = tokenizer.save_vocabulary("/tmp/arithmetic_vocab")[0]

    data = ArithmeticData(DATA_FILE, vocab_path)
    if n is not None:
        from torch.utils.data import Subset
        data.train = Subset(data.train, range(min(n, len(data.train))))
    print(f"train: {len(data.train)}  dev: {len(data.dev)}  test: {len(data.test)}")

    x, mask, attn, y = data.train[0]
    print("sample input ids:", x.tolist())
    print("sample label:    ", y)
    print("decoded:         ", tokenizer.convert_ids_to_tokens(x.tolist()))
