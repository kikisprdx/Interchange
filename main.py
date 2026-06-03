import argparse
import os
import subprocess
import sys

sys.path.insert(1, os.path.join("vendor", "interchange"))

import torch
from transformers import BertTokenizer

from reproduction_datasets.arithmetic import ArithmeticData, generate_data
from datasets.mqnli import MQNLIBertData

VENDOR_MQNLI = os.path.join("vendor", "interchange", "mqnli")
MQNLI_SIZE = 50_000
ARITHMETIC_VOCAB_DIR = "data/arithmetic_vocab"
MQNLI_REMAPPING = os.path.join("vendor", "interchange", "data", "tokenization", "bert-remapping.txt")
MQNLI_DATA_DIR = "data/mqnli"


def main():
    parser = argparse.ArgumentParser(prog="interchange")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen_p = sub.add_parser("generate")
    gen_p.add_argument("task", choices=["arithmetic", "mqnli"])

    train_p = sub.add_parser("train")
    train_p.add_argument("task", choices=["arithmetic", "mqnli"])

    interchange_p = sub.add_parser("interchange")
    interchange_p.add_argument("task", choices=["arithmetic"])
    interchange_p.add_argument("rest", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.cmd == "generate":
        if args.task == "arithmetic":
            os.makedirs("data", exist_ok=True)
            os.makedirs(ARITHMETIC_VOCAB_DIR, exist_ok=True)
            generate_data()
            print("generated data/arithmetic.csv")
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            tokens_to_add = [str(i) for i in range(-200, 201) if str(i) not in tokenizer.vocab]
            tokenizer.add_tokens(tokens_to_add)
            vocab_path = tokenizer.save_vocabulary(ARITHMETIC_VOCAB_DIR)[0]
            with open(vocab_path, "a", encoding="utf-8") as f:
                for token in tokens_to_add:
                    f.write(token + "\n")
            torch.save(ArithmeticData("data/arithmetic.csv", vocab_path), "data/arithmetic_preprocessed.pt")
            print("generated data/arithmetic_preprocessed.pt")

        elif args.task == "mqnli":
            save_dir = os.path.abspath(MQNLI_DATA_DIR)
            os.makedirs(save_dir, exist_ok=True)
            data_path = os.path.abspath(os.path.join(VENDOR_MQNLI, "data"))
            subprocess.run(
                ["python", "generate_data.py", str(MQNLI_SIZE), save_dir, data_path],
                cwd=os.path.abspath(VENDOR_MQNLI),
                check=True,
            )
            print(f"generated mqnli data → {save_dir}")
            train_f = os.path.join(MQNLI_DATA_DIR, "0gendata.train")
            dev_f   = os.path.join(MQNLI_DATA_DIR, "0gendata.val")
            test_f  = os.path.join(MQNLI_DATA_DIR, "0gendata.test")
            torch.save(MQNLIBertData(train_f, dev_f, test_f, MQNLI_REMAPPING), "data/mqnli_preprocessed.pt")
            print("generated data/mqnli_preprocessed.pt")

    elif args.cmd == "train":
        if args.task == "arithmetic":
            from train_arithmetic_bert import main as train

            train()

        elif args.task == "mqnli":
            from train_mqnli_bert import main as train

            train()

    elif args.cmd == "interchange":
        if args.task == "arithmetic":
            sys.argv = [sys.argv[0]] + args.rest
            from arithmetic_interchange_manager import main as run_interchange

            run_interchange()


if __name__ == "__main__":
    main()
