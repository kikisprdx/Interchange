import argparse
import os
import subprocess
import sys
import torch
from transformers import BertTokenizer
from datasets.arithmetic import generate_data, ArithmeticData

VENDOR_MQNLI = os.path.join("vendor", "interchange", "mqnli")
MQNLI_SIZE = 50_000


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
            os.makedirs("/tmp/arithmetic_vocab", exist_ok=True)
            generate_data()
            print("generated data/arithmetic.csv")
            vocab_path = BertTokenizer.from_pretrained("bert-base-uncased").save_vocabulary("/tmp/arithmetic_vocab")[0]
            torch.save(ArithmeticData("data/arithmetic.csv", vocab_path), "data/arithmetic_preprocessed.pt")
            print("generated data/arithmetic_preprocessed.pt")

        elif args.task == "mqnli":
            save_dir = os.path.abspath("data/mqnli")
            os.makedirs(save_dir, exist_ok=True)
            data_path = os.path.abspath(os.path.join(VENDOR_MQNLI, "data"))
            subprocess.run(
                ["python", "generate_data.py", str(MQNLI_SIZE), save_dir, data_path],
                cwd=os.path.abspath(VENDOR_MQNLI),
                check=True,
            )
            print(f"generated mqnli data → {save_dir}")

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
