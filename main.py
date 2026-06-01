import argparse
import os


def main():
    parser = argparse.ArgumentParser(prog="interchange")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("generate")

    train_p = sub.add_parser("train")
    train_p.add_argument("task", choices=["arithmetic"])

    args = parser.parse_args()

    if args.cmd == "generate":
        from datasets.arithmetic import generate_data
        os.makedirs("data", exist_ok=True)
        generate_data()
        print("generated data/arithmetic.csv")

    elif args.cmd == "train":
        if args.task == "arithmetic":
            from train_arithmetic_bert import main as train
            train()


if __name__ == "__main__":
    main()
