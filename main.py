import argparse

TASKS = ["arithmetic"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("-n", "--n", type=int, default=None)
    args = parser.parse_args()

    if args.task == "arithmetic":
        from run_arithmetic import main as run
        run(n=args.n)


if __name__ == "__main__":
    main()
