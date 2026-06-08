#!/usr/bin/env bash
while grep -q ",0," experiments.csv; do
  poetry run python main.py interchange arithmetic run \
    --csv_path experiments.csv \
    --script "poetry run python arithmetic_interchange.py"
done && echo "all done"
