#!/usr/bin/env bash
while grep -q ",0," experiments_mqnli.csv; do
  poetry run python main.py interchange mqnli run \
    -d experiments_mqnli.csv \
    --script "poetry run python mqnli_interchange.py"
done && echo "all done"
