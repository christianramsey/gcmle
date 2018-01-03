#!/usr/bin/env bash

# test out train command
python task.py --traindata ~/data/train.csv

# first training run
python task.py \
       --traindata ../data/train.csv \
       --output_dir ./trained_model \
       --evaldata ../data/test.csv
