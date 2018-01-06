#!/usr/bin/env bash

# test out train command
python task.py --traindata ~/data/train.csv

# first training run
python task.py \
       --traindata ../data/train.csv \
       --output_dir ./trained_model \
       --evaldata ../data/test.csv

# run as python module
export PYTHONPATH=${PYTHONPATH}:${PWD}/trainer
python -m trainer.task \
   --output_dir=./trained_model \
  --traindata $DATA_DIR/train* --evaldata $DATA_DIR/test*