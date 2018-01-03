#!/usr/bin/env bash

!/bin/bash

PATTERN="Flights-00001*"


BUCKET=flights_gcmle
REGION=us-central1
OUTPUT_DIR=gs://${BUCKET}/flights/chapter9/output
DATA_DIR=gs://${BUCKET}/flights/chapter8/output
JOBNAME=flights_$(date -u +%y%m%d_%H%M%S)
gcloud ml-engine jobs submit training $JOBNAME \
  --region=$REGION \
  --module-name=trainer.task \
  --package-path=$(pwd)../trainer \
  --job-dir=$OUTPUT_DIR \
  --staging-bucket=gs://$BUCKET \
  --scale-tier=STANDARD_1 \
   --output_dir=$OUTPUT_DIR \
   --traindata $DATA_DIR/train$PATTERN \
   --evaldata $DATA_DIR/test$PATTERN
