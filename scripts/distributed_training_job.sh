#  distributed_training_job.sh
#!/usr/bin/env bash

!/bin/bash

# run locally
python task.py \
       --traindata ../data/train.csv \
       --output_dir ../trained_model \
       --evaldata ../data/test.csv




PATTERN="Flights-00001*"


BUCKET=flights_gcmle
REGION=us-central1
OUTPUT_DIR=gs://${BUCKET}/flights/chapter9/output
DATA_DIR=gs://${BUCKET}/flights/chapter8/output
JOBNAME=flights_$(date -u +%y%m%d_%H%M%S)
gcloud ml-engine jobs submit training $JOBNAME \
  --region=$REGION \
  --module-name=trainer.task \
  --package-path=$(pwd) \
  --job-dir=$OUTPUT_DIR \
  --staging-bucket=gs://$BUCKET \
  --scale-tier=STANDARD_1 \
  -- \
   --output_dir=$OUTPUT_DIR \
   --traindata $DATA_DIR/train$PATTERN \
   --evaldata $DATA_DIR/test$PATTERN



#   premium gpus
BUCKET=flights_gcmle
REGION=us-central1
OUTPUT_DIR=gs://${BUCKET}/flights/chapter9/output
DATA_DIR=gs://${BUCKET}/flights/chapter8/output
JOBNAME=flights_$(date -u +%y%m%d_%H%M%S)
gcloud ml-engine jobs submit training $JOBNAME \
  --region=$REGION \
  --module-name=trainer.task \
  --package-path=$(pwd) \
  --job-dir=$OUTPUT_DIR \
  --staging-bucket=gs://$BUCKET \
  --scale-tier=PREMIUM_1 \
  -- \
   --output_dir=$OUTPUT_DIR \
   --traindata $DATA_DIR/train$PATTERN \
   --evaldata $DATA_DIR/test$PATTERN
