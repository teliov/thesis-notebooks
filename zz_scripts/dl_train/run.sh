#! /usr/bin/env bash

train_file=$1

python main.py --run_name dl_run \
    --train_file $train_file \
    --mlflow_uri $MLFLOW_URL
