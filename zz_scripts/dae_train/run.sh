#! /usr/bin/env bash

train_file=$1
input_dim=$2
target_dim=$3
num_symptoms=$4
epoch_count=20

python main.py --run_name dae_run \
    --train_file $train_file \
    --mlflow_uri $MLFLOW_URL \
    --input_dim ${input_dim} \
    --target_dim ${target_dim} \
    --num_symptoms ${num_symptoms} \
    --visdom_url $VISDOM_URL \
    --visdom_port $VISDOM_PORT \
    --visdom_username $VISDOM_USERNAME \
    --visdom_password $VISDOM_PASSWORD \
    --visdom_env mac_dae \
    --epochs ${epoch_count}

