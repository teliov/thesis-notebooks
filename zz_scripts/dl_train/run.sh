#! /usr/bin/env bash

train_file=$1
input_dim=$2
num_symptoms=$3
num_conditions=$4
layer_config_file=$5

python main.py --run_name test_run \
    --train_file $train_file \
    --mlflow_uri $MLFLOW_URL \
    --input_dim $input_dim \
    --num_symptoms $num_symptoms \
    --num_conditions $num_conditions \
    --visdom_url $VISDOM_URL \
    --visdom_port $VISDOM_PORT \
    --visdom_username $VISDOM_USERNAME \
    --visdom_password $VISDOM_PASSWORD \
    --visdom_env dl_run \
    --layer_config_file $layer_config_file
