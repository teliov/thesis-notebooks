#! /usr/bin/env bash

python main.py --run_name test_run \
    --train_file s3://qcedelft/dlruns/data/small_run_train.csv \
    --mlflow_uri http://mlflow.teliov.xyz \
    --input_dim 40 \
    --num_symptoms 33 \
    --num_conditions 14 \
    --visdom_url http://visdom.teliov.xyz \
    --visdom_port 80 \
    --visdom_username teliov \
    --visdom_password visdomadmin1234 \
    --visdom_env test_run \
    --layer_config_file s3://qcedelft/dlruns/data/basic_layer_config.json
