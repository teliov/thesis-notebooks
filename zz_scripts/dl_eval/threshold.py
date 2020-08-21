from thesislib.utils.dl.models import DNN, DEFAULT_LAYER_CONFIG
from thesislib.utils.dl.data import DLSparseMaker

import os
import pandas as pd
from timeit import default_timer as timer
import pathlib
import numpy as np
import argparse

import torch
import torch.nn.functional as F
import joblib

AGE_MEAN = 38.741316816862515
AGE_STD = 23.380120690086834

NUM_SYMPTOMS = 376
NUM_CONDITIONS = 801
INPUT_DIM = 383


def extract_stats(prob_data, prob_index, labels, threshold):

    index_threshold = prob_data[:, 0] >= threshold

    # num_predictions
    num_allowed_predictions = index_threshold.shape[0]

    # how many are accurate
    _labels = labels[num_allowed_predictions]
    _prob = prob_data[num_allowed_predictions, :]
    _index = prob_index[num_allowed_predictions, :]

    top_1 = np.sum(_index[:, 0] == _labels)
    top_5 = np.sum(_index == _labels.reshape(-1, 1))
    top_0 = num_allowed_predictions - top_1
    top_n5 = num_allowed_predictions - top_5

    return {
        "allowed_predictions": num_allowed_predictions,
        "num_top1_accurate": top_1,
        "num_top5_accurate": top_5,
        "num_top1_inaccurate": top_0,
        "num_top5_inaccurate": top_n5
    }


def threshold_definitions(state_dict_path, threshold_path, test_file_path, output_dir):
    begin = timer()
    if not os.path.exists(state_dict_path):
        raise ValueError("Invalid state dict path passed")

    if not os.path.exists(test_file_path):
        raise ValueError("Invalid test path passed")

    if not os.path.exists(threshold_path):
        raise ValueError("Invalid threshold path")

    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
    state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))

    model = DNN(input_dim=INPUT_DIM, output_dim=NUM_CONDITIONS, layer_config=DEFAULT_LAYER_CONFIG)
    model.load_state_dict(state_dict)

    df = pd.read_csv(test_file_path, index_col="Index")
    labels = df.LABEL.values
    df = df.drop(columns=['LABEL'])

    sparsifier = DLSparseMaker(num_symptoms=NUM_SYMPTOMS, age_mean=AGE_MEAN, age_std=AGE_STD)
    df = sparsifier.transform(df)

    with torch.no_grad():
        df = torch.FloatTensor(df.todense())

        out = model(df)
        predicted_prob = F.softmax(out, dim=1)
        sorted_prob, sorted_index = torch.sort(predicted_prob, descending=True, dim=1)

        sorted_prob = sorted_prob[:, :5].numpy()
        sorted_index = sorted_index[:, :5].numpy()

    threshold_data = joblib.load(threshold_path)
    mlp_data = threshold_data.get("mlp")

    quantiles = mlp_data.get("quantiles")
    means = mlp_data.get("means")

    data = {}
    map = ["top1", "top5", "ntop1", "ntop5"]

    for idx in range(len(map)):
        scores = []
        quantile_list = quantiles[idx]
        _mean = means[idx]

        for jdx in range(quantile_list.shape[0]):
            _val = quantile_list[jdx]
            res = extract_stats(
                sorted_prob,
                sorted_index,
                labels,
                _val
            )

            scores.append((_val, res))

        res = extract_stats(
            sorted_prob,
            sorted_index,
            labels,
            _mean
        )
        scores.append((_mean, res))

        data[map[idx]] = scores

    filename = os.path.join(output_dir, "mlp_threshold.joblib")
    joblib.dump(data, filename)

    duration = timer() - begin
    print("Took : %.7f seconds" % duration)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL Threshold analysis")

    parser.add_argument('--state_dict_path', type=str, help='Path to State Dict')
    parser.add_argument('--data_path', type=str, help='Path to data file')
    parser.add_argument('--threshold_path', type=str, help='Path to threshold joblib dump')
    parser.add_argument('--output_dir', type=str, help='Output directory')

    args = parser.parse_args()

    state_dict_path = args.state_dict_path
    data_path = args.data_path
    output_dir = args.output_dir
    threshold_path = args.threshold_path

    threshold_definitions(
        state_dict_path,
        threshold_path,
        data_path,
        output_dir
    )
