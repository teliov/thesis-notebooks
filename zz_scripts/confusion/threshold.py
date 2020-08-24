import pandas as pd
import numpy as np
import os
import joblib
import argparse
import pathlib
from thesislib.utils.ml import models

NUM_SYMPTOMS = 376
NUM_CONDITIONS = 801
INPUT_DIM = 383


def extract_stats(prob_data, prob_index, labels, threshold):
    index_threshold = prob_data[:, 0] >= threshold
    _index_threshold = index_threshold == False

    # num_predictions
    num_allowed_predictions = np.sum(index_threshold)
    num_rejected_predictions = np.sum(_index_threshold)

    # how many are accurate
    _labels = labels[index_threshold]
    _index = prob_index[index_threshold, :]

    top_1 = np.sum(_index[:, 0] == _labels)
    top_5 = np.sum(_index == _labels.reshape(-1, 1))
    top_0 = num_allowed_predictions - top_1
    top_n5 = num_allowed_predictions - top_5

    # how many are inaccurate
    n_labels = labels[_index_threshold]
    n_index = prob_index[_index_threshold, :]

    ntop_1 = np.sum(n_index[:, 0] == n_labels)
    ntop_5 = np.sum(n_index == n_labels.reshape(-1, 1))
    ntop_0 = num_rejected_predictions - ntop_1
    ntop_n5 = num_rejected_predictions - ntop_5

    return {
        "allowed": {
            "count": num_allowed_predictions,
            "num_top1_accurate": top_1,
            "num_top5_accurate": top_5,
            "num_top1_inaccurate": top_0,
            "num_top5_inaccurate": top_n5
        },
        "rejected": {
            "count": num_rejected_predictions,
            "num_top1_accurate": ntop_1,
            "num_top5_accurate": ntop_5,
            "num_top1_inaccurate": ntop_0,
            "num_top5_inaccurate": ntop_n5
        }
    }


def threshold_definitions(model_path, model_type, threshold_path, data_file, output_dir):
    clf_data = joblib.load(model_path)

    if model_type == "random_forest":
        clf = clf_data.get("clf")
        threshold_key = "rf"
    else:
        clf_serialized = clf_data.get("clf")
        clf = models.ThesisSparseNaiveBayes.load(clf_serialized)
        threshold_key = "nb"

    data = pd.read_csv(data_file, index_col='Index')
    label_values = data.LABEL.values
    data = data.drop(columns=['LABEL'])

    sparsifier = models.ThesisSymptomRaceSparseMaker(num_symptoms=NUM_SYMPTOMS)
    data = sparsifier.fit_transform(data)

    predicted_prob = clf.predict_proba(data)

    # sort idx, max first
    sorted_index = np.argsort(-predicted_prob, axis=1)[:, :5]
    sorted_prob = -np.sort(-predicted_prob, axis=1)[:, :5]

    threshold_data = joblib.load(threshold_path)
    _data = threshold_data.get(threshold_key)

    quantiles = _data.get("quantiles")
    means = _data.get("means")

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
                label_values,
                _val
            )

            scores.append((_val, res))

        res = extract_stats(
            sorted_prob,
            sorted_index,
            label_values,
            _mean
        )
        scores.append((_mean, res))

        data[map[idx]] = scores

    filename = os.path.join(output_dir, "%s_threshold.joblib" % model_type)
    joblib.dump(data, filename)

    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Medvice Confusion Matrix Resolver')
    parser.add_argument('--data', help='Path to train csv file', type=str)
    parser.add_argument('--model_path', help='Path to saved model', type=str)
    parser.add_argument('--threshold_path', help='Path to saved model', type=str)
    parser.add_argument('--output_dir', help='Directory where confusion matrix should be saved to', type=str)
    parser.add_argument('--model_type', help='Type of the model', type=str)
    parser.add_argument('--num_symptoms', help='The number of symptoms', type=int, default=376)

    args = parser.parse_args()
    data_file = args.data
    model_path = args.model_path
    model_type = args.model_type
    output_dir = args.output_dir
    threshold_path = args.threshold_path

    if not os.path.isfile(data_file):
        raise ValueError("Data file does not exist")
    if not os.path.isfile(model_path):
        raise ValueError("Model file does not exist")
    if not os.path.isfile(threshold_path):
        raise ValueError("Threshold file does not exist")

    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)

    threshold_definitions(model_path, model_type, threshold_path, data_file, output_dir)
