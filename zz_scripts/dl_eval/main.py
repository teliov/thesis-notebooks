from thesislib.utils.dl.models import DNN, DEFAULT_LAYER_CONFIG
from thesislib.utils.dl.data import DLSparseMaker
from thesislib.utils.dl.utils import compute_top_n, compute_precision, get_cnf_matrix

import json
import os
import pandas as pd
from timeit import default_timer as timer
import pathlib
import argparse

import torch

AGE_MEAN = 38.741316816862515
AGE_STD = 23.380120690086834

NUM_SYMPTOMS = 376
NUM_CONDITIONS = 801
INPUT_DIM = 383


def eval(state_dict_path, train_file_path, output_dir, run_name):
    begin = timer()
    if not os.path.exists(state_dict_path):
        raise ValueError("Invalid state dict path passed")

    if not os.path.exists(train_file_path):
        raise ValueError("Invalid train path passed")

    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)

    state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))

    model = DNN(input_dim=INPUT_DIM, output_dim=NUM_CONDITIONS, layer_config=DEFAULT_LAYER_CONFIG)
    model.load_state_dict(state_dict)

    df = pd.read_csv(train_file_path, index_col="Index")
    labels = df.LABEL
    df = df.drop(columns=['LABEL'])

    num_samples = labels.shape[0]

    sparsifier = DLSparseMaker(num_symptoms=NUM_SYMPTOMS, age_mean=AGE_MEAN, age_std=AGE_STD)
    df = sparsifier.transform(df)

    with torch.no_grad():
        labels = torch.LongTensor(labels)
        df = torch.FloatTensor(df.todense())

        out = model(df)
        _, y_pred = torch.max(out, dim=1)
        unique_labels = torch.LongTensor(range(NUM_CONDITIONS))

        cnf = get_cnf_matrix(labels, y_pred, unique_labels)

        precision = compute_precision(cnf)
        accuracy = torch.sum(torch.diagonal(cnf)) / num_samples
        top_5_acc, _ = compute_top_n(out, labels, 5)
        top_5_acc /= num_samples

    filename = os.path.join(output_dir, "%s_dl_eval.json" % run_name)
    with open(filename, "w") as fp:
        json.dump({
            "precision": precision,
            "accuracy": accuracy.item(),
            "top5": top_5_acc
        }, fp)

    duration = timer() - begin
    print("Took : %.7f seconds" % duration)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL Evaluator")

    parser.add_argument('--state_dict_path', type=str, help='Path to State Dict')
    parser.add_argument('--data_path', type=str, help='Path to data file')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--run_name', type=str, help='Name of this run')

    args = parser.parse_args()

    state_dict_path = args.state_dict_path
    data_path = args.data_path
    output_dir = args.output_dir
    run_name = args.run_name

    # def eval(state_dict_path, train_file_path, output_dir, run_name):
    eval(state_dict_path, data_path, output_dir, run_name)
