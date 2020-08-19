from thesislib.utils.dl.models import DNN, DEFAULT_LAYER_CONFIG
from thesislib.utils.dl.data import DLSparseMaker

import os
import pandas as pd
from timeit import default_timer as timer
import pathlib
import argparse

import torch
import joblib

AGE_MEAN = 38.741316816862515
AGE_STD = 23.380120690086834

NUM_SYMPTOMS = 376
NUM_CONDITIONS = 801
INPUT_DIM = 383



EVAL_CONDITIONS = [182, 200, 229, 414, 441, 788, 93, 629, 56]


def top_5_qual(state_dict_path, train_file_path, output_dir):
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
    df = df[df['LABEL'].isin(EVAL_CONDITIONS)]

    labels = df.LABEL.values
    df = df.drop(columns=['LABEL'])

    sparsifier = DLSparseMaker(num_symptoms=NUM_SYMPTOMS, age_mean=AGE_MEAN, age_std=AGE_STD)
    df = sparsifier.transform(df)

    with torch.no_grad():
        df = torch.FloatTensor(df.todense())
        labels = torch.LongTensor(labels)

        out = model(df)
        sorted_prob = torch.argsort(out, dim=1, descending=True)
        top_5 = sorted_prob[:, :5]

        # when it's top 5 accurate
        combined = top_5 == labels.view(-1, 1)
        combined = torch.sum(combined, dim=1).to(dtype=torch.bool)

        _top_5_acc = top_5[combined, :].numpy()
        _top_5_labels = labels[combined].numpy()
        op_file = os.path.join(output_dir, "mlp_top_5_pred.joblib")
        joblib.dump({
            'acc': _top_5_acc,
            'labels': _top_5_labels
        }, op_file)

        # when it's not top 5 accurate
        non_top_5 = combined == False
        _ntop_5_acc = top_5[non_top_5, :].numpy()
        _ntop_5_labels = labels[non_top_5].numpy()
        op_file = os.path.join(output_dir, "mlp_top_n5_pred.joblib")
        joblib.dump({
            'acc': _ntop_5_acc,
            'labels': _ntop_5_labels
        }, op_file)

    duration = timer() - begin
    print("Took : %.7f seconds" % duration)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL Top 5 analysis")

    parser.add_argument('--state_dict_path', type=str, help='Path to State Dict')
    parser.add_argument('--data_path', type=str, help='Path to data file')
    parser.add_argument('--output_dir', type=str, help='Output directory')

    args = parser.parse_args()

    state_dict_path = args.state_dict_path
    data_path = args.data_path
    output_dir = args.output_dir

    top_5_qual(state_dict_path, data_path, output_dir)
