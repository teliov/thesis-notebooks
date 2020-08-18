from thesislib.utils.dl.models import DNN, DEFAULT_LAYER_CONFIG
from thesislib.utils.dl.data import DLSparseMaker

import os
import pandas as pd
from timeit import default_timer as timer
import pathlib
import argparse

import torch
import torch.nn.functional as F
import joblib

AGE_MEAN = 38.741316816862515
AGE_STD = 23.380120690086834

NUM_SYMPTOMS = 376
NUM_CONDITIONS = 801
INPUT_DIM = 383


def confidence(state_dict_path, train_file_path, output_dir):
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

    sparsifier = DLSparseMaker(num_symptoms=NUM_SYMPTOMS, age_mean=AGE_MEAN, age_std=AGE_STD)
    df = sparsifier.transform(df)

    with torch.no_grad():
        labels = torch.LongTensor(labels)
        df = torch.FloatTensor(df.todense())

        out = model(df)
        predicted_prob = F.softmax(out, dim=1)
        sorted_predicted_prob, sort_idx = torch.sort(predicted_prob, dim=1, descending=True)

        predicted_labels = sort_idx[:, 0]

        # for all the data
        op_file = os.path.join(output_dir, "mlp_confidence_matrix_all.joblib")
        joblib.dump(sorted_predicted_prob[:, :5].numpy(), op_file)

        # when it;s correct
        op_file = os.path.join(output_dir, "mlp_confidence_matrix_1.joblib")
        joblib.dump(sorted_predicted_prob[predicted_labels == labels, :5].numpy(), op_file)

        # when it's wrong
        op_file = os.path.join(output_dir, "mlp_confidence_matrix_0.joblib")
        joblib.dump(sorted_predicted_prob[predicted_labels != labels, : 5].numpy(), op_file)

        # when it's top 5 ?
        top_5 = sort_idx[:, :5]
        combined = torch.sum(top_5 == labels.view(-1, 1), dim=1)
        op_file = os.path.join(output_dir, "mlp_confidence_matrix_5.joblib")
        joblib.dump(sorted_predicted_prob[combined == 1, : 5].numpy(), op_file)

        # where it's not even top 5 accurate ?
        op_file = os.path.join(output_dir, "mlp_confidence_matrix_n5.joblib")
        joblib.dump(sorted_predicted_prob[combined == 0, : 5], op_file)

    duration = timer() - begin
    print("Took : %.7f seconds" % duration)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL Confidence Evaluator")

    parser.add_argument('--state_dict_path', type=str, help='Path to State Dict')
    parser.add_argument('--data_path', type=str, help='Path to data file')
    parser.add_argument('--output_dir', type=str, help='Output directory')

    args = parser.parse_args()

    state_dict_path = args.state_dict_path
    data_path = args.data_path
    output_dir = args.output_dir

    confidence(state_dict_path, data_path, output_dir)
