import pandas as pd
import numpy as np
import os
import joblib
import argparse
import pathlib
from thesislib.utils.ml import models

EVAL_CONDITIONS = [182, 200, 229, 414, 441, 788, 93, 629, 56]


def top_5_qual(model_path, model_type, data_file, output_dir, num_symptoms):
    clf_data = joblib.load(model_path)

    if model_type == "random_forest":
        clf = clf_data.get("clf")
    else:
        clf_serialized = clf_data.get("clf")
        clf = models.ThesisSparseNaiveBayes.load(clf_serialized)

    data = pd.read_csv(data_file, index_col='Index')
    data = data[data['LABEL'].isin(EVAL_CONDITIONS)]
    label_values = data.LABEL.values
    data = data.drop(columns=['LABEL'])

    sparsifier = models.ThesisSymptomRaceSparseMaker(num_symptoms=num_symptoms)
    data = sparsifier.fit_transform(data)

    predicted_prob = clf.predict_proba(data)

    # sort idx, max first
    sort_idx = np.argsort(-predicted_prob, axis=1)
    top_5 = sort_idx[:, :5]

    combined = top_5 == label_values.reshape(-1, 1)
    combined = np.sum(combined, axis=1).astype(np.bool)

    _top_5_acc = top_5[combined, :]
    _top_5_labels = label_values[combined]
    op_file = os.path.join(output_dir, "%s_top_5_pred.joblib" % model_type)
    joblib.dump({
        'acc': _top_5_acc,
        'labels': _top_5_labels
    }, op_file)

    # when it's not top 5 accurate
    non_top_5 = combined == False
    _ntop_5_acc = top_5[non_top_5, :]
    _ntop_5_labels = label_values[non_top_5]
    op_file = os.path.join(output_dir, "%s_top_n5_pred.joblib" % model_type)
    joblib.dump({
        'acc': _ntop_5_acc,
        'labels': _ntop_5_labels
    }, op_file)

    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Medvice Confusion Matrix Resolver')
    parser.add_argument('--data', help='Path to train csv file', type=str)
    parser.add_argument('--model_path', help='Path to saved model', type=str)
    parser.add_argument('--output_dir', help='Directory where confusion matrix should be saved to', type=str)
    parser.add_argument('--model_type', help='Type of the model', type=str)
    parser.add_argument('--num_symptoms', help='The number of symptoms', type=int, default=376)

    args = parser.parse_args()
    data_file = args.data
    model_path = args.model_path
    model_type = args.model_type
    output_dir = args.output_dir
    num_symptoms = args.num_symptoms

    if not os.path.isfile(data_file):
        raise ValueError("Data file does not exist")
    if not os.path.isfile(model_path):
        raise ValueError("Model file does not exist")

    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)

    top_5_qual(model_path, model_type, data_file, output_dir, num_symptoms)
