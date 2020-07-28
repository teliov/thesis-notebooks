import pandas as pd
import numpy as np
import os
import joblib
import argparse
import pathlib
from thesislib.utils.ml import models


def confidence(model_path, model_type, data_file, output_dir, num_symptoms):
    clf_data = joblib.load(model_path)

    if model_type == "random_forest":
        clf = clf_data.get("clf")
    else:
        clf_serialized = clf_data.get("clf")
        clf = models.ThesisSparseNaiveBayes.load(clf_serialized)

    data = pd.read_csv(data_file, index_col='Index')
    label_values = data.LABEL.values
    data = data.drop(columns=['LABEL'])

    sparsifier = models.ThesisSymptomRaceSparseMaker(num_symptoms=num_symptoms)
    data = sparsifier.fit_transform(data)

    predicted_prob = clf.predict_proba(data)

    # sort idx, max first
    sort_idx = np.argsort(-predicted_prob, axis=1)
    sorted_predicted_prob = -np.sort(-predicted_prob, axis=1)

    predicted_labels = sort_idx[:, 0]

    # for all the data
    op_file = os.path.join(output_dir, "%s_confidence_matrix_all.joblib" % model_type)
    joblib.dump(sorted_predicted_prob[:, :5], op_file)

    # when it;s correct
    op_file = os.path.join(output_dir, "%s_confidence_matrix_1.joblib" % model_type)
    correct_idx = np.where(predicted_labels == label_values)[0]
    joblib.dump(sorted_predicted_prob[correct_idx, :5], op_file)

    # when it's wrong
    op_file = os.path.join(output_dir, "%s_confidence_matrix_0.joblib" % model_type)
    wrong_idx = np.where(predicted_labels != label_values)[0]
    joblib.dump(sorted_predicted_prob[wrong_idx, : 5], op_file)

    # when it's top 5 ?
    top_n_predictions = sort_idx[:, :5]
    encoded_probability = np.take_along_axis(predicted_prob, label_values[:, None], axis=1)
    encoded_probability = encoded_probability.reshape(encoded_probability.shape[0], )

    bool_top_n = label_values.reshape(-1, 1) == top_n_predictions
    bool_top_n = np.sum(bool_top_n, axis=1).astype(np.bool)
    combined = np.logical_and(bool_top_n, (encoded_probability > 0))
    top_5_idx = np.where(combined == True)[0]
    op_file = os.path.join(output_dir, "%s_confidence_matrix_5.joblib" % model_type)
    joblib.dump(sorted_predicted_prob[top_5_idx, : 5], op_file)

    # where it's not even top 5 accurate ?
    not_top_5_idx = np.where(combined == False)[0]
    op_file = os.path.join(output_dir, "%s_confidence_matrix_n5.joblib" % model_type)
    joblib.dump(sorted_predicted_prob[not_top_5_idx, : 5], op_file)

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

    confidence(model_path, model_type, data_file, output_dir, num_symptoms)
