import pandas as pd
import os
import joblib
import argparse
import pathlib
from thesislib.utils.ml import models, report
from sklearn.metrics import confusion_matrix


def confusion(model_path, model_type, data_file, output_dir, num_symptoms):
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

    predicted_labels = clf.predict(data)

    cnf_mat = confusion_matrix(label_values, predicted_labels)

    op_file = os.path.join(output_dir, "%s_confusion_matrix.joblib" % model_type)
    joblib.dump(cnf_mat, op_file)

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

    confusion(model_path, model_type, data_file, output_dir, num_symptoms)
