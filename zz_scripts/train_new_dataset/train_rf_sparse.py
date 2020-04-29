import scipy.sparse as sparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit
import json
import os
import joblib
import argparse
from timeit import default_timer as timer
from sklearn.preprocessing import LabelEncoder


def check_is_in(needles, haystack):
    if needles.shape[0] != haystack.shape[0]:
        raise ValueError("Needles and Haystack shape mismatch")

    result = np.zeros((needles.shape[0], ), dtype=bool)

    for idx in range(haystack.shape[0]):
        result[idx] = np.isin(needles[idx], haystack[idx, :]).reshape(1, )[0]

    return result


def top_n_score(y_target, y_pred, class_labels, top_n=10, weighted=True):
    """
    This method returns the top_n_score.
    The top_n_score returns 1 if the target_label is in the first n predictions in y_pred
    :param y_target: This is an array of target labels with shape (n_samples,)
    :param y_pred: This is an array of predicted probabilities with shape (n_samples, n_classes)
    :param class_labels: This is the list of all possible classes
    :param top_n: This determines how many predictions to consider
    :param weighted: Return the raw score or weighted by the number of samples
    :return:
    """

    if top_n >= len(class_labels):
        top_n -= 1

    labelbin = LabelEncoder()
    labelbin.fit(class_labels)

    encoded_labels = labelbin.transform(y_target)

    sorted_prob = np.argsort(-y_pred, axis=1)

    top_n_predictions = sorted_prob[:, :top_n]
    encoded_probability = np.take_along_axis(y_pred, encoded_labels[:, None], axis=1)
    encoded_probability = encoded_probability.reshape(encoded_probability.shape[0], )

    bool_top_n = check_is_in(encoded_labels, top_n_predictions)
    combined = np.logical_and(bool_top_n, (encoded_probability > 0))

    score = sum(combined) if not weighted else sum(combined)/combined.shape[0]
    return score


def train_rf(data_file, symptoms_db_json, conditions_db_json, output_dir):
    print("Starting Random Forest Classification")
    begin = timer()
    with open(symptoms_db_json) as fp:
        symptoms_db = json.load(fp)
        num_symptoms = len(symptoms_db)

    with open(conditions_db_json) as fp:
        conditions_db = json.load(fp)
        num_conditions = len(conditions_db)

    classes = list(range(num_conditions))

    print("Reading CSV")
    start = timer()
    df = pd.read_csv(data_file, index_col='Index')
    end = timer()
    print("Reading CSV: %.5f secs" % (end - start))

    num_rows = df.shape[0]
    print("DataFrame Shape: ", df.shape)

    print("Prepping Sparse Representation")
    start = timer()
    label_values = df.LABEL.values
    symptoms = df.SYMPTOMS
    df = df.drop(columns=['LABEL', 'SYMPTOMS'])

    dense_matrix = sparse.coo_matrix(df.values)
    symptoms = symptoms.apply(lambda v: [int(idx) for idx in v.split(",")])

    columns = []
    rows = []
    for idx, val in enumerate(symptoms):
        rows += [idx for item in val]
        columns += val

    print("N_Row: %d\tN_col: %d" % (len(rows), len(columns)))
    print("Max_Row: %d\tMax_col: %d" % (np.max(rows), np.max(columns)))

    data = np.ones(len(rows))
    symptoms_coo = sparse.coo_matrix((data, (rows, columns)), shape=(num_rows, num_symptoms))

    data_csc = sparse.hstack([dense_matrix, symptoms_coo])
    data_csc = data_csc.tocsc()
    end = timer()
    print("Prepping Sparse Representation: %.5f secs" % (end - start))

    print("Shuffling Data")
    start = timer()
    split_t = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_data = None
    train_labels = None
    test_data = None
    test_labels = None
    for train_index, test_index in split_t.split(data_csc, label_values):
        train_data = data_csc[train_index]
        train_labels = label_values[train_index]
        test_data = data_csc[test_index]
        test_labels = label_values[test_index]

    end = timer()
    print("Shuffling Data: %.5f secs" % (end - start))

    print("Training RF Classifier")
    start = timer()
    clf = RandomForestClassifier(n_estimators=5, criterion='gini', max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                 max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                 bootstrap=True, oob_score=False, n_jobs=2, random_state=None, verbose=0,
                                 warm_start=False, class_weight=None)

    clf = clf.fit(train_data, train_labels)
    end = timer()
    print("Training RF Classifier: %.5f secs" % (end - start))

    print("Calculating Accuracy")
    start = timer()

    accuracy_scorer = make_scorer(accuracy_score)
    top_2_scorer = make_scorer(top_n_score, needs_proba=True, class_labels=classes, top_n=2)
    top_5_scorer = make_scorer(top_n_score, needs_proba=True, class_labels=classes, top_n=5)


    train_score = accuracy_scorer(clf, train_data, train_labels)
    test_score = accuracy_scorer(clf, test_data, test_labels)
    top_2_train_score = top_2_scorer(clf, train_data, train_labels)
    top_2_test_score = top_2_scorer(clf, test_data, test_labels)
    top_5_train_score = top_5_scorer(clf, train_data, train_labels)
    top_5_test_score = top_5_scorer(clf, test_data, test_labels)

    end = timer()
    print("Calculating Accuracy: %.5f secs" % (end - start))

    train_results = {
        "name": "Random Forest Classifier",
        "test_score": test_score,
        "train_score": train_score,
        "top_2_train_score": top_2_train_score,
        "top_2_test_score": top_2_test_score,
        "top_5_train_score": top_5_train_score,
        "top_5_test_score": top_5_test_score,

    }

    train_results_file = os.path.join(output_dir, "rf_train_results_sparse.json")
    with open(train_results_file, "w") as fp:
        json.dump(train_results, fp)

    estimator_serialized = {
        "clf": clf,
        "name": "random forest classifier on sparse"
    }
    estimator_serialized_file = os.path.join(output_dir, "rf_serialized_sparse.joblib")
    joblib.dump(estimator_serialized, estimator_serialized_file)

    finish = timer()
    print("Completed Random Forest Classification: %.5f secs" % (finish - begin))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medvice RandomForest Trainer')
    parser.add_argument('--data', help='Path to train csv file')
    parser.add_argument('--symptoms_json', help='Path to symptoms db.json')
    parser.add_argument('--conditions_json', help='Path to conditions db.json')
    parser.add_argument('--output_dir', help='Directory where results and trained model should be saved to')

    args = parser.parse_args()
    data_file = args.data
    output_dir = args.output_dir
    symptoms_db_json = args.symptoms_json
    conditions_db_json = args.conditions_json

    if not os.path.isfile(data_file):
        raise ValueError("data file does not exist")

    if not os.path.isfile(symptoms_db_json):
        raise ValueError("Invalid symptoms db file passed")

    if not os.path.isfile(conditions_db_json):
        raise ValueError("Invalid conditions db file passed")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    train_rf(data_file=data_file, output_dir=output_dir, symptoms_db_json=symptoms_db_json, conditions_db_json=conditions_db_json)
