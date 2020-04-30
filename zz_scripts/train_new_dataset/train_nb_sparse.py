import pandas as pd
from sklearn.metrics import accuracy_score, make_scorer, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import json
import os
import joblib
import argparse
from timeit import default_timer as timer
from thesislib.utils.ml import models, report
from sklearn import naive_bayes


def train_nb(data_file, symptoms_db_json, conditions_db_json, output_dir):
    print("Starting Naive Bayes Classification")
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
    data = pd.read_csv(data_file, index_col='Index')
    end = timer()
    print("Reading CSV: %.5f secs" % (end - start))

    print("Prepping Sparse Representation")
    start = timer()
    label_values = data.LABEL.values
    ordered_keys = ['GENDER', 'RACE', 'AGE', 'SYMPTOMS']
    data = data[ordered_keys]

    sparsifier = models.ThesisSymptomSparseMaker(num_symptoms=num_symptoms)
    data = sparsifier.fit_transform(data)

    end = timer()
    print("Prepping Sparse Representation: %.5f secs" % (end - start))

    print("Shuffling Data")
    start = timer()
    split_t = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_data = None
    train_labels = None
    test_data = None
    test_labels = None
    for train_index, test_index in split_t.split(data, label_values):
        train_data = data[train_index]
        train_labels = label_values[train_index]
        test_data = data[test_index]
        test_labels = label_values[test_index]

    end = timer()
    print("Shuffling Data: %.5f secs" % (end - start))

    print("Training Naive Bayes")
    start = timer()
    symptom_clf = naive_bayes.BernoulliNB()
    gender_clf = naive_bayes.BernoulliNB()
    race_clf = naive_bayes.MultinomialNB()
    age_clf = naive_bayes.GaussianNB()

    classifier_map = [
        [gender_clf, [0, False]],
        [race_clf, [1, False]],
        [age_clf, [2, False]],
        [symptom_clf, [(3, None), True]],
    ]

    clf = models.ThesisSparseNaiveBayes(classifier_map=classifier_map, classes=classes)

    clf.fit(train_data, train_labels)
    end = timer()
    print("Training Naive Classifier: %.5f secs" % (end - start))

    print("Calculating Accuracy")
    start = timer()

    accuracy_scorer = make_scorer(accuracy_score)
    f1_scorer_unweighted = make_scorer(f1_score, average='macro')
    f1_scorer_weighted = make_scorer(f1_score, average='weighted')
    recall_scorer_unweighted = make_scorer(recall_score, average='macro')
    recall_scorer_weighted = make_scorer(recall_score, average='weighted')
    precision_scorer_unweighted = make_scorer(precision_score, average='macro', zero_division=1)
    precision_scorer_weighted = make_scorer(precision_score, average='weighted', zero_division=1)
    roc_auc_scorer_unweighted = make_scorer(roc_auc_score, average='macro', multi_class='ovo')
    roc_auc_scorer_weighted = make_scorer(roc_auc_score, average='weighted', multi_class='ovr')

    top_2_scorer = make_scorer(report.top_n_score, needs_proba=True, class_labels=classes, top_n=2)
    top_5_scorer = make_scorer(report.top_n_score, needs_proba=True, class_labels=classes, top_n=5)

    train_score = accuracy_scorer(clf, train_data, train_labels)
    test_score = accuracy_scorer(clf, test_data, test_labels)
    top_2_train_score = top_2_scorer(clf, train_data, train_labels)
    top_2_test_score = top_2_scorer(clf, test_data, test_labels)
    top_5_train_score = top_5_scorer(clf, train_data, train_labels)
    top_5_test_score = top_5_scorer(clf, test_data, test_labels)

    f1_train_score_unweighted = f1_scorer_unweighted(clf, train_data, train_labels)
    recall_train_score_unweighted = recall_scorer_unweighted(clf, train_data, train_labels)
    precision_train_score_unweighted = precision_scorer_unweighted(clf, train_data, train_labels)
    roc_train_score_unweighted = roc_auc_scorer_unweighted(clf, train_data, train_labels)

    f1_train_score_weighted = f1_scorer_weighted(clf, train_data, train_labels)
    recall_train_score_weighted = recall_scorer_weighted(clf, train_data, train_labels)
    precision_train_score_weighted = precision_scorer_weighted(clf, train_data, train_labels)
    roc_train_score_weighted = roc_auc_scorer_weighted(clf, train_data, train_labels)

    f1_test_score_unweighted = f1_scorer_unweighted(clf, test_data, test_labels)
    recall_test_score_unweighted = recall_scorer_unweighted(clf, test_data, test_labels)
    precision_test_score_unweighted = precision_scorer_unweighted(clf, test_data, test_labels)
    roc_test_score_unweighted = roc_auc_scorer_unweighted(clf, test_data, test_labels)

    f1_test_score_weighted = f1_scorer_weighted(clf, test_data, test_labels)
    recall_test_score_weighted = recall_scorer_weighted(clf, test_data, test_labels)
    precision_test_score_weighted = precision_scorer_weighted(clf, test_data, test_labels)
    roc_test_score_weighted = roc_auc_scorer_weighted(clf, test_data, test_labels)

    end = timer()
    print("Calculating Accuracy: %.5f secs" % (end - start))

    train_results = {
        "name": "Naive Bayes Classifier",
        "accuracy": {
            "train": train_score,
            "test": test_score
        },
        "top_2": {
            "train": top_2_train_score,
            "test": top_2_test_score
        },
        "top_5": {
            "train": top_5_train_score,
            "test": top_5_test_score
        },
        "f1_unweighted": {
            "train": f1_train_score_unweighted,
            "test": f1_test_score_unweighted
        },
        "f1_weighted": {
            "train": f1_train_score_weighted,
            "test": f1_test_score_weighted
        },
        "recall_unweighted": {
            "train": recall_train_score_unweighted,
            "test": recall_test_score_unweighted
        },
        "recall_weighted": {
            "train": recall_train_score_weighted,
            "test": recall_test_score_weighted
        },
        "precision_weighted": {
            "train": precision_train_score_weighted,
            "test": precision_test_score_weighted
        },
        "precision_unweighted": {
            "train": precision_train_score_unweighted,
            "test": precision_test_score_unweighted
        },
        "roc_unweighted": {
            "train": roc_train_score_unweighted,
            "test": roc_test_score_unweighted
        },
        "roc_weighted": {
            "train": roc_train_score_weighted,
            "test": roc_test_score_weighted
        }
    }

    train_results_file = os.path.join(output_dir, "nb_train_results_sparse.json")
    with open(train_results_file, "w") as fp:
        json.dump(train_results, fp, indent=4)

    estimator_serialized = {
        "clf": clf.serialize(),
        "name": "naive bayes classifier on sparse"
    }
    estimator_serialized_file = os.path.join(output_dir, "nb_serialized_sparse.joblib")
    joblib.dump(estimator_serialized, estimator_serialized_file)

    finish = timer()
    print("Completed Naive Classification: %.5f secs" % (finish - begin))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medvice NaiveBayes Trainer')
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

    train_nb(data_file=data_file, output_dir=output_dir, symptoms_db_json=symptoms_db_json, conditions_db_json=conditions_db_json)
