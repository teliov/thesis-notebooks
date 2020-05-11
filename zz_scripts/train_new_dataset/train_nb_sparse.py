import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import json
import os
import joblib
import argparse
from timeit import default_timer as timer
from sklearn import naive_bayes
import sys
import logging


def train_nb(data_file, symptoms_db_json, output_dir, name=""):
    logger = report.Logger("Naive Bayes %s Classification on QCE" %name)

    try:
        message = "Starting Naive Bayes Classification"
        logger.log(message)
        begin = timer()
        with open(symptoms_db_json) as fp:
            symptoms_db = json.load(fp)
            num_symptoms = len(symptoms_db)

        logger.log("Reading CSV")
        start = timer()
        data = pd.read_csv(data_file, index_col='Index')
        end = timer()
        logger.log("Reading CSV: %.5f secs" % (end - start))

        classes = data.LABEL.unique().tolist()

        logger.log("Prepping Sparse Representation")
        start = timer()
        label_values = data.LABEL.values
        ordered_keys = ['GENDER', 'RACE', 'AGE', 'SYMPTOMS']
        data = data[ordered_keys]

        sparsifier = models.ThesisSymptomSparseMaker(num_symptoms=num_symptoms)
        data = sparsifier.fit_transform(data)

        end = timer()
        logger.log("Prepping Sparse Representation: %.5f secs" % (end - start))

        logger.log("Shuffling Data")
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
        logger.log("Shuffling Data: %.5f secs" % (end - start))

        logger.log("Training Naive Bayes")
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
        logger.log("Training Naive Classifier: %.5f secs" % (end - start))

        logger.log("Calculating Accuracy")
        start = timer()

        scorers = report.get_tracked_metrics(classes=classes)
        train_results = {
            "name": "Naive Bayes Classifier",
        }

        for key, scorer in scorers.items():
            logger.log("Starting Score: %s" % key)
            scorer_timer_train = timer()
            train_score = scorer(clf, train_data, train_labels)
            scorer_timer_test = timer()
            test_score = scorer(clf, test_data, test_labels)
            train_results[key] = {
                "train": train_score,
                "test": test_score
            }
            scorer_timer_end = timer()
            train_duration = scorer_timer_test - scorer_timer_train
            test_duration = scorer_timer_end - scorer_timer_test
            duration = scorer_timer_end - scorer_timer_train
            logger.log("Finished score: %s.\nTook: %.5f seconds\nTrain: %.5f, %.5f secs\n Test: %.5f, %.5f secs"
                       % (key, duration, train_score, train_duration, test_score, test_duration))

        end = timer()
        logger.log("Calculating Accuracy: %.5f secs" % (end - start))

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
        logger.log("Completed Naive Classification: %.5f secs" % (finish - begin))
        res = True
    except Exception as e:
        message = e.__str__()
        logger.log(message, logging.ERROR)
        res = False

    print(logger.to_string())
    return res


if __name__ == "__main__":
    notebook_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "../.."))

    if not (notebook_path in sys.path):
        sys.path.insert(0, notebook_path)

    from thesislib.utils.ml import models, report

    parser = argparse.ArgumentParser(description='Medvice NaiveBayes Trainer')
    parser.add_argument('--data', help='Path to train csv file')
    parser.add_argument('--symptoms_json', help='Path to symptoms db.json')
    parser.add_argument('--output_dir', help='Directory where results and trained model should be saved to')
    parser.add_argument('--name', type=str, default="")

    args = parser.parse_args()
    data_file = args.data
    output_dir = args.output_dir
    symptoms_db_json = args.symptoms_json
    name = args.name

    if not os.path.isfile(data_file):
        raise ValueError("data file does not exist")

    if not os.path.isfile(symptoms_db_json):
        raise ValueError("Invalid symptoms db file passed")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    train_nb(data_file=data_file, output_dir=output_dir, symptoms_db_json=symptoms_db_json, name=name)
