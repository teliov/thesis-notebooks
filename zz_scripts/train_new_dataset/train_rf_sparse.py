import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import json
import os
import joblib
import argparse
from timeit import default_timer as timer
from thesislib.utils.ml import models, report
import logging


class RFParams(object):
    n_estimators = 20
    criterion = 'gini'
    max_depth = 380
    min_samples_split = 2
    min_samples_leaf = 2
    max_leaf_nodes = None
    min_impurity_decrease = 0.0
    max_features = 'log2'


def train_rf(data_file, symptoms_db_json, output_dir, rfparams, name=""):
    logger = report.Logger("Random Forest %s Classification on QCE" % name)

    try:
        logger.log("Starting Random Forest Classification")
        begin = timer()
        with open(symptoms_db_json) as fp:
            symptoms_db = json.load(fp)
            num_symptoms = len(symptoms_db)

        logger.log("Reading CSV")
        start = timer()
        df = pd.read_csv(data_file, index_col='Index')
        end = timer()
        logger.log("Reading CSV: %.5f secs" % (end - start))

        classes = df.LABEL.unique().tolist()

        logger.log("Prepping Sparse Representation")
        start = timer()
        label_values = df.LABEL.values
        ordered_keys = ['GENDER', 'RACE', 'AGE', 'SYMPTOMS']
        df = df[ordered_keys]

        sparsifier = models.ThesisSymptomSparseMaker(num_symptoms=num_symptoms)
        data_csc = sparsifier.fit_transform(df)

        end = timer()
        logger.log("Prepping Sparse Representation: %.5f secs" % (end - start))

        logger.log("Shuffling Data")
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
        logger.log("Shuffling Data: %.5f secs" % (end - start))

        logger.log("Training RF Classifier")
        start = timer()
        clf = RandomForestClassifier(
            n_estimators=rfparams.n_estimators,
            criterion=rfparams.criterion,
            max_depth=rfparams.max_depth,
            min_samples_split=rfparams.min_samples_split,
            min_samples_leaf=rfparams.min_samples_leaf,
            min_weight_fraction_leaf=0.0,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            bootstrap=True,
            oob_score=False,
            n_jobs=2,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None
        )

        clf = clf.fit(train_data, train_labels)
        end = timer()
        logger.log("Training RF Classifier: %.5f secs" % (end - start))

        print("Calculating Accuracy")
        start = timer()

        scorers = report.get_tracked_metrics(classes=classes, metric_name=[
            report.ACCURACY_SCORE,
            report.PRECISION_WEIGHTED,
            report.RECALL_WEIGHTED,
            report.TOP5_SCORE
        ])

        train_results = {
            "name": "Random Forest",
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

        train_results_file = os.path.join(output_dir, "rf_train_results_sparse_grid_search_best.json")
        with open(train_results_file, "w") as fp:
            json.dump(train_results, fp)

        estimator_serialized = {
            "clf": clf,
            "name": "random forest classifier on sparse"
        }
        estimator_serialized_file = os.path.join(output_dir, "rf_serialized_sparse_grid_search_best.joblib")
        joblib.dump(estimator_serialized, estimator_serialized_file)

        finish = timer()
        logger.log("Completed Random Forest Classification: %.5f secs" % (finish - begin))
        res = True
    except Exception as e:
        message = e.__str__()
        logger.log(message, logging.ERROR)
        res = False

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medvice RandomForest Trainer')
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

    rfparams = RFParams()
    train_rf(data_file=data_file, output_dir=output_dir, symptoms_db_json=symptoms_db_json, rfparams=rfparams, name=name)
