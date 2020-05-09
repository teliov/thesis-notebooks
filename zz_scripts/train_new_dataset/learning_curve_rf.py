import pandas as pd
import json
import os
import joblib
import argparse
from timeit import default_timer as timer
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import sys
import logging
import numpy as np

notebook_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "../.."))
if not (notebook_path in sys.path):
    sys.path.insert(0, notebook_path)

from thesislib.utils.ml import models, report


class RFParams(object):
    n_estimators = 20
    criterion = 'gini'
    max_depth = 380
    min_samples_split = 2
    min_samples_leaf = 2
    max_leaf_nodes = None
    min_impurity_decrease = 0.0
    max_features = 'log2'


def learning_curve_rf(data_file, symptoms_db_json, output_dir, num_splits=5, scorer_name=report.ACCURACY_SCORE):
    logger = report.Logger("Random Forest Learning Curve on QCE")

    try:
        message = "Starting Random Forest Learning Curve Evaluation with : %s and %d splits" % (scorer_name, num_splits)
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

        logger.log("Evaluating Random Forest Learning Curve with : %s and %d splits" % (scorer_name, num_splits))
        start = timer()

        rfparams = RFParams()
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
            n_jobs=1,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None
        )

        scorer = report.get_tracked_metrics(metric_name=scorer_name, classes=classes)

        num_jobs = 1

        train_sizes = np.linspace(0.1, 1, num_splits)
        train_sizes_abs, train_scores, test_scores, fit_times, score_times = learning_curve(clf, data, label_values,
                       train_sizes=train_sizes, cv=3, pre_dispatch="n_jobs",
                       scoring=scorer, n_jobs=num_jobs, verbose=0,
                       shuffle=False, return_times=True)
        end = timer()
        logger.log("Evaluating Random Forest Learning Curve with : %s and %d splits: %.5f secs" % (scorer_name, num_splits, (end - start)))

        results = {
            "train_sizes_abs": train_sizes_abs,
            "train_scores": train_scores,
            "test_scores": test_scores,
            "fit_times": fit_times,
            "score_times": score_times
        }

        summary = {
            "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
            "train_scores_std": np.std(train_scores, axis=1).tolist(),
            "test_scores_mean": np.mean(test_scores, axis=1).tolist(),
            "test_scores_std": np.std(test_scores, axis=1).tolist(),
            "fit_times_mean": np.mean(fit_times, axis=1).tolist(),
            "fit_times_std": np.std(fit_times, axis=1).tolist(),
            "score_times_mean": np.mean(score_times, axis=1).tolist(),
            "score_times_std": np.std(score_times, axis=1).tolist(),
        }

        summary_file =  os.path.join(output_dir, "rf_learning_curve_%s_%d_sparse.json" % (scorer_name, num_splits))
        with open(summary_file, "w") as fp:
            json.dump(summary, fp, indent=4)
        serialized_file = os.path.join(output_dir, "rf_learning_curve_%s_%d_sparse.joblib" % (scorer_name, num_splits))
        joblib.dump(results, serialized_file)

        finish = timer()
        message = "Completed  Random Forest Learning Curve Evaluation with : %s and %d splits %.5f secs" % (scorer_name, num_splits, (finish - begin))
        logger.log(message)
        res = True
    except Exception as e:
        message = e.__str__()
        logger.log(message, logging.ERROR)
        res = False

    print(logger.to_string())
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medvice Random Forest Trainer')
    parser.add_argument('--data', help='Path to train csv file')
    parser.add_argument('--symptoms_json', help='Path to symptoms db.json')
    parser.add_argument('--scorer_name', type=str, default=report.ACCURACY_SCORE, help='Scoring function during evaluation')
    parser.add_argument('--num_splits', help='How many splits to evaluate the data on', type=int, default=20)
    parser.add_argument('--output_dir', help='Directory where results and trained model should be saved to')

    args = parser.parse_args()
    data_file = args.data
    output_dir = args.output_dir
    symptoms_db_json = args.symptoms_json
    scorer_name = args.scorer_name
    num_splits = args.num_splits

    if not os.path.isfile(data_file):
        raise ValueError("data file does not exist")

    if not os.path.isfile(symptoms_db_json):
        raise ValueError("Invalid symptoms db file passed")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    learning_curve_rf(data_file=data_file, output_dir=output_dir, symptoms_db_json=symptoms_db_json, num_splits=num_splits,
                      scorer_name=scorer_name)
