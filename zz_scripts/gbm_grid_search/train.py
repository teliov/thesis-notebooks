import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import json
import os
import argparse
from timeit import default_timer as timer
import pickle
import sys
from thesislib.utils.ml import models, report

class RFParams(object):
    n_estimators = 100
    criterion = 'gini'
    max_depth = None
    min_samples_split = 2
    min_samples_leaf = 1
    max_leaf_nodes = None
    max_features = 'auto'# plo

    def get_params(self):
        return {
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_leaf_nodes": self.max_leaf_nodes,
            "max_features": self.max_features
        }

    def __str__(self):
        params = self.get_params()
        name = ""
        for key, val in params.items():
            name += "%s_%s_" % (key, str(val))
        name = name.strip("_")
        return name


def train_rf(data_file, symptoms_db_json, output_dir, rfparams):
    try:
        begin = timer()
        with open(symptoms_db_json) as fp:
            symptoms_db = json.load(fp)
            num_symptoms = len(symptoms_db)

        df = pd.read_csv(data_file, index_col='Index')

        classes = df.LABEL.unique().tolist()

        label_values = df.LABEL.values
        ordered_keys = ['GENDER', 'RACE', 'AGE', 'SYMPTOMS']
        df = df[ordered_keys]

        sparsifier = models.ThesisSymptomSparseMaker(num_symptoms=num_symptoms)
        data_csc = sparsifier.fit_transform(df)

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

        clf = RandomForestClassifier(
            n_estimators=rfparams.n_estimators,
            criterion=rfparams.criterion,
            max_depth=rfparams.max_depth,
            min_samples_split=rfparams.min_samples_split,
            min_samples_leaf=rfparams.min_samples_leaf,
            max_features=rfparams.max_features,
            max_leaf_nodes=rfparams.max_leaf_nodes,
            n_jobs=2
        )

        clf = clf.fit(train_data, train_labels)

        train_time = timer()

        scorers = report.get_tracked_metrics(classes=classes, metric_name=[
            report.ACCURACY_SCORE,
            report.TOP2_SCORE,
            report.TOP5_SCORE
        ])

        train_results = {
            "name": "RF %s" % rfparams.__str__(),
        }

        for key, scorer in scorers.items():
            train_score = scorer(clf, train_data, train_labels)
            test_score = scorer(clf, test_data, test_labels)
            train_results[key] = {
                "train": train_score,
                "test": test_score
            }

        score_time = timer()

        clf = pickle.dumps(clf)
        model_size =  sys.getsizeof(clf)
        train_results["stats"] = {
            "score_time": score_time - train_time,
            "train_time": train_time - begin,
            "model_size": model_size
        }

        train_results_file = os.path.join(output_dir, "rf_gsearch_%s.json" % rfparams.__str__())
        with open(train_results_file, "w") as fp:
            json.dump(train_results, fp)

        res = True
    except Exception as e:
        print(e.__str__())
        res = False

    return res


if __name__ == "__main__":
    """
    "n_estimators": self.n_estimators,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_leaf_nodes": self.max_leaf_nodes,
            "max_features": self.max_features
    """
    parser = argparse.ArgumentParser(description='Medvice RandomForest Trainer')
    parser.add_argument('--data', help='Path to train csv file')
    parser.add_argument('--symptoms_json', help='Path to symptoms db.json')
    parser.add_argument('--max_depth')
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    parser.add_argument('--max_leaf_nodes')
    parser.add_argument('--max_features', default="auto")
    parser.add_argument('--estimators', default=100, type=int)
    parser.add_argument('--output_dir', help='Directory where results and trained model should be saved to')

    args = parser.parse_args()
    data_file = args.data
    output_dir = args.output_dir
    symptoms_db_json = args.symptoms_json

    if not os.path.isfile(data_file):
        raise ValueError("data file does not exist")

    if not os.path.isfile(symptoms_db_json):
        raise ValueError("Invalid symptoms db file passed")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    rfparams = RFParams()
    rfparams.max_depth = int(args.max_depth) if args.max_depth else args.max_depth
    rfparams.min_samples_split = args.min_samples_split
    rfparams.min_samples_leaf = args.min_samples_leaf
    rfparams.max_leaf_nodes = int(args.max_leaf_nodes) if args.max_leaf_nodes else args.max_leaf_nodes
    rfparams.max_features = args.max_features
    rfparams.n_estimators = args.estimators

    print("Running: %s" % rfparams.__str__())

    train_rf(data_file=data_file, output_dir=output_dir, symptoms_db_json=symptoms_db_json, rfparams=rfparams)
