import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import os
import json
import joblib
import argparse
from timeit import default_timer as timer
from sklearn import naive_bayes
import mlflow
import pathlib
import time
from thesislib.utils.ml import models, report


def train_nb(
        data_file,
        num_symptoms,
        output_dir,
        train_size,
        fold_number,
        name=None,
        mlflow_uri=None,
        save_model=False):

    # create the directory
    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)

    if name is None:
        name = "nb_%d_%02d_%02d" % (int(time.time()), train_size, fold_number)

    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)

    run_metrics = {}

    mlflow.set_experiment(name)
    with mlflow.start_run():
        begin = timer()
        try:
            mlflow.log_params({
                "train_size": train_size,
                "fold_num": fold_number,
                "model": "naive_bayes"
            })

            # read the csv file
            start = timer()
            data = pd.read_csv(data_file, index_col='Index')
            end = timer()
            run_metrics["csv_read_time"] = end - start

            classes = data.LABEL.unique().tolist()

            # Prepping Sparse Representation
            start = timer()
            label_values = data.LABEL.values
            data = data.drop(columns=['LABEL'])

            sparsifier = models.ThesisSymptomRaceSparseMaker(num_symptoms=num_symptoms)
            data = sparsifier.fit_transform(data)

            end = timer()
            run_metrics['sparsify_time'] = end-start

            # need to select just the portion of the data that is required for this run
            data_split_time = 0
            if train_size != 1:
                start = timer()
                split_selector = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=int(train_size * 10))

                for split_index, _ in split_selector.split(data, label_values):
                    data = data[split_index]
                    label_values = label_values[split_index]

                end = timer()
                data_split_time = end - start

            run_metrics["data_split_time"] = data_split_time
            run_metrics['train_sample_size'] = data.shape[0]

            # now we do the kfold bit!
            # we're doing 5 folds by default, so train = 4, test=1 => 0.8 train_size, 0.2 test_size
            start = timer()
            split_t = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=fold_number)
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
            run_metrics['fold_timer'] = end - start

            # now we start training
            # age is gaussian, the rest is bernoulli!
            start = timer()
            bernoulli_clf = naive_bayes.BernoulliNB()
            age_clf = naive_bayes.GaussianNB()

            classifier_map = [
                [age_clf, [0, False]],
                [bernoulli_clf, [(1, None), True]]
            ]

            clf = models.ThesisSparseNaiveBayes(classifier_map=classifier_map, classes=classes)
            clf.fit(train_data, train_labels)
            end = timer()

            run_metrics['train_time'] = end - start

            # need to estimate metrics
            # interested in recall, precision, top5 accuracy and accuracy
            start = timer()
            metric_names = ['accuracy', 'top_5', 'precision_weighted', 'recall_weighted']
            scorers = report.get_tracked_metrics(classes=classes, metric_name=metric_names)

            score_params = {}

            for key, scorer in scorers.items():
                scorer_timer_train = timer()
                train_score = scorer(clf, train_data, train_labels)
                scorer_timer_test = timer()
                test_score = scorer(clf, test_data, test_labels)
                scorer_timer_end = timer()

                train_score_key = "%s_train_score" % key
                test_score_key = "%s_test_score" % key
                train_time_key = "%s_train_score_time" % key
                test_time_key = "%s_test_score_time" % key

                train_score_time = scorer_timer_test - scorer_timer_train
                test_score_time = scorer_timer_end - scorer_timer_test

                score_params[train_score_key] = train_score
                score_params[test_score_key] = test_score
                score_params[train_time_key] = train_score_time
                score_params[test_time_key] = test_score_time

            end = timer()
            score_params['scoring_time'] = end - start

            run_metrics.update(score_params)

            if save_model:
                estimator_serialized = {
                    "clf": clf.serialize(),
                    "name": "%s_classifier" % name
                }
                estimator_serialized_file = os.path.join(output_dir, "nb_%d_%d.joblib" % (train_size, fold_number))
                joblib.dump(estimator_serialized, estimator_serialized_file)
            res = 1
            message = "success"
        except Exception as e:
            message = e.__str__()
            res = 0

        finish = timer()
        run_metrics['run_time'] = finish - begin
        run_metrics['completed'] = res
        mlflow.log_metrics(run_metrics)
        mlflow.log_param('message', message)
    return res == 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Medvice NaiveBayes Trainer')
    parser.add_argument('--data', help='Path to train csv file', type=str)
    parser.add_argument('--num_symptoms', help='Number of symptoms in DB', type=int)
    parser.add_argument('--output_dir', help='Directory trained model should be saved to', type=str)
    parser.add_argument('--train_size', help='Size of training data to use. Useful for making a learning curve.'
                                             ' A float between 0 to 1', type=float, default=1)
    parser.add_argument('--fold_number', help='Assumes 5 Stratifed K-fold. '
                                              'Uses this as random state when splitting', type=int, default=1)
    parser.add_argument('--name', type=str, help="Name for this experiment. "
                                                 "When tracking multiple runs, good to make this unique")
    parser.add_argument('--mlflow_uri', type=str, help="URI to Mlflow tracking server")
    parser.add_argument('--save_model', type=int, help="Should the models be saved? Defaults to false", default=0)

    args = parser.parse_args()
    data_file = args.data
    num_symptoms = args.num_symptoms
    output_dir = args.output_dir
    train_size = args.train_size
    fold_number = args.fold_number
    name = args.name
    mlflow_uri = args.mlflow_uri
    save_model = args.save_model > 0

    if not os.path.isfile(data_file):
        raise ValueError("Data file does not exist")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    train_nb(
        data_file,
        num_symptoms,
        output_dir,
        train_size,
        fold_number,
        name,
        mlflow_uri,
        save_model)
