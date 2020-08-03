import pandas as pd
import os
import joblib
import argparse
from timeit import default_timer as timer
from thesislib.utils.ml import models, report
import mlflow

BASE_DIR="/home/oagba/bulk/data/"
TARGET_DIRS=[
    "output_combined_15k",
    "output_basic_15k",
    "output_basic_2_cnt_15k",
    "output_basic_3_cnt_15k",
    "output_basic_4_cnt_15k",
    "output_basic_avg_cnt_15k",
    "output_basic_pct_10_15k",
    "output_basic_pct_20_15k",
    "output_basic_pct_30_15k",
    "output_basic_pct_50_15k",
    "output_basic_pct_70_15k",
    "output_basic_inc_1_15k",
    "output_basic_inc_2_15k",
    "output_basic_inc_3_15k"
]


def compare(directory, model_type, num_symptoms, mlflow_uri, only_self=True):

    if model_type == "random_forest":
        model_path = os.path.join(BASE_DIR, directory, "symptoms/csv/parsed/learning/rf/rf_1_5.joblib")
    else:
        model_path = os.path.join(BASE_DIR, directory, "symptoms/csv/parsed/learning/nb/nb_1_5.joblib")

    if not os.path.isfile(model_path):
        raise ValueError("Trained %s model does not exist!" % model_type)

    mlflow.set_tracking_uri(mlflow_uri)
    name = "eval_%s" % directory

    run_metrics = {}
    mlflow.set_experiment(name)
    with mlflow.start_run():
        begin = timer()

        mlflow.log_params({
            "source": directory,
            "model": model_type
        })

        try:
            start = timer()
            clf_data = joblib.load(model_path)
            if model_type == "random_forest":
                clf = clf_data.get("clf")
            else:
                clf_serialized = clf_data.get("clf")
                clf = models.ThesisSparseNaiveBayes.load(clf_serialized)

            end = timer()
            run_metrics["model_load_time"] = end - start

            start_eval = timer()
            if only_self:
                dirlist = [directory]
            else:
                dirlist = TARGET_DIRS

            for dirname in dirlist:
                start_0 = timer()
                data_file = os.path.join(BASE_DIR, dirname, "symptoms/csv/parsed/test.csv_sparse.csv")
                data = pd.read_csv(data_file, index_col='Index')
                classes = data.LABEL.unique().tolist()
                label_values = data.LABEL.values
                data = data.drop(columns=['LABEL'])
                sparsifier = models.ThesisSymptomRaceSparseMaker(num_symptoms=num_symptoms)
                data = sparsifier.fit_transform(data)

                run_metrics["%s_data_load_time" % dirname] = timer() - start_0

                metric_names = ['accuracy', 'top_5', 'precision_weighted']
                scorers = report.get_tracked_metrics(classes=classes, metric_name=metric_names)

                start_1 = timer()
                for key, scorer in scorers.items():
                    start = timer()
                    score = scorer(clf, data, label_values)
                    score_key = "%s_%s_score" % (dirname, key)
                    score_time_key = "%s_%s_time" % (dirname, key)
                    run_metrics[score_key] = score
                    run_metrics[score_time_key] = timer() - start

                run_metrics["%s_score_time" % dirname] = timer() - start_1
                run_metrics["%s_eval_time" % dirname] = timer() - start_0

            run_metrics["eval_time"] = timer() - start_eval
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
        return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Medvice Tester')
    parser.add_argument('--dir', help='Path to saved model', type=str)
    parser.add_argument('--model_type', help='Type of the model', type=str)
    parser.add_argument('--num_symptoms', help='The number of symptoms', type=int, default=376)
    parser.add_argument('--mlflow_uri', help='MLFlow URI', type=str)
    parser.add_argument('--only_self', help='Only test for current model', type=int, default=1)

    args = parser.parse_args()
    directory = args.dir
    model_type = args.model_type
    num_symptoms = args.num_symptoms
    mlflow_uri = args.mlflow_uri
    only_self = args.only_self > 0

    compare(directory, model_type, num_symptoms, mlflow_uri, only_self)
