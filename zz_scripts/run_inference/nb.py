import os
import joblib
import json
import argparse
from thesislib.utils.ml import models, report
import logging
from timeit import default_timer as timer
import pandas as pd
import pathlib


def nb_inference(model_file, test_files, symptoms_db_json, output_dir, name=""):
    logger = report.Logger("NB Forest %s Inference on QCE" % name)
    unserialized = joblib.load(model_file)
    clf_data = unserialized["clf"]
    clf = models.ThesisSparseNaiveBayes.load(clf_data)


    with open(test_files) as fp:
        test_data = json.load(fp)

    with open(symptoms_db_json) as fp:
        symptoms_db = json.load(fp)
        num_symptoms = len(symptoms_db)

    try:
        logger.log("Starting NB Forest Inference")
        for test_name, test_csv in test_data.items():
            logger.log("Reading %s CSV" % test_name)
            start = timer()
            df = pd.read_csv(test_csv, index_col='Index')
            end = timer()
            logger.log("Reading %s CSV: %.5f secs" % (test_name, end - start))

            classes = df.LABEL.unique().tolist()
            logger.log("Prepping Sparse Representation %s" % test_name)
            start = timer()
            label_values = df.LABEL.values
            ordered_keys = ['GENDER', 'RACE', 'AGE', 'SYMPTOMS']
            df = df[ordered_keys]

            sparsifier = models.ThesisSymptomSparseMaker(num_symptoms=num_symptoms)
            data_csc = sparsifier.fit_transform(df)

            end = timer()
            logger.log("Prepping Sparse Representation %s: %.5f secs" % (test_name, end - start))

            logger.log("Running NB %s Inference" % test_name)
            scorers = report.get_tracked_metrics(classes=classes, metric_name=[
                report.ACCURACY_SCORE,
                report.PRECISION_WEIGHTED,
                report.RECALL_WEIGHTED,
                report.TOP5_SCORE
            ])

            test_results = {
                "name": "Naive Bayes Classifier"
            }
            for score_name, scorer in scorers.items():
                logger.log("Starting %s Score: %s" % (test_name, score_name))
                scorer_timer_test = timer()
                test_score = scorer(clf, data_csc, label_values)
                test_results[score_name] = {
                    "test": test_score
                }
                scorer_timer_end = timer()
                test_duration = scorer_timer_end - scorer_timer_test
                logger.log("Finished %s score: %s.\nTook: %.5f seconds"
                           % (test_name, score_name, test_duration))

            test_result_file = os.path.join(output_dir, "nb_%s_inference.json" % test_name)
            with open(test_result_file ,"w") as fp:
                json.dump(test_results, fp, indent=4)

            end = timer()
            logger.log("Completed %s NB Inference: %.5f secs" % (test_name, end - start))
        res= True
    except Exception as e:
        message = e.__str__()
        logger.log(message, logging.ERROR)
        res = False

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medvice NB Inference Engine')
    parser.add_argument('--model', help='Path model')
    parser.add_argument('--output_dir', help='Directory where results should be saved to')
    parser.add_argument('--test_files', help="Json file that contains on each line location of test csv files")
    parser.add_argument('--symptoms_json', help='Path to symptoms db.json')

    args = parser.parse_args()
    model_file = args.model
    output_dir = args.output_dir
    test_files = args.test_files
    symptoms_db_json = args.symptoms_json

    if not os.path.isfile(model_file):
        raise ValueError("Model file does not exist")

    if not os.path.isfile(symptoms_db_json):
        raise ValueError("Symptoms db file does not exist")

    if not os.path.isfile(test_files):
        raise ValueError("Test file does not exist")

    if not os.path.isfile(symptoms_db_json):
        raise ValueError("Invalid symptoms db file passed")

    if not os.path.isdir(output_dir):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    nb_inference(model_file=model_file, test_files=test_files, symptoms_db_json=symptoms_db_json, output_dir=output_dir)
