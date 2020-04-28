import numpy as np
import pandas as pd
import argparse
import json
import os


RACE_CODE = {'white': 0, 'black':1, 'asian':2, 'native':3, 'other':4}


def _symptom_transform(val, labels):
    """
    Val is a string in the form: "symptom_0;symptom_1;...;symptom_n"
    :param val:
    :param labels:
    :return:
    """
    parts = val.split(";")
    res = ",".join([labels.get(item) for item in parts])
    return res


def parse_data(filepath, conditions_db_json, symptoms_db_json, output_path):

    with open(symptoms_db_json) as fp:
        symptoms_db = json.load(fp)

    with open(conditions_db_json) as fp:
        conditions_db = json.load(fp)

    condition_labels = {code: idx for idx, code in enumerate(sorted(conditions_db.keys()))}
    symptom_map = {code: str(idx) for idx, code in enumerate(sorted(symptoms_db.keys()))}

    usecols = ['GENDER', 'RACE', 'AGE_BEGIN', 'PATHOLOGY', 'NUM_SYMPTOMS', 'SYMPTOMS']

    symptoms_df = pd.read_csv(filepath, usecols=usecols)

    filename = filepath.split("/")[-1]

    # drop the guys that have no symptoms
    df = symptoms_df[symptoms_df.NUM_SYMPTOMS > 0]
    df['LABEL'] = df.PATHOLOGY.apply(lambda v: condition_labels.get(v))
    df['RACE'] = df.RACE.apply(lambda v: RACE_CODE.get(v))
    df['GENDER'] = df.GENDER.apply(lambda gender: 0 if gender == 'F' else 1)
    df = df.rename(columns={'AGE_BEGIN': 'AGE'})
    df['SYMPTOMS'] = df.SYMPTOMS.apply(_symptom_transform, labels=symptom_map)
    ordered_keys = ['LABEL', 'GENDER', 'RACE', 'AGE', 'SYMPTOMS']
    df = df[ordered_keys]
    df.index.name = "Index"

    output_file = os.path.join(output_path, "%s_sparse.csv" % filename)
    df.to_csv(output_file)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medvice Sparse Data Parser')

    parser.add_argument('--symptom_file', help='Symptom file to parse')

    parser.add_argument('--symptoms_json', help='Path to symptoms db.json')

    parser.add_argument('--conditions_json', help='Path to conditions_db.json')

    parser.add_argument('--output_dir', help='Directory where the parsed csv output should be written to')

    args = parser.parse_args()

    symptom_file = args.symptom_file
    symptoms_db_json = args.symptoms_json
    conditions_db_json = args.conditions_json
    output_dir = args.output_dir

    use_headers = args.use_headers

    if not os.path.isfile(symptoms_db_json) or not os.path.isfile(conditions_db_json):
        raise ValueError("Invalid symptoms and/or conditions db file passed")

    if not os.path.isfile(symptom_file):
        raise ValueError("Symptom file does not exist")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with open(symptoms_db_json) as fp:
        symptom_db = json.load(fp)
    with open(conditions_db_json) as fp:
        condition_db = json.load(fp)

    parse_data(symptom_file, conditions_db_json, symptoms_db_json, output_dir)
