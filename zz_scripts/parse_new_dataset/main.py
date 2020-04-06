import argparse
import json
import os
import pandas as pd
import numpy as np
from collections import OrderedDict


def _race_txform(val):
    race_code = {'white': 0, 'black':1, 'asian':2, 'native':3, 'other':4}
    return race_code.get(val)


def _label_txform(val, labels):
    return labels.get(val)


def _symptom_transform(val, labels):
    if type(val) is not str:
        print(val)
    parts = val.split(";")
    res = sum([labels.get(item) for item in parts])
    return res


def handle_bit_wise(val, comp):
    if val & comp > 0:
        return 1
    else:
        return 0


def parse_data(filepath, condition_labels, symptom_map, output_path, use_header=True):

    symptom_columns = ['PATIENT', 'GENDER', 'RACE', 'ETHNICITY', 'AGE_BEGIN', 'AGE_END',
                       'PATHOLOGY', 'NUM_SYMPTOMS', 'SYMPTOMS']

    if use_header:
        symptoms_df = pd.read_csv(filepath, names=symptom_columns)
    else:
        symptoms_df = pd.read_csv(filepath)
    filename = filepath.split("/")[-1]

    # drop the guys that have no symptoms
    symptoms_df = symptoms_df.loc[symptoms_df.NUM_SYMPTOMS > 0]

    symptoms_df['LABEL'] = symptoms_df.PATHOLOGY.apply(_label_txform, labels=condition_labels)
    symptoms_df.RACE = symptoms_df.RACE.apply(_race_txform)
    symptoms_df.GENDER = symptoms_df.GENDER.apply(lambda gender: 0 if gender == 'F' else 1)
    symptoms_df = symptoms_df.rename(columns={'AGE_BEGIN': 'AGE'})
    symptoms_df['NSYMPTOMS'] = symptoms_df.SYMPTOMS.apply(_symptom_transform, labels=symptom_map)

    vector = sorted(symptom_map.keys())
    for idx, code in enumerate(vector):
        symptoms_df[code] = (symptoms_df.NSYMPTOMS & 2 ** idx).gt(0).astype(np.uint8)

    ordered_keys = ['LABEL', 'GENDER', 'RACE', 'AGE'] + vector
    output_file = os.path.join(output_path, "%s.csv" % filename)
    symptoms_df = symptoms_df[ordered_keys]
    symptoms_df.to_csv(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medvice Data Parser')

    parser.add_argument('--symptom_file', help='Symptom file to parse')

    parser.add_argument('--symptoms_json', help='Path to symptoms db.json')

    parser.add_argument('--conditions_json', help='Path to conditions_db.json')

    parser.add_argument('--output_dir', help='Directory where the parsed csv output should be written to')

    parser.add_argument('--use_headers', action='store_true', help='Does the file have a header already?')

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

    symptom_vector = sorted(list(symptom_db.keys()))
    condition_codes = sorted(list(condition_db.keys()))
    condition_labels = {code: idx for idx, code in enumerate(condition_codes)}
    symptom_index_map = OrderedDict({code: 2 ** idx for idx, code in enumerate(symptom_vector)})

    parse_data(symptom_file, condition_labels, symptom_index_map, output_dir, use_headers)
