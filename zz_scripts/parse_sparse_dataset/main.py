import pandas as pd
import argparse
import json
import os
from thesislib.utils.ml import process

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

    if not os.path.isfile(symptoms_db_json) or not os.path.isfile(conditions_db_json):
        raise ValueError("Invalid symptoms and/or conditions db file passed")

    if not os.path.isfile(symptom_file):
        raise ValueError("Symptom file does not exist")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    process.parse_data(symptom_file, conditions_db_json, symptoms_db_json, output_dir)
