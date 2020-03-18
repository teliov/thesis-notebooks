import argparse
from dateutil.parser import parse as date_parser
from joblib import Parallel, delayed
import json
import math
import os
import pandas as pd


def parse_symptoms(patient_conditions, condition_labels, symptom_vector, output_path, file_data):
    file_index, symptom_file, pass_columns = file_data

    race_code = {'white': 0, 'black': 1, 'asian': 2, 'native': 3, 'other': 4}

    columns = ["CONDITION_ID", "PATIENT", "SYMPTOM_CODE", "SYMPTOM_DISPLAY", "VALUE_CODE", "VALUE_DISPLAY"]
    usecols = ['CONDITION_ID', 'PATIENT', 'SYMPTOM_CODE']

    if pass_columns:
        symptoms = pd.read_csv(symptom_file, names=columns, usecols=usecols)
    else:
        symptoms = pd.read_csv(symptom_file, usecols=usecols)

    if symptoms.shape[1] <= 0:
        return None

    _tmp = symptoms.merge(patient_conditions, how='left', left_on='CONDITION_ID', right_on='Id', suffixes=('_symp', ''))

    # free memory ?
    del symptoms

    grp = _tmp.groupby(['CONDITION_ID'])
    design_matrix = {
        "label": [],
        "age": [],
        "gender": [],
        "race": [],
    }

    for item in symptom_vector:
        design_matrix[item] = []

    for item, df in grp.__iter__():
        vector = {_: 0 for _ in symptom_vector}

        onset_date = date_parser(df['ONSET'].iloc[0])
        patient_birthdate = date_parser(df["BIRTHDATE"].iloc[0])
        vector['age'] = abs(patient_birthdate.year - onset_date.year)
        vector['gender'] = 0 if df['GENDER'].iloc[0] == 'F' else 1
        vector['race'] = race_code[df['RACE'].iloc[0]]
        vector['label'] = condition_labels[df['CODE'].iloc[0]]

        for idx, symptom_code in df["SYMPTOM_CODE"].items():
            vector[symptom_code] = 1

        for k, v in vector.items():
            design_matrix[k].append(v)

    output_file = os.path.join(output_path, "processed_%d.json" % file_index)
    with open(output_file, 'w') as fp:
        json.dump(design_matrix, fp)


def parse_files(data_dir, symptoms_dir, output_dir, num_cpus = None):
    # read in the required meta files
    with open(os.path.join(data_dir, "condition_codes.json")) as fp:
        condition_codes = json.load(fp)
    with open(os.path.join(data_dir, "symptom_vector.json")) as fp:
        symptom_vector = json.load(fp)

    # check that the output dir is good ?
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    condition_label = {item: idx for idx, item in enumerate(condition_codes)}

    # parse the symptoms
    _temp_list = []

    for path in os.listdir(symptoms_dir):
        if os.path.isdir(os.path.join(symptoms_dir, path)):
            continue

        _temp_list.append(path)

    # we sort, because the first file in the list (alphabetically in ascending order) is the only one that has the
    # first row as the header row
    # it can be very time consuming to append a header at the beginning of a file so we need to let pandas handle
    # this for all n>1 files (where 1 corresponds to the first file)
    _temp_list.sort()

    symptoms_list = [(idx, os.path.join(symptoms_dir, path), idx == 0) for idx, path in enumerate(_temp_list)]

    # read in the patient and condition files
    patient_columns = ['Id', 'BIRTHDATE', 'RACE', 'ETHNICITY', 'GENDER']
    patients = pd.read_csv(os.path.join(data_dir, "patients.csv"), usecols=patient_columns)

    condition_columns = ['Id', 'PATIENT', 'CODE', 'ONSET']
    conditions = pd.read_csv(os.path.join(data_dir, "patient_conditions.csv"), usecols=condition_columns)

    # we'll use half the number of available cpus
    num_cpus = os.cpu_count()
    num_jobs = int(math.pow(2, math.ceil(math.log2(max(1, num_cpus//4)))))

    patient_conditions = conditions.merge(patients, how='left', left_on='PATIENT', right_on='Id', suffixes=('', '_pat'))

    # free memory ?
    del patients
    del conditions

    # now we call joblib and let it work it's magic!

    _ = Parallel(n_jobs=num_jobs)(
        delayed(parse_symptoms)(patient_conditions, condition_label, symptom_vector, output_dir, file_data) for
        file_data in symptoms_list)

    # we should all be done now!
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medvice Parser')

    parser.add_argument('--data_dir', help='Directory which contains the patients.csv, patient_conditions.csv, '
                                           'condition_codes.json and symptom_vector.json files')

    parser.add_argument('--output_dir', help='Directory where the parsed json files should be written to')

    parser.add_argument('--symptoms_dir', help='Directory which contains the split patient_condition_symptoms.csv files')

    parser.argument_default('--num_cpus', help='Specify the number of cpus to be used for this task')

    args = parser.parse_args()

    datadir = args.data_dir
    outputdir = args.output_dir
    symptomdir = args.symptoms_dir
    cpus = args.num_cpus

    if not datadir or not os.path.isdir(datadir):
        raise ValueError("A valid data directory is required")
    if not outputdir or not os.path.isdir(outputdir):
        raise ValueError("A valid otuput directory is required")
    if not symptomdir or not os.path.isdir(symptomdir):
        raise ValueError("A valid symptom directory is required")

    # let the ball roll!
    parse_files(datadir, symptomdir, outputdir, cpus)
