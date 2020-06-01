import argparse
import os
import pathlib
import numpy as np
import pandas as pd
import json


def weighted_selection(item, cnd_code_list, symp_map, cnd_symp_hash, is_random=False):
    num_symptoms = len(item.SYMPTOMS.split(","))
    cnd_code = cnd_code_list[item.LABEL]
    cnd_symptoms_map = cnd_symp_hash[cnd_code]
    cnd_symptoms = sorted(cnd_symptoms_map.keys())
    cnd_probs = [cnd_symptoms_map[sym] for sym in cnd_symptoms]
    sum_probs = sum(cnd_probs)
    cnd_probs = [idx / sum_probs for idx in cnd_probs]
    cnd_symptoms = [symp_map[sym] for sym in cnd_symptoms]
    rng = np.random.default_rng()
    if is_random:
        selected_symp = rng.choice(cnd_symptoms, num_symptoms, replace=False)
    else:
        selected_symp = rng.choice(cnd_symptoms, num_symptoms, replace=False, p=cnd_probs)

    return ",".join(selected_symp.tolist())


def process_generation(symptom_dir, output_dir, cond_db_file, symp_db_file, cnd_symp_map_file, is_random=True):
    train_file = os.path.join(symptom_dir, "symptoms/csv/parsed/train.csv_sparse.csv")
    test_file = os.path.join(symptom_dir, "symptoms/csv/parsed/test.csv_sparse.csv")
    if not os.path.isfile(train_file) or not os.path.isfile(test_file):
        raise ValueError("Train or test csv files do not exist")

    final_op_dir = os.path.join(output_dir, "symptoms/csv/parsed")
    pathlib.Path(final_op_dir).mkdir(parents=True, exist_ok=True)

    with open(cond_db_file) as fp:
        conditions_db = json.load(fp)
    with open(symp_db_file) as fp:
        symptoms_db = json.load(fp)
    with open(cnd_symp_map_file) as fp:
        condition_prob_hash = json.load(fp)

    sorted_conditions = sorted(conditions_db.keys())
    sorted_symptoms = sorted(symptoms_db.keys())
    symptom_map = {code: str(idx) for idx, code in enumerate(sorted_symptoms)}

    for item in [train_file, test_file]:
        basename = os.path.basename(item)
        df = pd.read_csv(item, index_col="Index")
        cnd_sym = df[['LABEL', 'SYMPTOMS']]
        df['SYMPTOMS'] = cnd_sym.apply(
            weighted_selection,
            axis=1,
            cnd_code_list=sorted_conditions,
            symp_map=symptom_map,
            cnd_symp_hash=condition_prob_hash,
            is_random=is_random
        )
        op_file = os.path.join(final_op_dir, basename)
        df.to_csv(op_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medvice NB Inference Engine')
    parser.add_argument('--symptom_dir', help='Symptom path')
    parser.add_argument('--output_dir', help='Directory where results should be saved to')
    parser.add_argument('--cond_db', help="Condition db")
    parser.add_argument('--symp_db', help="Symptom db")
    parser.add_argument('--random', help="Symptom db", action="store_true")
    parser.add_argument('--cnd_symp_map', help='condition symptom map')

    args = parser.parse_args()
    symp_dir = args.symptom_dir
    cond_db = args.cond_db
    output_dir = args.output_dir
    symp_db = args.symp_db
    cnd_symp_map = args.cnd_symp_map
    is_random = args.random

    if not os.path.isfile(cond_db):
        raise ValueError("Condition db file does not exist")

    if not os.path.isfile(symp_db):
        raise ValueError("Symptoms db file does not exist")

    if not os.path.isfile(cnd_symp_map):
        raise ValueError("Condition symptom map does not exist")

    process_generation(
        symptom_dir=symp_dir,
        output_dir=output_dir,
        cond_db_file=cond_db,
        symp_db_file=symp_db,
        cnd_symp_map_file=cnd_symp_map,
        is_random=is_random
    )
