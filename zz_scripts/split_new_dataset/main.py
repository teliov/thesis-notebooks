import os
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def split_data(symptom_file, output_path, use_headers=False, train_split=0.8):
    symptom_columns = ['PATIENT', 'GENDER', 'RACE', 'ETHNICITY', 'AGE_BEGIN', 'AGE_END',
                       'PATHOLOGY', 'NUM_SYMPTOMS', 'SYMPTOMS']

    if use_headers:
        df = pd.read_csv(symptom_file, names=symptom_columns)
    else:
        df = pd.read_csv(symptom_file)

    df.index.name = "Index"

    labels = df["PATHOLOGY"]
    splitter = StratifiedShuffleSplit(1, train_size=train_split)
    train_index = None
    test_index = None
    for tr_idx, tst_index in splitter.split(df, labels):
        train_index = tr_idx
        test_index = tst_index
        break

    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

    train_df.to_csv(os.path.join(output_path, "train.csv"))
    test_df.to_csv(os.path.join(output_path, "test.csv"))

    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Medvice Data Splitter')
    parser.add_argument('--symptom_file', help='Symptom file to split')
    parser.add_argument('--output_dir', help='Directory where the split csv output should be written to')
    parser.add_argument('--use_headers', action='store_true', help='Is the file missing headers')
    parser.add_argument('--train_split', type=float, default=0.8, help="train split")

    args = parser.parse_args()
    symptom_file = args.symptom_file
    output_dir = args.output_dir
    use_headers = args.use_headers
    train_split = args.train_split

    if not os.path.isfile(symptom_file):
        raise ValueError("Symptom file does not exist")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    split_data(symptom_file, output_dir, use_headers, train_split)
