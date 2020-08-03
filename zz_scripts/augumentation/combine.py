import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
import pathlib

BASE_DIR="/home/oagba/bulk/data/"
TARGET_SOURCES = {
    "output_basic_15k": 20,
    "output_basic_2_cnt_15k": 10,
    "output_basic_3_cnt_15k": 10,
    "output_basic_4_cnt_15k": 10,
    "output_basic_avg_cnt_15k": 10,
    "output_basic_pct_10_15k": 10,
    "output_basic_pct_20_15k": 10,
    "output_basic_pct_30_15k": 10,
    "output_basic_pct_50_15k": 10,
    "output_basic_pct_70_15k": 10,
    "output_basic_inc_1_15k": 10,
    "output_basic_inc_2_15k": 10,
    "output_basic_inc_3_15k": 10
}

if __name__ == "__main__":

    test_dfs = []
    train_dfs = []

    for dirname, pct in TARGET_SOURCES.items():

        split_t = StratifiedShuffleSplit(n_splits=1, test_size=pct/100)

        test_file = os.path.join(BASE_DIR, dirname, "symptoms/csv/parsed/test.csv_sparse.csv")
        train_file = os.path.join(BASE_DIR, dirname, "symptoms/csv/parsed/train.csv_sparse.csv")

        files = [test_file, train_file]

        for idx in range(2):
            df = pd.read_csv(files[idx], index_col="Index")
            label_values = df.LABEL.values

            for _, test_index in split_t.split(df, label_values):
                df = df.iloc[test_index]

            if idx == 0:
                test_dfs.append(df)
            else:
                train_dfs.append(df)

    opdir = os.path.join(BASE_DIR, "output_combined_15k", "symptoms/csv/parsed")
    pathlib.Path(opdir).mkdir(exist_ok=True, parents=True)

    # test
    df = pd.concat(test_dfs)
    filename = os.path.join(opdir, "test.csv_sparse.csv")
    df.to_csv(filename)

    # train
    df = pd.concat(train_dfs)
    filename = os.path.join(opdir, "train.csv_sparse.csv")
    df.to_csv(filename)
