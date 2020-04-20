from glob import glob
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
import os
import argparse


def concatenate_and_split(filepath, output_path, train_split=0.8):
    filenames = sorted(glob(filepath))
    print("Reading %d files" % len(filenames))
    df = [pd.read_csv(file) for file in filenames]
    print("Done Reading %d files" % len(filenames))
    print("Concatenating %d dataframes" % len(filenames))
    df = pd.concat(df)
    print("Done Concatenating %d dataframes" % len(filenames))

    # now we have one big df
    labels = df.LABEL

    splitter = StratifiedShuffleSplit(1, train_size=train_split)
    train_index = None
    test_index = None
    for tr_idx, tst_index in splitter.split(df, labels):
        train_index = tr_idx
        test_index = tst_index
        break

    train_df = df.iloc[train_index]
    print("Saving train set")
    train_df.to_csv(os.path.join(output_path, "train.csv"))
    print("Done Saving train set")
    del train_df

    test_df = df.iloc[test_index]
    print("Saving test set")
    test_df.to_csv(os.path.join(output_path, "test.csv"))
    print("Done Saving test set")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medvice Data Concat and Split')

    parser.add_argument('--data_dir', help='Regex to split files', type=str)
    parser.add_argument('--output_dir', help='Directory to save concat files', type=str)
    parser.add_argument('--train_split', help='Size of train split', type=float, default=0.8)

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    train_split = args.train_split

    concatenate_and_split(data_dir, output_dir, train_split)
