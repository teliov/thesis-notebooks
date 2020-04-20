from glob import glob
import pandas as pd
import sys
import os
import math


def combine_and_split(csv_dir, output_dir, train_split=70, validate_split=10):
    csv_regex = os.path.join(csv_dir, "*.csv")
    csv_files = glob(csv_regex)
    df = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df)
    num_elements = df.shape[0]
    train_idx = int(math.floor(train_split*num_elements/100))
    val_idx = train_idx + int(math.floor(validate_split*num_elements/100))
    train_df = df[:train_idx]
    val_df = df[train_idx: val_idx]
    test_df = df[val_idx:]

    filename = csv_dir.split("/")[-1] + ".csv"
    train_df.to_csv(os.path.join(output_dir, "train", filename))
    test_df.to_csv(os.path.join(output_dir, "test", filename))
    val_df.to_csv(os.path.join(output_dir, "val", filename))
    return True


if __name__ == "__main__":
    output_dir = "/shares/bulk/oagba/data/output_basic_50k/symptoms/"

    dirname = sys.argv[1]
    if not os.path.isdir(dirname):
        raise ValueError("Invalid directory")

    combine_and_split(dirname, output_dir)
