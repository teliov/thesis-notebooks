import os
import sys
import argparse


def train_rf(data_file, output_dir):
    pass

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.realpath(__file__))

    module_path = os.path.join(file_path, "../..")
    module_path = os.path.abspath(module_path)

    if module_path not in sys.path:
        sys.path.append(module_path)

    from thesislib.utils.ml import models

    parser = argparse.ArgumentParser(description='Medvice RandomForest Trainer')
    parser.add_argument('--data', help='Path to train csv file')
    parser.add_argument('--output_dir', help='Directory where results and trained model should be saved to')

    args = parser.parse_args()
    data_file = args.data
    output_dir = args.output_dir

    if not os.path.isfile(data_file):
        raise ValueError("data file does not exist")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    train_nb(data_file=data_file, output_dir=output_dir)