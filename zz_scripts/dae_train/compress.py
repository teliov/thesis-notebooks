"""
This script uses a trained DAE model to compress the dataset by transforming
the sparse features into the latent representation
"""

from thesislib.utils.dl.dae import DAE
from thesislib.utils.dl.utils import to_device, get_default_device
from thesislib.utils.dl.models import DLSparseMaker

import os
import pandas as pd
import numpy as np
import pathlib
import argparse

import torch
from torch.utils.data import DataLoader


def compress(state_dict_path, train_file_path, num_symptoms, output_dir):
    if not os.path.exists(state_dict_path):
        raise ValueError("Invalid state dict path passed")

    if not os.path.exists(train_file_path):
        raise ValueError("Invalid train path passed")

    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)

    train_artifacts = torch.load(state_dict_path)
    if not "input_dim" in train_artifacts or not "target_dim" in train_artifacts \
            or not "model_dict" in train_artifacts:
        raise ValueError("Invalid train artifacts retrieved from state dict")

    input_dim = train_artifacts.get("input_dim")
    target_dim = train_artifacts.get("target_dim")
    state_dict = train_artifacts.get("model_dict")

    dae = DAE(input_dim=input_dim, target_dim=target_dim)
    dae.load_state_dict(state_dict)

    df = pd.read_csv(train_file_path, index_col="Index")

    sparsifier = DLSparseMaker(num_symptoms=num_symptoms)
    df = sparsifier.fit_transform(df)

    race_symptoms = df[:, 3:]
    df = df[:, :3]

    assert input_dim == race_symptoms.shape[1], \
        "Dimension of prepped data (%d) does not match specified input dimension (%d)" % (input_dim, race_symptoms.shape[1])

    device = get_default_device()
    dae = to_device(dae, device)

    num_samples = race_symptoms.shape[0]
    batch_size = 1024*1024
    start = 0
    end = start + batch_size
    with torch.no_grad():
        if num_samples <= batch_size:
            tensor = torch.FloatTensor(race_symptoms.todense())
            compressed = dae.encoder(to_device(tensor, device)).cpu().numpy()
        else:
            compressed = np.zeros((num_samples, target_dim), dtype=np.float32)
            while end <= num_samples:
                tensor = torch.FloatTensor(race_symptoms[start: end, :].todense())
                temp = dae.encoder(to_device(tensor, device)).cpu().numpy()
                compressed[start:end, :] = temp

                start += batch_size
                end += batch_size

                if end > num_samples:
                    end = num_samples

    df = np.hstack((df.todense(), compressed))
    df = pd.DataFrame(df)

    opfilename = "comp_%s" % os.path.basename(train_file_path)
    opfilepath = os.path.join(output_dir, opfilename)

    df.to_csv(opfilepath, index_label="Index")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compressor")

    parser.add_argument('--state_dict_path', type=str, help='Path to State Dict')
    parser.add_argument('--data_path', type=str, help='Path to train file')
    parser.add_argument('--num_symptoms', type=int, help='Number of symptoms')
    parser.add_argument('--output_dir', type=str, help='Output directory')

    args = parser.parse_args()

    state_dict_path = args.state_dict_path
    data_path = args.data_path
    num_symptoms = args.num_symptoms
    output_dir = args.output_dir

    compress(state_dict_path, data_path, num_symptoms, output_dir)
