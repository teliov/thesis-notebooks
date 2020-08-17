import argparse
import os

from thesislib.utils.dl.bench import train_dl, train_aedl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medvice AWS Runner")

    train_dl_group = parser
    train_dl_group.add_argument('--run_name', type=str, help='Run Name')
    train_dl_group.add_argument('--train_file', type=str, help='S3 path to train Data')
    train_dl_group.add_argument('--mlflow_uri', type=str, help="URI for MlFlow tracking server")
    train_dl_group.add_argument('--input_dim', type=int, help="Input dimension for model", default=383)
    train_dl_group.add_argument('--num_symptoms', type=int, help="Num of symptoms in DB", default=376)
    train_dl_group.add_argument('--num_conditions', type=int, help="Num of Conditions in DB", default=801)
    train_dl_group.add_argument('--visdom_url', type=str, help="URL to visdom server", default="")
    train_dl_group.add_argument('--visdom_port', type=str, help="Port to visdom server", default="")
    train_dl_group.add_argument('--visdom_username', type=str, help="username to visdom server", default="")
    train_dl_group.add_argument('--visdom_password', type=str, help="password to visdom server", default="")
    train_dl_group.add_argument('--visdom_env', type=str, help="env to visdom server", default="")
    train_dl_group.add_argument('--layer_config_file', type=str, help="path to layer configuration", default="")
    train_dl_group.add_argument('--epochs', type=int, help="Number of epochs to run for", default=50)
    train_dl_group.add_argument('--pre_train', action='store_true', help="Use Pretrained DAE for compression")
    train_dl_group.add_argument('--dae_path', type=str, help="Path to trained DAE", default="")

    args = parser.parse_args()

    run_name = args.run_name
    train_file = args.train_file
    mlflow_uri = args.mlflow_uri
    input_dim = args.input_dim
    num_symptoms = args.num_symptoms
    num_conditions = args.num_conditions
    visdom_url = args.visdom_url
    visdom_port = args.visdom_port
    visdom_username = args.visdom_username
    visdom_password = args.visdom_password
    visdom_env = args.visdom_env
    layer_config_file = args.layer_config_file
    epochs = args.epochs
    use_pre_train = args.pre_train
    dae_path = args.dae_path

    if use_pre_train:
        if len(dae_path) < 0 or not os.path.exists(dae_path):
            raise ValueError("If using the pre-trained procedure, you must provide a valid path to the DAE module")

        train_aedl(
            run_name,
            dae_path,
            train_file,
            mlflow_uri,
            input_dim,
            num_symptoms,
            num_conditions,
            visdom_url,
            visdom_port,
            visdom_username,
            visdom_password,
            visdom_env,
            layer_config_file,
            epochs=epochs
        )

    else:
        train_dl(
            train_file,
            mlflow_uri,
        )

