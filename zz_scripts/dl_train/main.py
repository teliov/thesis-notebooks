import argparse

from thesislib.utils.dl.bench import train_dl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medvice AWS Runner")

    train_dl_group = parser
    train_dl_group.add_argument('--run_name', type=str, help='Run Name')
    train_dl_group.add_argument('--train_file', type=str, help='S3 path to train Data')
    train_dl_group.add_argument('--mlflow_uri', type=str, help="URI for MlFlow tracking server")
    train_dl_group.add_argument('--input_dim', type=int, help="Input dimension for model")
    train_dl_group.add_argument('--num_symptoms', type=int, help="Num of symptoms in DB")
    train_dl_group.add_argument('--num_conditions', type=int, help="Num of Conditions in DB")
    train_dl_group.add_argument('--visdom_url', type=str, help="URL to visdom server")
    train_dl_group.add_argument('--visdom_port', type=str, help="Port to visdom server")
    train_dl_group.add_argument('--visdom_username', type=str, help="username to visdom server")
    train_dl_group.add_argument('--visdom_password', type=str, help="password to visdom server")
    train_dl_group.add_argument('--visdom_env', type=str, help="env to visdom server")
    train_dl_group.add_argument('--layer_config_file', type=str, help="path to layer configuration")

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

    train_dl(
        run_name,
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
        epochs=10
    )

