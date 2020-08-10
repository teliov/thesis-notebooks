import argparse
from datetime import datetime
from parse import parse_symptoms
from train_rf import train_rf
from helpers import terminate_instance
from .train_dl import train_dl


def build_parser():
    parser = argparse.ArgumentParser(description="Medvice AWS Runner")

    subparsers = parser.add_subparsers(dest='cmd')

    parse_group = subparsers.add_parser('parse', help='Parse a symptoms file stored in S3')
    parse_group.add_argument('--symptoms_db', type=str, help='S3 key to symptoms db')
    parse_group.add_argument('--conditions_db', type=str, help='S3 key to conditions db')
    parse_group.add_argument('--file', type=str, help='Gzipped symptoms file to be parsed')
    parse_group.add_argument('--run', type=str, help='The name for this run. The current date and time would be appended to this')

    train_rf_group = subparsers.add_parser('train_rf', help='Fit a RandomForest Classifier')
    train_rf_group.add_argument('--file', type=str, help='S3 path to train Data')
    train_rf_group.add_argument('--run', type=str, help='The name for this run. The current date and time would be appended to this')

    train_dl_group = subparsers.add_parser('train_dl', help='Fit a DNN')
    train_dl_group.add_argument('--run_name', type=str, help='Run Name')
    train_dl_group.add_argument('--train_file', type=str, help='S3 path to train Data')
    train_dl_group.add_argument('--mlflow_uri', type=str, help="URI for MlFlow tracking server")
    train_dl_group.add_argument('--input_dim', type=str, help="Input dimension for model")
    train_dl_group.add_argument('--num_symptoms', type=str, help="Num of symptoms in DB")
    train_dl_group.add_argument('--num_conditions', type=str, help="Num of Conditions in DB")
    train_dl_group.add_argument('--visdom_url', type=str, help="URL to visdom server")
    train_dl_group.add_argument('--visdom_port', type=str, help="Port to visdom server")
    train_dl_group.add_argument('--visdom_username', type=str, help="username to visdom server")
    train_dl_group.add_argument('--visdom_password', type=str, help="password to visdom server")
    train_dl_group.add_argument('--visdom_env', type=str, help="env to visdom server")
    train_dl_group.add_argument('--layer_config_file', type=str, help="path to layer configuration")



    return parser


if __name__ == "__main__":

    parser = build_parser()

    args = parser.parse_args()

    if args.cmd == 'parse':
        symptoms_db_file = args.symptoms_db
        conditions_db_file = args.conditions_db
        data_file = args.file
        run_name = args.run

        now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        run_name = "%s_%s" % (run_name, now)

        parse_symptoms(run_name, data_file, symptoms_db_file, conditions_db_file)
        terminate_instance()
    elif args.cmd == 'train_rf':
        data_file = args.file
        run_name = args.run
        now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        run_name = "%s_%s" % (run_name, now)
        train_rf(data_file, run_name)
        terminate_instance()

    elif args.cmd == 'train_dl':
        run_name = args.run_name
        train_file = args.train_file
        mlflow_uri = args.mlflow_uri
        input_dim = args.input_dim
        num_sympoms = args.num_symptoms
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
            num_sympoms,
            num_conditions,
            visdom_url,
            visdom_port,
            visdom_username,
            visdom_password,
            visdom_env,
            layer_config_file
        )
    terminate_instance()
