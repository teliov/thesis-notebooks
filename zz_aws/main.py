import argparse
from datetime import datetime
from parse import parse_symptoms


def build_parser():
    parser = argparse.ArgumentParser(description="Medvice AWS Runner")

    subparsers = parser.add_subparsers(dest='cmd')

    parse_group = subparsers.add_parser('parse', help='Parse a symptoms file stored in S3')
    parse_group.add_argument('--symptoms_db', type=str, help='S3 key to symptoms db')
    parse_group.add_argument('--conditions_db', type=str, help='S3 key to conditions db')
    parse_group.add_argument('--file', type=str, help='Gzipped symptoms file to be parsed')
    parse_group.add_argument('--run', type=str, help='The name for this run. The current date and time would be appended to this')

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
