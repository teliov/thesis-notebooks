import argparse
from glob import glob
import os

JOB_STR = """#!/bin/bash
#SBATCH -J parse-python
#SBATCH -o op.%j.out
#SBATCH -p general
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH -c 2
#SBATCH -t 00:30:00
#SBATCH --mem  20G

source /shares/bulk/oagba/work/medvice-parser/bin/activate
"""


def generate_job(symptom_file, symptom_json, condition_json, output_dir, use_headers, job_directory):

    filename = symptom_file.split("/")[-1]
    jobname = "run_job_%s" % filename

    dir_path = os.path.dirname(os.path.realpath(__file__))
    exec_file = os.path.join(dir_path, "main.py")

    python_path = "/shares/bulk/oagba/work/medvice-parser/bin/python"

    command = "%s %s --symptom_file %s --symptoms_json %s --conditions_json %s --output_dir %s" % (
        python_path,
        exec_file,
        symptom_file,
        symptom_json,
        condition_json,
        output_dir
    )

    if use_headers:
        command += " --use_headers"

    jobfile = os.path.join(job_dir, jobname)

    jobstr = JOB_STR + "\n" + command + "\n"
    with open(jobfile, "w") as fp:
        fp.write(jobstr)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medvice Job Creator')

    parser.add_argument('--data_dir', help='Path to symptom files')

    parser.add_argument('--regex', help='Regex for globbing symptom files. Defaults to \'x*\'')

    parser.add_argument('--symptoms_json', help='Path to symptoms db.json')

    parser.add_argument('--conditions_json', help='Path to conditions_db.json')

    parser.add_argument('--output_dir', help='Directory where the parsed csv output should be written to')
    parser.add_argument('--job_dir', help='Directory where the job files should be written to')

    args = parser.parse_args()

    data_dir = args.data_dir
    symptoms_db_json = args.symptoms_json
    conditions_db_json = args.conditions_json
    output_dir = args.output_dir
    job_dir = args.job_dir

    symp_regex = args.regex

    if not symp_regex:
        symp_regex = "x*"

    if not os.path.isfile(symptoms_db_json) or not os.path.isfile(conditions_db_json):
        raise ValueError("Invalid symptoms and/or conditions db file passed")

    if not os.path.isdir(data_dir):
        raise ValueError("Data directory does not")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not os.path.isdir(job_dir):
        os.mkdir(job_dir)

    symptom_pattern = os.path.join(data_dir, symp_regex)
    symptom_files = sorted(glob(symptom_pattern))

    for idx, filepath in enumerate(symptom_files):
        use_headers = idx > 0

        generate_job(filepath, symptoms_db_json, conditions_db_json, output_dir, use_headers, job_dir)
