"""
Generates sbatch jobs to run for all the specified data points
"""
import numpy as np
import os
import pathlib

template_str = """#!/bin/bash
#SBATCH -J train_{model_name}
#SBATCH -o {model_name}.%j.out
#SBATCH -p general
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH -t 02:00:00
#SBATCH --mem  140G

workdir="/home/oagba/bulk/thesis-notebooks/zz_scripts/learner"
dir="{data_directory}"
source /shares/bulk/oagba/work/medvice-parser/bin/activate
output_dir="{output_dir}"

if [[ ! -d $output_dir ]];
then
    mkdir -p $output_dir
fi

python "${{workdir}}/train_rf.py" --data "{data_file}" \\
    --num_symptoms {num_symptoms} \\
    --output_dir ${{output_dir}} \\
    --train_size {train_size:.8f} \\
    --fold_number {fold_number} \\
    --name $dir \\
    --mlflow_uri {mlflow_uri} \\
    --save_model {save_model}

"""

base_path = "/home/oagba/bulk/data/"
data_directories = [
    "output_basic_15k",
    "output_basic_2_cnt_15k",
    "output_basic_3_cnt_15k",
    "output_basic_4_cnt_15k",
    "output_basic_avg_cnt_15k",
    "output_basic_pct_10_15k",
    "output_basic_pct_20_15k",
    "output_basic_pct_30_15k",
    "output_basic_rand_symptoms",
    "output_basic_weighted_symptoms"
]

train_file_location = "symptoms/csv/parsed/train.csv_sparse.csv"
output_dir_base = "symptoms/csv/parsed/learning"
NUM_SYMPTOMS = 376
TYPE="rf"
MLFLOW_URI="http://131.180.106.142:15000"

if __name__ == "__main__":
    train_sizes = np.arange(0.1, 1.1, 0.1)
    fold_numbers = list(range(1, 6))

    jobs_dir = os.path.join(os.getcwd(), "rfjobs")
    pathlib.Path(jobs_dir).mkdir(exist_ok=True, parents=True)

    job_names = []

    for item in data_directories:
        for train_count, train_size in enumerate(train_sizes):
            for fold_number in fold_numbers:
                output_directory = os.path.join(base_path, item, output_dir_base, TYPE)
                save_model = 1 if train_size >= 1.0 and fold_number == 5 else 0
                template_file = template_str.format(
                    model_name="nb",
                    data_directory=item,
                    output_dir=output_directory,
                    data_file=os.path.join(base_path, item, train_file_location),
                    num_symptoms=NUM_SYMPTOMS,
                    train_size= train_size,
                    fold_number=fold_number,
                    mlflow_uri=MLFLOW_URI,
                    save_model=save_model
                )

                job_name = "%s_%d_%d.job" %(item, train_count+1, fold_number)
                filename = os.path.join(jobs_dir, job_name)
                with open(filename, "w") as fp:
                    fp.write(template_file)
                job_names.append(job_name)

    run_main_file = os.path.join(jobs_dir, "run_main.sh")
    run_main_cv = os.path.join(jobs_dir, "run_main_cv.sh")
    run_file = os.path.join(jobs_dir, "run.sh")

    # when writing the run.sh file, we want to first handle the "important cases"
    # so full training size, fold_number_5 for all the data_directories
    # then we can build up data for the other folds for full training_size
    # then we add the rest for the learning curve

    with open(run_main_file, "w") as fp:
        fp.write("#!/usr/bin/env bash\n")
        for item in data_directories:
            job_cmd = "sbatch %s_10_5.job\n" % item
            fp.write(job_cmd)
        fp.write("\n")

    with open(run_main_cv, "w") as fp:
        # then we build up for the full training sizes, the other folds
        fold_numbers = list(range(1, 5))
        fp.write("#!/usr/bin/env bash\n")
        count = 0
        for item in data_directories:
            for fold_number in fold_numbers:
                job_cmd = "sbatch %s_10_%d.job\n" % (item,fold_number)
                fp.write(job_cmd)
                if (count+1) % 10 == 0:
                    fp.write("sleep 1\n")
                count +=1
        fp.write("\n")

    with open(run_file, "w") as fp:
        fp.write("#!/usr/bin/env bash\n")
        training_sizes = list(range(1, 10))
        # for all the rest
        count = 0
        for item in data_directories:
            for train_size in training_sizes:
                job_cmd = "sbatch %s_%d_%d.job\n" % (item, train_size, 5)
                fp.write(job_cmd)
                if (count + 1) % 10 == 0:
                    fp.write("sleep 1\n")
                count += 1

        fp.write("\n")
