import os

max_depth = [None] + [190, 200, 380]
min_samples_split = list(range(2, 10, 2))
max_features = ['sqrt', 'log2']
estimators = list(range(10, 100, 10))

params = []
for mxdepth in max_depth:
    for mss in min_samples_split:
        for mf in max_features:
            for est in estimators:
                param = {
                    "max_depth": mxdepth,
                    "min_samples_split": mss,
                    "min_samples_leaf": mss,
                    "max_leaf_nodes": None,
                    "max_features": mf,
                    "estimators": est
                }
                params.append(param)

JOB_TPL = """#!/bin/bash
#SBATCH -J gs_rf
#SBATCH -o gs.%j.out
#SBATCH -p general
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH -t 06:00:00
#SBATCH --mem  120G

source /shares/bulk/oagba/work/medvice-parser/activate

"""

if not os.path.isdir("run_batch"):
    os.mkdir("run_batch")

cmd_tpl = "python /home/oagba/bulk/thesis-notebooks/zz_scripts/rf_grid_search/train.py --data '/shares/bulk/oagba/data/output_basic_5k/symptoms/csv/parsed/train.csv_sparse.csv' \
    --output_dir '/home/oagba/bulk/data/output_basic_5k/symptoms/csv/training/random_forest/grid_search' \
    --symptoms_json '/shares/bulk/oagba/data/kk/json/symptom_db.json' "

count = 1
for param in params:
    cmd_str = ""
    for key, value in param.items():
        if value is None:
            continue
        cmd_str += "--%s %s " % (key, str(value))

    cmd = cmd_tpl + cmd_str

    filename = "run_batch/%d.job" % count

    job = JOB_TPL + cmd

    with open(filename, "w") as fp:
        fp.write(job)

    count += 1
