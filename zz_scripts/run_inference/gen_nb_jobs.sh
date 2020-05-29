#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

JOB_STR="#!/bin/bash
#SBATCH -J infer
#SBATCH -o nb.%j.out
#SBATCH -p general
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH -t 00:30:00
#SBATCH --mem  140G

source /shares/bulk/oagba/work/medvice-parser/activate

"

OPTIND=1

# variable init
data_dir=""
test_files=""
symptom_db=""
run_sbatch=0
output_dir=""
data_paths=()
is_nb=1

while getopts "d:s:t:o:p:er" opt; do
    case "$opt" in
    d)
        data_dir=$OPTARG
        ;;
    s)
        symptom_db=$OPTARG
        ;;
    t)
        test_files=$OPTARG
        ;;
    e)
        run_sbatch=1
        ;;
    o)
        output_dir=$OPTARG
        ;;
    p)
        for dir in $(echo $OPTARG | sed "s/,/ /g"); do
            data_paths=(${data_paths[@]} ${dir})
        done
        ;;
    r)
        is_nb=0
        ;;
    *)
        printf "Got unknown option $OPTSTRING \n"
        exit 1
    esac
done

res_dir="${data_dir}/symptoms/csv/inference"
if [[ $is_nb -eq 1 ]];then
model_file="${data_dir}/symptoms/csv/trained/nb/nb_serialized_sparse.joblib"
SPARSER=${SCRIPT_DIR}/nb.py
else
model_file="${data_dir}/symptoms/csv/trained/rf/rf_serialized_sparse_grid_search_best.joblib"
SPARSER=${SCRIPT_DIR}/rf.py
fi

if [[ ! -d res_dir ]]; then
    mkdir -p ${res_dir}
fi

if [[ ! -d output_dir ]]; then
    mkdir -p ${output_dir}
fi

for val in "${data_paths[@]}"; do
    val_base="$(basename $val)"
    op_file="${output_dir}/${val_base}.job"
    parse_cmd="python ${SPARSER} --model_file ${model_file} --symptoms_json ${symptom_db} --output_dir ${res_dir}\
                --test_files ${test_files}"
    job_content="${JOB_STR}${parse_cmd}"
    echo "$job_content" | tee -a ${op_file} > /dev/null 2>&1
done

# run through all the jobs and execute them
if [[ ${run_sbatch} -eq 1 ]];then
    cd ${output_dir}
    for entry in "${output_dir}/*.job"
    do
      sbatch ${entry}
    done
fi