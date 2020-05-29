#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SPARSER=${SCRIPT_DIR}/main.py

JOB_STR="#!/bin/bash
#SBATCH -J parse-python
#SBATCH -o op.%j.out
#SBATCH -p general
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH -c 2
#SBATCH -t 00:30:00
#SBATCH --mem  20G

source /shares/bulk/oagba/work/medvice-parser/bin/activate

"

OPTIND=1

# variable init
output_dir=""
data_paths=()
symptom_db=""
condition_db=""
run_sbatch=0

while getopts "d:o:s:c:e" opt; do
    case "$opt" in
    o)
        output_dir=$OPTARG
        ;;
    d)
        for dir in $(echo $OPTARG | sed "s/,/ /g"); do
            data_paths=(${data_paths[@]} ${dir})
        done
        ;;
    s)
        symptom_db=$OPTARG
        ;;
    c)
        condition_db=$OPTARG
        ;;
    e)
        run_sbatch=1
        ;;
    esac
done

if [[ ! -d ${output_dir} ]];then
    mkdir -p ${output_dir}
fi

# delete all contents from the directory (a previous run perhaps)
if [[ "$(ls -A ${output_dir})" ]]; then
    rm -rf $output_dir/*
fi

for val in "${data_paths[@]}"; do
    val_base="$(basename $val)"
    op_file="${output_dir}/${val_base}.job"
    symptom_file="${val}/symptoms/csv/split/test.csv"
    op_dir="${val}/symptoms/csv/parsed"
    parse_cmd="${SPARSER} --symptom_file ${symptom_file} --symptoms_json ${symptom_db} --conditions_json ${condition_db}\
                --output_dir ${op_dir}"
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