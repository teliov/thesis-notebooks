#!/usr/bin/env bash


concat_dir="/shares/bulk/oagba/data/output_basic_50k/symptoms/concat"

if [[ ! -d "jobs" ]];
then
    mkdir jobs
fi

cd jobs

for entry in "$concat_dir"/*
do

    bname="$(basename $entry)"
    cp "../job_template" "$bname.job"
    echo "python ../concat.py $entry" | tee -a "$bname.job"
    sbatch "$bname.job"
done