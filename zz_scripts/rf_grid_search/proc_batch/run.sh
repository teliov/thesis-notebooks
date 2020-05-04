#!/usr/bin/env bash


concat_dir="../run_batch"
concat_dir=`readlink -f $concat_dir`

if [[ ! -d "jobs" ]];
then
    mkdir jobs
fi

cd jobs

for entry in "$concat_dir"/*
do

    sbatch $entry
done