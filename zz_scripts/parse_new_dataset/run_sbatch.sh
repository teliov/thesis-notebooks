#! /usr/bin/env bash

search_dir="/home/oagba/parse-jobs"
job_dir="/home/oagba/op_batch"

cd $job_dir

for entry in "$search_dir"/*
do
  sbatch $entry
done