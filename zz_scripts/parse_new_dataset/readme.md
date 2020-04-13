### Description

This repo contains code for parsing Synthea generated data according to the afansi-symptom-export format.

This format is advantageous for one main reason:
- It stores all the required data in one file:
    - All relevant patient data is included: age, gender, race, ethnicity, etc
    - Condition code is also included
    - Symptoms are also included

As a result when processing the data, no joins/merges are required as was the case when
the data was split across three files (symptoms, conditions and patients)

Also since all the data is one file, the file can be split into smaller chunks and processed individually.
In the previous format the only file that could be split was the symptoms file. The patients and conditions
file could not be split and would have to be included with every symptom split-chunk.

Since the new format only contains data that is actually needed, it is much smaller in size. As an illustration,
for 100k patients, the previous format generated a 13M patients file, a 12G conditions file and a 60G symptoms file. In
comparison this new format generates a single 17G file containing all relevant data.

The generator is also faster as it can be configured to skip exporting to every other format.

Using these favourable characteristics, this repo provides three scripts:

1. A python script that parses a split chunk and outputs a csv file in the desired format
2. A python script that generates an sbatch job file for all the split chunks that are available so they can be 
submitted to the cluster
3. A bash runner that submits all the generated sbatch jobs to the cluster for processing.


The chunks should be split in such a way that they can comfortably fit in the memory and hence pandas can be used 
to process the data. The size of each chunk should thus depend on how much resources can be requested from the cluster.

Also the generated output file is approximately 2.5 times larger than the initial input chunk. This means that given a 
chunk split of 1.1G, the final output file would be approximately 2.75G in size. Allowing for other memory consuming
processes, about 4G of RAM would be sufficient to process the data quite comfortably.
This example illustrates the kind of calculations that should be made when determining how large the chunk split should be.

On average `2000` lines in the file corresponds to 1M of data.

The command for splitting the files is:
```bash
split -l <num_lines_per_file> symptoms.csv
```

where `num_lines_per_file` is chosen to create the desired size of chunk splits. 

Also note that an advantage of using easy to fit in memory chunks is that it is possible to request more cores from the
cluster with a small memory requirement which would allow for more parallelism on each job by panda, shorter processing times
and makes it easier for the job to get scheduled in the queue.
