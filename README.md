## Description

This repository contains Jupyter notebooks and scripts which were utilized during the course of my Thesis topic. 

The notebooks serve as a sort of log of activities and are grouped in folders according to date using the naming schema: `MM-DD`.

For the most part, the model training and evaluation was carried out on the 
[TU Delft QCE Server Cluster](http://qce-it-infra.ewi.tudelft.nl/qce_servers.html). This cluster utilized 
[SLURM](https://slurm.schedmd.com/overview.html) for scheduling jobs on the cluster. The `zz_scripts` directory contains
scripts which were run on the cluster. Some of the folders also include sample SLURM job scripts which were submitted (and also
job script generators - which were used to generate scripts to allow submitting multiple related jobs).

During the course of the project, the feasibility of using AWS as a compute platform was explored. While this was abandoned 
due to the cost implications, the `zz_aws` folder contains code which shows provisioning of AWS ec2 instance(s), running 
a provided script (job), logging results in AWS S3 and terminating the instance.

While the notebooks mentioned earlier were run on a local computer - specifically a 2015 Mac Book running 
macOS High Sierra version 10.13 with 8GB of Ram and 2.9GHz Intel Core i5 - there were cases were it was required to run
the notebooks on the cluster either due to computation limitations of the local machine or because the input data was 
located on the server. The `zz_qce_server` contains these notebooks.

## Requirements

### Conda

The notebooks were run in an [anaconda](https://www.anaconda.com/) environment. A `requirements.txt` file which should
help with reproducing this environment is included in this repo.


### Thesislib

As the project progressed, key components of the data processing,  model training and evaluation tasks, etc were bundled 
in a python library unambiguously called [`Thesislib`](https://github.com/teliov/thesislib). As at the time of writing, 
the [latest version](https://github.com/teliov/thesislib/commit/3adb023c6d54ac6b8ca3ccb58328ac575a315968) of this library should also be installed
in your conda environment (or other environment in which this code is to be run).

### Data Files

While exploring in the notebook, certain assumptions were made regarding the location of the data. Hence absolute directories
were used more often than not. Also since the data used was often generated, the data files are not included. The generation
process however is easily reproducible.