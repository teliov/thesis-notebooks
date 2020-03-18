This script attempts to do a parallel processing of large csv files using pandas and joblibe

Since multiple csv files need to combined to get the input dataset, this script parses the files in a 
possibly *scalable, parallel* manner.

There are three files which this script operates on:
1. `patients.csv` which holds patient data
2. `patient_conditions.csv` which holds the condition data
3. `patient_condition_symptoms.csv` which holds data for the symptoms.

**N.B.**:

For the contents of these file see the `symcat-to-synthea` project

Typically the `patient_condition_symptoms.csv` file is the larger file and the one that needs to be handled with care.

In the current case it is **51G** and definitely cannot fit into memory. 

In a pre-processing step this file should be split into smaller file sizes using 
Linux's `split` command like so:

```bash
split -l <num_of_lines> patient_condition_symptoms.csv
```

The `<num_of_lines>` should be chosen to create decently sized smaller files. A value of 1800000 should
produce files that are about 380MB in size (which can fit in memory in most systems)

The script can then be run on these files in combination with the `patients.csv` and `patient_conditions.csv` files.