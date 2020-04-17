### Description

As a requirement for this method to work, a system which can hold all the processed
data in memory is required. 
This way we can avoid having to use a partial fit solution and just train at once.
With no incidence limit in the modules, and 50K patients generated, a 60GB memory allocation
would be ideal

#### Steps
- Use pandas to concatenate the result, and create an 80/20 train test split.