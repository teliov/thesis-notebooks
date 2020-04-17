This notebook attempts using the `partial_fit` attribute when training a classifier.

We'll try with two models: RandomForest and Naive Bayes.

### Update

RandomForest does not have a partial fit parameter. And the tricky thing is that getting it to work with the partial fit might be a real pain and would require some serious level of verification to confirm that it does what is actually says.

The good news is that I don't need to do it that way. I can request resources from the cluster that would be able to fit the data I have generated comfortably in memory. I would of course loose the advantage of running in a notebook, but I think this is a smaller price to pay than figuring out the partial fit business just now.

#### Attack Plan
- Run the data processing pipeline on the generated data
- Use a sequential panda script running on a system with enough resources to concatenate all the symptoms into one file.
- Do a shuffle split to get train and test split.
- Save the test split to a particular location
- Test run with the train split, no cross validation, no grid search.
- Then come back here and expand this list.
