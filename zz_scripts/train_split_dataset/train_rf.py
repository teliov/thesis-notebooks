import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier

def do_train(data_file):
    data_df = pd.read_csv(data_file)
    labels = data_df.LABEL
    data_df = data_df.drop(columns=['LABEL'])

    split_t = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    train_index = None
    test_index = None

    for tr_idx, tst_idx in split_t.split(data_df, labels):
        train_index = tr_idx
        test_index = tst_idx
        break

    train_data = data_df.iloc[train_index]
    train_label = labels.iloc[train_index]

    test_data = data_df.iloc[test_index]
    test_label = data_df.iloc[test_index]

    clf = RandomForestClassifier(n_estimators=140, criterion='gini', max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                 max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                 bootstrap=True, oob_score=False, n_jobs=2, random_state=None, verbose=0,
                                 warm_start=False, class_weight=None)
    pass
