import numpy as np
import botocore.session
import json
from collections import OrderedDict
import helpers
from sklearn.model_selection import StratifiedShuffleSplit
import time
import logging
import swifter


def parse_symptoms(data_name, data_file, symptoms_db_file, conditions_db_file, telegram=True):

    # setup logging
    logger = helpers.Logger(data_name)
    try:
        start_time = time.time()
        message = "Starting Reading Files from S3"
        logger.log(message)

        session = botocore.session.get_session()

        s3 = session.create_client('s3', region_name=helpers.AWS_REGION)

        symptoms_db_obj = s3.get_object(Bucket=helpers.S3_BUCKET, Key=symptoms_db_file)
        conditions_db_obj = s3.get_object(Bucket=helpers.S3_BUCKET, Key=conditions_db_file)

        symptoms_db = json.load(symptoms_db_obj['Body'])
        conditions_db = json.load(conditions_db_obj['Body'])

        symptom_vector = sorted(symptoms_db.keys())
        condition_codes = sorted(conditions_db.keys())
        condition_labels = {code: idx for idx, code in enumerate(condition_codes)}

        usecols = ["GENDER", "RACE", "AGE_BEGIN", "PATHOLOGY", "NUM_SYMPTOMS", "SYMPTOMS"]
        dtype = {
            "GENDER": "category",
            "RACE": "category",
            "AGE_BEGIN": np.uint16,
            "PATHOLOGY": np.object,
            "SYMPTOMS": np.object,
            "NUM_SYMPTOMS": np.uint8
        }
        symptoms_df = helpers.s3_to_pandas(s3, helpers.S3_BUCKET, data_file, usecols=usecols, dtype=dtype)

        curr_time = time.time()
        message = "Finished Reading Files from S3\n Took: %.3f seconds.\nStarting symptoms processor" % (curr_time - start_time)
        logger.log(message)

        start_proc_time = time.time()

        symptoms_df = symptoms_df.loc[symptoms_df.NUM_SYMPTOMS > 0]
        symptoms_df['LABEL'] = symptoms_df.PATHOLOGY.swifter.progress_bar(False).apply(helpers.label_txform, labels=condition_labels).astype(np.uint16)
        symptoms_df.RACE = symptoms_df.RACE.swifter.progress_bar(False).apply(helpers.race_txform).astype(np.uint8)
        symptoms_df.GENDER = symptoms_df.GENDER.swifter.progress_bar(False).apply(lambda gender: 0 if gender == 'F' else 1).astype(np.uint8)
        symptoms_df = symptoms_df.rename(columns={'AGE_BEGIN': 'AGE'})

        # handle the transformation of the symptoms ...
        symptom_index_map = OrderedDict({code: 2 ** idx for idx, code in enumerate(symptom_vector)})

        symptoms_df['NSYMPTOMS'] = symptoms_df.SYMPTOMS.swifter.progress_bar(False).apply(helpers.symptom_transform, labels=symptom_index_map).astype(np.object)

        for idx, code in enumerate(symptom_vector):
            symptoms_df[code] = symptoms_df.NSYMPTOMS.swifter.progress_bar(False).apply(helpers.check_bitwise, comp=2**idx)

        ordered_keys = ['LABEL', 'GENDER', 'RACE', 'AGE'] + symptom_vector
        symptoms_df = symptoms_df[ordered_keys]

        end_proc_time = time.time()

        message = "Completed symptoms processing.\n Took: %.3f seconds.\n Starting stratified split and shuffling" % (end_proc_time - start_proc_time)
        logger.log(message)

        # generate a stratified test and train split
        start_shuffling_time = time.time()
        split = StratifiedShuffleSplit(n_splits=1)
        labels = symptoms_df.LABEL

        train_index = None
        test_index = None
        for tr_idx, tst_idx in split.split(symptoms_df, labels):
            train_index = tr_idx
            test_index = tst_idx

        train_df = symptoms_df.iloc[train_index]
        test_df = symptoms_df.iloc[test_index]

        end_shuffling_time = time.time()

        message = "Completed data split and shuffle.\n Took %.3f seconds.\nSaving to s3" % (end_shuffling_time - start_shuffling_time)
        logger.log(message)

        # save to s3
        start_s3_dump = time.time()
        output_directory = "output/%s" % data_name
        train_file = "%s/%s" % (output_directory, "train.csv.gz")
        test_file = "%s/%s" % (output_directory, "test.csv.gz")

        helpers.pandas_to_s3(train_df, s3, helpers.S3_BUCKET, train_file)
        helpers.pandas_to_s3(test_df, s3, helpers.S3_BUCKET, test_file)

        end_s3_dump = time.time()

        message = "Completed dump to s3. Took %.3f seconds.\n Run on %s complete!\n Took %.3f seconds" % (
            end_s3_dump - start_s3_dump,
            data_name,
            end_s3_dump - start_time
        )
        logger.log(message)

        log_file = "%s/%s" % (output_directory, "parse.log")
        s3.put_object(Bucket=helpers.S3_BUCKET, Key=log_file, Body=logger.to_string())

        res = True
    except Exception as e:
        message = e.__str__()
        logger.log(message, logging.ERROR)
        res = False

    return res



