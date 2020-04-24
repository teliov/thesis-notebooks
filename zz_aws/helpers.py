import requests
import pandas as pd
import io
import gzip
import logging

TELEGRAM_CHAT_ID = "@oagba_qce_aws"

BASE_URL = "https://api.telegram.org/bot1150250526:AAEFxR2OCQZN5p7ppJtHiIe2tDTb1ebFAKY/"

S3_BUCKET = "qcedelft"

AWS_REGION = "us-east-1"

TERMINATE_URL = "http://999-term.teliov.xyz/lasaksalkslasl"


class Logger(object):

    def __init__(self, logger_name):

        self.logger_name = logger_name
        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)

        self.logger.addHandler(self.handler)

    def log(self, message, level=logging.DEBUG, to_telegram=True):
        message = "%s: %s" % (self.logger_name, message)
        self.logger.log(level, message)

        if to_telegram:
            send_message_telegram(message)
        return True

    def to_string(self):
        self.handler.flush()

        return self.stream.getvalue()


def log(logger, message, level=logging.DEBUG, to_telegram=True):
    logger.log(level, message)

    if to_telegram:
        send_message_telegram(message)

    return True


def terminate_instance():
    url = "http://169.254.169.254/latest/meta-data/instance-id"
    r = requests.get(url)

    instance_id = r.content.decode("utf-8")
    if not instance_id:
        send_message_telegram("unable to get ec2 instance")
        return False
    else:
        params = {
            "instance_id": instance_id
        }
        try:
            requests.get(TERMINATE_URL, params=params)
            send_message_telegram("Terminated instance: " + instance_id)
            return True
        except Exception as e:
            error_message = e.__str__()
            message = "Failed to terminate instance: %s. i.v.m: %s" % (instance_id, error_message)
            send_message_telegram(message)
            return  False


def send_message_telegram(message):
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message
    }

    url = BASE_URL + "sendMessage"
    try:
        requests.get(url, params=payload);
    except Exception:
        pass

    return True


def race_txform(val):
    race_code = {'white': 0, 'black':1, 'asian':2, 'native':3, 'other':4}
    return race_code.get(val)


def label_txform(val, labels):
    return labels.get(val)


def symptom_transform(val, labels):
    parts = val.split(";")
    res = sum([labels.get(item) for item in parts])
    return res


def handle_bit_wise(val, comp):
    if val & comp > 0:
        return 1
    else:
        return 0


def s3_to_pandas(client, bucket, key):
    # get key using boto3 client
    obj = client.get_object(Bucket=bucket, Key=key)

    is_gzipped = key.split('.')[-1].lower() == 'gz'

    gz = obj['Body']

    if is_gzipped:
        gz = gzip.GzipFile(fileobj=obj['Body'])

    # load stream directly to DF
    return pd.read_csv(gz)


def pandas_to_s3(df, client, bucket, key):
    # write DF to string stream
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    # reset stream position
    csv_buffer.seek(0)
    # create binary stream
    gz_buffer = io.BytesIO()

    # compress string stream using gzip
    with gzip.GzipFile(mode='w', fileobj=gz_buffer) as gz_file:
        gz_file.write(bytes(csv_buffer.getvalue(), 'utf-8'))

    # write stream to S3
    obj = client.put_object(Bucket=bucket, Key=key, Body=gz_buffer.getvalue())
    return True
