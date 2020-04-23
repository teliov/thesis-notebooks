#! /usr/bin/env python

import sys
import io
import os
import json
import base64
import random,string
import subprocess


def random_word(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

symptom_file = sys.argv[1]
run_file = sys.argv[1]

stream = io.StringIO()

stream.write("#! /usr/bin/env bash\n\n")
stream.write("SYMPTOM_FILE=%s\n" % symptom_file)
stream.write("RUN_NAME=%s\n" % run_file)

with open("provision_parser.tpl") as fp:
    contents = fp.read()

stream.write(contents + "\n")

string_contents = stream.getvalue()
stream.close()

op_directory = os.path.join(os.getcwd(), "outputs")
if not os.path.isdir(op_directory):
    os.mkdir(op_directory)


provision_script = os.path.join(op_directory, "%s.sh" % random_word(12))
with open(provision_script, "w") as fp:
    fp.write(string_contents)

user_data = base64.b64encode(bytes(string_contents, "utf-8"))
user_data = user_data.decode("utf-8")

with open('parse_specification.json') as fp:
    specs = json.load(fp)

specs["UserData"] = user_data

json_launch_file = os.path.join(op_directory, "%s.json" % random_word(12))

with open(json_launch_file, "w") as fp:
    json.dump(specs, fp)

commands = ["aws", "ec2", "request-spot-instances" "--launch-specification file://%s" % json_launch_file]
subprocess.run(commands, shell=True, stdout=subprocess.PIPE)