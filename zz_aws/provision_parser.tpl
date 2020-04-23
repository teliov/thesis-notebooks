WORKDIR="/home/ubuntu/work"

sudo apt-get update

sudo apt-get -yq install python3 python3-env

if [[ ! -d $WORKDIR ]];
then
    mkdir $WORKDIR
fi

if [[ ! -d "/home/ubuntu/.ssh" ]];
then
    mkdir -p "/home/ubuntu/.ssh"
fi

wget -O "/home/ubuntu/.ssh/id_rsa" "https://qcedelft.s3.amazonaws.com/config/qce-delft.pem"
chmod 600 "/home/ubuntu/.ssh/id_rsa"

# allow pull from git
ssh-keyscan -H github.com >> "/home/ubuntu/.ssh/known_hosts"


git clone git@github.com:teliov/thesis-notebooks.git "$WORKDIR/notebooks"

cd "$WORKDIR/notebooks/zz_aws"

python3 -m venv "$WORKDIR/notebooks/zz_aws/venv"

source $WORKDIR/notebooks/zz_aws/venv/bin/activate

pip install -r "$WORKDIR/notebooks/zz_aws/requirements.txt"

python $WORKDIR/notebooks/zz_aws/main.py parse --symptoms_db symptoms/symptoms_db.json \
    --conditions_db symptoms/conditions_db.json \
    --file $SYMPTOM_FILE \
    --run $RUN_NAME