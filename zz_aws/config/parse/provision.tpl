WORKDIR="/home/ubuntu/work"

sudo apt-get update

sudo apt-get -yq install python3 python3-venv python3-dev build-essential

su - ubuntu << EOUSER
if [[ ! -d $WORKDIR ]];
then
    mkdir $WORKDIR
fi

if [[ ! -d "/home/ubuntu/.ssh" ]];
then
    mkdir -p "/home/ubuntu/.ssh"
fi

if [[ ! -d "/home/ubuntu/.aws" ]];
then
    mkdir -p "/home/ubuntu/.aws"
fi

wget -O "/home/ubuntu/.ssh/id_rsa" "https://qcedelft.s3.amazonaws.com/config/qce-delft.pem"
chmod 600 "/home/ubuntu/.ssh/id_rsa"

wget -O "/home/ubuntu/.aws/config" "https://qcedelft.s3.amazonaws.com/config/config"
wget -O "/home/ubuntu/.aws/credentials" "https://qcedelft.s3.amazonaws.com/config/credentials"

# allow pull from git
echo "github.com,192.30.253.112 ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAq2A7hRGmdnm9tUDbO9IDSwBK6TbQa+PXYPCPy6rbTrTtw7PHkccKrpp0yVhp5HdEIcKr6pLlVDBfOLX9QUsyCOV0wzfjIJNlGEYsdlLJizHhbn2mUjvSAHQqZETYP81eFzLQNnPHt4EVVUh7VfDESU84KezmD5QlWpXLmvU31/yMf+Se8xhHTvKSCZIFImWwoG6mbUoWf9nzpIoaSjB+weqqUUmpaaasXVal72J+UX2B+2RPW3RcT0eOzQgqlJL3RKrTJvdsjE3JEAvGq3lGHSZXy28G3skua2SmVi/w4yCE6gbODqnTWlg7+wC604ydGXA8VJiS5ap43JXiUFFAaQ==" \
    | tee -a "/home/ubuntu/.ssh/known_hosts"


git clone git@github.com:teliov/thesis-notebooks.git "$WORKDIR/notebooks"

cd "$WORKDIR/notebooks/zz_aws"

python3 -m venv "$WORKDIR/notebooks/zz_aws/venv"

source $WORKDIR/notebooks/zz_aws/venv/bin/activate

pip install wheel
pip install -r "$WORKDIR/notebooks/zz_aws/requirements.txt"

python $WORKDIR/notebooks/zz_aws/main.py parse --symptoms_db symptoms/symptoms_db.json \
    --conditions_db symptoms/conditions_db.json \
    --file $SYMPTOM_FILE \
    --run $RUN_NAME
EOUSER
