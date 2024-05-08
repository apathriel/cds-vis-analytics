#!/usr/bin/bash

python -m pip install --user virtualenv
python -m virtualenv va04_env

source ./va04_env/Scripts/activate

pip install --upgrade pip
pip install -r requirements.txt

deactivate