#!/usr/bin/bash

python -m pip install --user virtualenv

python -m virtualenv va03_env

source ./va03_env/Scripts/activate

pip install --upgrade pip
pip install -r requirements.txt

deactivate