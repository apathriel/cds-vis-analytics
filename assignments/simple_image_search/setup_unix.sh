#!/usr/bin/bash

python -m pip install --user virtualenv

python -m virtualenv env

source ./env/bin/activate

pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

deactivate