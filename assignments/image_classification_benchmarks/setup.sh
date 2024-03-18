python -m pip install --user virtualenv

python -m virtualenv a02_env

source ./a02_env/Scripts/activate

pip install --upgrade pip
pip install -r requirements.txt

deactivate