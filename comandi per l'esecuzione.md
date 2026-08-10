# Virtual Environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Commands for running the experiments
python3 run_experiments.py control

python3 run_experiments.py prediction

python3 run_experiments.py sensitivity

python3 run_experiments.py failure