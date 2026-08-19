### Virtual Environment
To activate python's virtual environment run: `./create_venv.sh`

**NOTE**: if `python3` gives issues just use `python` wherever you see `python3` is being used.
# Cartpole
## Commands for running cartpole's experiments
`python3 run_experiments.py control` runs Q-learning and SARSA
`python3 run_experiments.py prediction` runs TD(0) and Monte-Carlo
`python3 run_experiments.py sensitivity` runs sensitivity analysis
`python3 run_experiments.py failure`

# Acrobot
## Training
`python3 -m acrobot.run_acrobot_experiment`

## Plots
`python3 -m acrobot.plot_acrobot_results`

## Sensitivity
`python3 -m acrobot.sensitivity`

## Grafici utilizzabili con i due run completi gia' disponibili.
`python3 -m acrobot.plot_acrobot_results --seeds 123 456`