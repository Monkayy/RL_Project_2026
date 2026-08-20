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
`python3 -m acrobot.sensitivity` (note that this takes a long time, on my laptop it took almost 6h)

# Breakout / Deep RL
`python3 -m breakout.run_ablation --only full --steps 50000` to see if everything works fine, not to be used as a reference
`python3 -m breakout.run_ablation --steps 250000 --seed 42` the actual run, without a CUDA environment this could take a while so take a seat
`python3 -m breakout.plot_results --seed 42`
