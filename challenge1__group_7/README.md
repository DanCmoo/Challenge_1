# MsPacman DQN — Atari Deep Q-Network with Stable-Baselines3

This repository provides a single, self-contained script to train, evaluate
and compare Deep Q-Network (DQN) agents on MsPacman supported by the
Arcade Learning Environment (ALE). The implementation uses Stable-Baselines3
and the Gymnasium Atari wrappers to apply the standard preprocessing pipeline
and frame-stacking used in classic DQN experiments.

Project files provided:

- `mspacman_dqn.py` — Main script implementing train, play, sweep and
  inspect modes.
- `sweep_configs_mspacman.json` — JSON array of ten named hyperparameter experiments
  used by the sweep runner.
- `pyproject.toml` — Project metadata and dependencies.

Generated at runtime:

- `models/` — Saved model archives (`.zip`).
- `logs/mspacman_dqn/` — TensorBoard event files for monitoring training.

Prerequisites

The code targets Python 3.11 and relies on a standard scientific Python
tooling stack.

Installation (Linux)

Create and activate a Python 3.11 virtual environment, then install dependencies with Poetry:

```bash
pyenv virtualenv 3.11.10 ml_old
pyenv activate ml_old
poetry env use $(pyenv which python)
poetry install
```

Option 1: using `poetry shell`

```bash
poetry shell
python mspacman_dqn.py --mode train --model-path models/mspacman_exp15
```

Option 2: without `poetry shell`

```bash
poetry run python mspacman_dqn.py --mode train --model-path models/mspacman_exp15
```

Core usage

All commands must be executed from the directory containing
`mspacman_dqn.py`.

If you are not inside `poetry shell`, prepend each command with `poetry run`.

- Train best model (saves to `models/mspacman_exp15.zip`):

```bash
python mspacman_dqn.py --mode train --model-path models/mspacman_exp15
```

- Train model 2 using a named experiment from the sweep file:

```bash
python mspacman_dqn.py --mode train --experiment exp_15_lr_high \
  --model-path models/mspacman_exp15
```

- Watch a saved model play (requires a display):

```bash
python mspacman_dqn.py --mode play --model-path models/mspacman_exp15 --episodes 3
```

- Inspect the saved model constructor parameters:

```bash
python mspacman_dqn.py --mode inspect --model-path models/mspacman_exp15
```

Monitoring

Start TensorBoard to view training metrics and compare sweep runs:

```bash
tensorboard --logdir=logs --port 6006
```

Then open http://localhost:6006 in your browser.

Key scalars emitted by the script:

- `rollout/ep_rew_mean`: Rolling mean episode reward (last 100 episodes).
- `training/episode_reward`: Per-episode total reward.
- `training/epsilon`: Exploration (ε) scheduled over time.
- `train/loss`: Temporal-difference loss during learning.

Link to the video: https://udistritaleduco-my.sharepoint.com/:v:/g/personal/jsrodriguezc_udistrital_edu_co/IQAwdpCSX1mVT6L7lgqm7WVqAZ30X2xQFI7EPc455PM14Fo?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=cl0Sih

Timestaps:
0:00 – 1:45: Explanation of hyperparameters and experiments
1:45 – 3:05: TensorBoard results and justification of the chosen model
3:05 – 6:10: Agent behavior and performance outcomes


# Team Members - Student ID

1. Juan Diego Lozano Luna - 20231020040  
2. Daniel Esteban Camacho Ospina - 20231020046  
3. Juan Sebastian Rodriguez Carreño - 20231020107