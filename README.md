# Chess AI Module

This repository contains a minimal chess reinforcement learning setup based on
`python-chess` and PyTorch. The code can train an agent through self-play and
lets you play against it from the command line.

## Installation

1. Ensure Python 3.11+ is available.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

The requirements file installs a CPU only build of PyTorch and `python-chess`.

## Training

Use `train.py` to run self‑play training. The script saves a checkpoint after
training.

```bash
python -m chess_ai.train --episodes 100 --checkpoint agent.pth
```

Arguments:

- `--episodes` – number of self-play games to run.
- `--checkpoint` – file where the model will be saved.
- `--load` – optional path to an existing model to continue training.

## Playing Against the Agent

Once trained, you can play against the agent using `play_ui.py`:

```bash
python -m chess_ai.play_ui --model agent.pth
```

The UI prints the board in ASCII. When it is your move (always White), enter
moves in UCI notation (e.g. `e2e4`). The agent responds automatically until the
game is finished.

## File Overview

- `chess_ai/chess_env.py` – thin wrapper around `python-chess` providing board
  state and legal move handling.
- `chess_ai/rl_agent.py` – simple value-based reinforcement learning agent
  implemented with PyTorch.
- `chess_ai/train.py` – command line interface to train the agent through
  self-play.
- `chess_ai/play_ui.py` – minimal text interface to play against the trained
  agent.

The implementation is intentionally compact and meant for experimentation rather
than competitive play.
