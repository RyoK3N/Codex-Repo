# Chess AI Project

This project demonstrates a simple reinforcement learning agent that learns to play chess through self‑play.

## Setup

1. Create a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

Run the training script to let the agent learn by self-play. A Plotly HTML
file can be generated to visualize rewards over time:

```bash
python -m chess_ai.train --episodes 1000 --checkpoint agent.pth --plot training.html
```

This will store a model checkpoint at `agent.pth`.

## Play against the AI

After training, you can play against the agent in the terminal or with a
graphical interface.

```bash
python -m chess_ai.play_ui --model agent.pth

```

For a graphical board using Pygame (with simple animations) run:

```bash
python -m chess_ai.pygame_ui --model agent.pth
```

Moves are entered in UCI notation (e.g., `e2e4`).

## Repository Structure

- `chess_ai/` – Package containing the environment, agent, and scripts
- `requirements.txt` – Project dependencies
- `README.md` – Project overview and instructions
