# Chess AI Project

This project demonstrates a simple reinforcement learning agent that learns to play chess through self‑play.  An interactive Pygame board allows you to play against the agent and watch its training progress with Plotly charts.

## Setup

1. Create a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

Run the training script to let the agent learn by self-play:

```bash
python -m chess_ai.train --episodes 1000 --checkpoint agent.pth
```
This will store a model checkpoint at `agent.pth` and open a Plotly window
showing the episode rewards as training progresses.

## Play against the AI

After training, you can play against the agent using the Pygame interface:

```bash
python -m chess_ai.play_ui --model agent.pth
```
Use the mouse to select the source and destination squares.  The board updates
with simple animations as pieces move.  Set `PYGAME_HEADLESS=1` when running
on a server without a display.

## Repository Structure

- `chess_ai/` – Package containing the environment, agent, and scripts
- `requirements.txt` – Project dependencies
- `README.md` – Project overview and instructions
