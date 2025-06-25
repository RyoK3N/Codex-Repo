# Chess AI Project

This project demonstrates a simple reinforcement learning agent that learns to play chess through self‑play.

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

This will store a model checkpoint at `agent.pth`.

## Play against the AI

After training, you can play against the agent either in the terminal or using
the graphical interface.

### Terminal UI

```bash
python -m chess_ai.play_ui --model agent.pth
```

Moves are entered in UCI notation (e.g., `e2e4`).

### Pygame UI

```bash
python -m chess_ai.play_pygame --model agent.pth
```

When a piece is selected, the square is highlighted and all legal target squares
are marked. Click a highlighted square to move the piece. The AI will then make
its move in response.

## Repository Structure

- `chess_ai/` – Package containing the environment, agent, and scripts
  - `assets/svg/` – SVG icons for chess pieces
  - `play_pygame.py` – interactive Pygame interface
  - `play_ui.py` – terminal interface
  - `train.py` – training entry point
- `requirements.txt` – Project dependencies
- `README.md` – Project overview and instructions
