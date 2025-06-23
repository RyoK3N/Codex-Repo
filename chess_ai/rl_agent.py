import random
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD_SIZE = 64

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(BOARD_SIZE, 128)
        self.fc2 = nn.Linear(128, BOARD_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def select_move(policy_net: PolicyNetwork, state: np.ndarray, legal_moves):
    """Return a legal move in UCI format selected by the policy."""
    with torch.no_grad():
        logits = policy_net(torch.tensor(state, dtype=torch.float32))
        probs = F.softmax(logits, dim=0).numpy()

    move_scores = []
    for move in legal_moves:
        square = chess.SQUARE_NAMES.index(move[:2])
        move_scores.append(probs[square])
    if move_scores:
        return random.choices(legal_moves, weights=move_scores, k=1)[0]
    return random.choice(legal_moves)
