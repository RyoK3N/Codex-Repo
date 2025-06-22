import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .chess_env import ChessEnv


def _state_tensor(board):
    return torch.from_numpy(ChessEnv.board_tensor(board))


class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


class RLAgent:
    def __init__(self, lr: float = 1e-3, gamma: float = 0.99, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ValueNetwork().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.memory: List[Tuple[torch.Tensor, float, torch.Tensor, bool]] = []
        self.loss_fn = nn.MSELoss()

    def predict(self, board) -> float:
        with torch.no_grad():
            tensor = _state_tensor(board).unsqueeze(0).to(self.device)
            value = self.model(tensor).item()
        return value

    def choose_move(self, board, epsilon: float = 0.1) -> str:
        moves = list(board.legal_moves)
        if random.random() < epsilon:
            return random.choice(moves).uci()
        best_move = None
        best_value = -float("inf")
        for m in moves:
            board.push(m)
            v = self.predict(board)
            board.pop()
            if v > best_value or best_move is None:
                best_value = v
                best_move = m
        return best_move.uci()

    def remember(self, state, reward, next_state, done):
        self.memory.append((state, reward, next_state, done))

    def train_step(self, batch_size: int = 32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        rewards = torch.tensor(rewards, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        target = rewards + self.gamma * self.model(next_states).squeeze() * (1 - dones)
        values = self.model(states).squeeze()
        loss = self.loss_fn(values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_self_play(self, episodes: int = 1, epsilon: float = 0.2, batch_size: int = 32):
        env = ChessEnv()
        for _ in range(episodes):
            env.reset()
            done = False
            while not done:
                state_tensor = _state_tensor(env.board)
                move = self.choose_move(env.board, epsilon)
                _, reward, done, _ = env.step(move)
                next_tensor = _state_tensor(env.board)
                self.remember(state_tensor, reward, next_tensor, done)
                self.train_step(batch_size)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
