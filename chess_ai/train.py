import argparse
import torch
import torch.optim as optim
import chess
import torch.nn.functional as F
from .chess_env import ChessEnv
from .rl_agent import PolicyNetwork, select_move
import plotly.graph_objects as go


def self_play_episode(env, policy, optimizer, gamma=0.99):
    state = env.reset()
    log_probs = []
    rewards = []
    done = False
    while not done:
        legal = env.legal_moves
        move = select_move(policy, state, legal)
        logits = policy(torch.tensor(state, dtype=torch.float32))
        move_index = chess.SQUARE_NAMES.index(move[:2])
        log_prob = F.log_softmax(logits, dim=0)[move_index]
        state, reward, done = env.step(move)
        log_probs.append(log_prob)
        rewards.append(reward)
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    if returns.std() != 0:
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    loss = -torch.stack([lp * G for lp, G in zip(log_probs, returns)]).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return sum(rewards)


def train(episodes: int, checkpoint: str, plot_file: str | None = None):
    env = ChessEnv()
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    rewards = []
    for ep in range(episodes):
        ep_reward = self_play_episode(env, policy, optimizer)
        rewards.append(ep_reward)
        if (ep + 1) % 50 == 0:
            torch.save(policy.state_dict(), checkpoint)
            print(f"Episode {ep+1}: checkpoint saved")
    torch.save(policy.state_dict(), checkpoint)
    if plot_file:
        fig = go.Figure(data=go.Scatter(y=rewards))
        fig.update_layout(title="Episode Rewards", xaxis_title="Episode", yaxis_title="Return")
        fig.write_html(plot_file)
        print(f"Training plot saved to {plot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default="agent.pth")
    parser.add_argument("--plot", type=str, help="Optional HTML output for training curve")
    args = parser.parse_args()
    train(args.episodes, args.checkpoint, args.plot)
