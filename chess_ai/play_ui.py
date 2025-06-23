import argparse
import chess
from .chess_env import ChessEnv
from .rl_agent import PolicyNetwork, select_move
import torch


def play(model_path: str):
    env = ChessEnv()
    policy = PolicyNetwork()
    policy.load_state_dict(torch.load(model_path))
    state = env.reset()
    turn_white = True
    while True:
        if turn_white:
            print(env.board)
            move = input("Your move: ")
            try:
                state, reward, done = env.step(move)
            except Exception as e:
                print(e)
                continue
        else:
            move = select_move(policy, state, env.legal_moves)
            print(f"AI move: {move}")
            state, reward, done = env.step(move)
        if done:
            print(env.board)
            print("Game over:", env.board.result())
            break
        turn_white = not turn_white


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    play(args.model)
