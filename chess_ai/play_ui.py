import argparse
import chess

from .chess_env import ChessEnv
from .rl_agent import RLAgent


def main():
    parser = argparse.ArgumentParser(description="Play against the trained agent")
    parser.add_argument("--model", type=str, default="agent.pth", help="Path to trained model")
    args = parser.parse_args()

    env = ChessEnv()
    agent = RLAgent()
    try:
        agent.load(args.model)
    except FileNotFoundError:
        print("Model not found, playing with untrained agent.")

    while not env.board.is_game_over():
        print(env.board)
        if env.board.turn == chess.WHITE:
            move = input("Your move (UCI): ")
        else:
            move = agent.choose_move(env.board, epsilon=0.0)
            print(f"AI move: {move}")
        try:
            env.step(move)
        except ValueError:
            print("Illegal move. Try again.")
    print(env.board)
    print("Game over:", env.board.result())


if __name__ == "__main__":
    main()
