import argparse

from .rl_agent import RLAgent


def main():
    parser = argparse.ArgumentParser(description="Train RL chess agent")
    parser.add_argument("--episodes", type=int, default=10, help="Number of self-play episodes")
    parser.add_argument("--checkpoint", type=str, default="agent.pth", help="Path to save checkpoint")
    parser.add_argument("--load", type=str, help="Optional model path to continue training")
    args = parser.parse_args()

    agent = RLAgent()
    if args.load:
        agent.load(args.load)
    agent.train_self_play(args.episodes)
    agent.save(args.checkpoint)


if __name__ == "__main__":
    main()
