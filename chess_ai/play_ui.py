import argparse
import io
import os
import chess
import cairosvg
if os.environ.get("PYGAME_HEADLESS"):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
import torch
from .chess_env import ChessEnv
from .rl_agent import PolicyNetwork, select_move

SQUARE_SIZE = 80
BOARD_SIZE = SQUARE_SIZE * 8


def board_surface(board: chess.Board) -> pygame.Surface:
    svg_data = chess.svg.board(board).encode()
    png_bytes = cairosvg.svg2png(bytestring=svg_data)
    return pygame.image.load(io.BytesIO(png_bytes))


def play(model_path: str):
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
    pygame.display.set_caption("Chess AI")

    env = ChessEnv()
    policy = PolicyNetwork()
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    state = env.reset()

    selected_square = None
    turn_white = True
    clock = pygame.time.Clock()
    running = True
    while running:
        screen.blit(pygame.transform.smoothscale(board_surface(env.board), (BOARD_SIZE, BOARD_SIZE)), (0, 0))
        pygame.display.flip()
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif turn_white and event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col = x // SQUARE_SIZE
                row = 7 - (y // SQUARE_SIZE)
                square = chess.square(col, row)
                if selected_square is None:
                    selected_square = square
                else:
                    move = chess.Move(selected_square, square)
                    if move in env.board.legal_moves:
                        state, _, _ = env.step(move.uci())
                        turn_white = False
                    selected_square = None
        if not turn_white and running:
            move = select_move(policy, state, env.legal_moves)
            state, _, _ = env.step(move)
            turn_white = True
        if env.board.is_game_over():
            pygame.time.wait(2000)
            running = False
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    play(args.model)
