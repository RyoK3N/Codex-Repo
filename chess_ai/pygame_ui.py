import argparse
import io
import pygame
import cairosvg
import torch
import chess
from .chess_env import ChessEnv
from .rl_agent import PolicyNetwork, select_move

SQUARE_SIZE = 60
BOARD_SIZE = SQUARE_SIZE * 8

PIECE_FILES = {
    'P': 'plt.svg', 'N': 'nlt.svg', 'B': 'blt.svg', 'R': 'rlt.svg', 'Q': 'qlt.svg', 'K': 'klt.svg',
    'p': 'pdt.svg', 'n': 'ndt.svg', 'b': 'bdt.svg', 'r': 'rdt.svg', 'q': 'qdt.svg', 'k': 'kdt.svg'
}


def load_piece_images():
    images = {}
    for key, fname in PIECE_FILES.items():
        with open(f"assets/svg/{fname}", "rb") as f:
            png_bytes = cairosvg.svg2png(file_obj=f)
        img = pygame.image.load(io.BytesIO(png_bytes))
        img = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))
        images[key] = img
    return images


def draw_board(screen, board, images, animation=None):
    colors = [(240, 217, 181), (181, 136, 99)]
    for rank in range(8):
        for file in range(8):
            color = colors[(rank + file) % 2]
            rect = pygame.Rect(file * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            key = piece.symbol()
            img = images[key]
            pos = (col * SQUARE_SIZE, row * SQUARE_SIZE)
            if animation and animation['square'] == square:
                pos = animation['pos']
            screen.blit(img, pos)


def animate_move(screen, board, images, move):
    start = chess.square_file(move.from_square) * SQUARE_SIZE, (7 - chess.square_rank(move.from_square)) * SQUARE_SIZE
    end = chess.square_file(move.to_square) * SQUARE_SIZE, (7 - chess.square_rank(move.to_square)) * SQUARE_SIZE
    frames = 10
    piece = board.piece_at(move.from_square)
    if not piece:
        return
    for i in range(1, frames + 1):
        interp = (i / frames)
        x = start[0] + (end[0] - start[0]) * interp
        y = start[1] + (end[1] - start[1]) * interp
        screen.fill((0, 0, 0))
        draw_board(screen, board, images, {'square': move.from_square, 'pos': (x, y)})
        pygame.display.flip()
        pygame.time.delay(30)


def play(model_path: str):
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
    pygame.display.set_caption("Chess AI")
    images = load_piece_images()

    env = ChessEnv()
    policy = PolicyNetwork()
    policy.load_state_dict(torch.load(model_path))
    state = env.reset()
    running = True
    selected = None
    turn_white = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and turn_white:
                x, y = event.pos
                col = x // SQUARE_SIZE
                row = 7 - (y // SQUARE_SIZE)
                square = chess.square(col, row)
                if selected is None:
                    if env.board.piece_at(square) and env.board.piece_at(square).color == chess.WHITE:
                        selected = square
                else:
                    move = chess.Move(selected, square)
                    if move in env.board.legal_moves:
                        animate_move(screen, env.board, images, move)
                        state, reward, done = env.step(move.uci())
                        selected = None
                        turn_white = False
        if not turn_white:
            move = select_move(policy, state, env.legal_moves)
            m = chess.Move.from_uci(move)
            animate_move(screen, env.board, images, m)
            state, reward, done = env.step(move)
            turn_white = True
        screen.fill((0, 0, 0))
        draw_board(screen, env.board, images)
        pygame.display.flip()
        if env.board.is_game_over():
            print("Game over:", env.board.result())
            running = False
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    play(args.model)
