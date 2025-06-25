import argparse
import os
import pygame
import cairosvg
import io
import chess
from .chess_env import ChessEnv
from .rl_agent import PolicyNetwork, select_move
import torch
import plotly.graph_objects as go

ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'svg')
SQUARE_SIZE = 64
BOARD_SIZE = SQUARE_SIZE * 8

# Preload piece surfaces from SVG icons
PIECE_IMAGES = {}
for filename in os.listdir(ASSET_DIR):
    if filename.endswith('.svg'):
        name = filename.split('.')[0]
        png_bytes = cairosvg.svg2png(url=os.path.join(ASSET_DIR, filename))
        PIECE_IMAGES[name] = pygame.image.load(io.BytesIO(png_bytes))
        PIECE_IMAGES[name] = pygame.transform.scale(PIECE_IMAGES[name], (SQUARE_SIZE, SQUARE_SIZE))

def draw_board(screen, board, selected=None, targets=None):
    """Draw the board, pieces and optional highlights."""
    colors = [pygame.Color("white"), pygame.Color("gray")]
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)
            color = colors[(rank + file) % 2]
            rect = pygame.Rect(file * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)

            # highlight selected square
            if selected is not None and square == selected:
                pygame.draw.rect(screen, pygame.Color("orange"), rect, 3)

            # highlight possible target squares
            if targets and square in targets:
                center = rect.center
                pygame.draw.circle(screen, pygame.Color("blue"), center, SQUARE_SIZE // 8)

            piece = board.piece_at(square)
            if piece:
                key = f"Chess_{piece.symbol().lower()}{'l' if piece.color == chess.WHITE else 'd'}t45"
                img = PIECE_IMAGES.get(key)
                if img:
                    screen.blit(img, rect)

def animate_move(screen, board, move):
    start = move.from_square
    end = move.to_square
    piece = board.piece_at(start)
    if not piece:
        return
    key = f"Chess_{piece.symbol().lower()}{'l' if piece.color == chess.WHITE else 'd'}t45"
    img = PIECE_IMAGES.get(key)
    if not img:
        return
    start_pos = (chess.square_file(start) * SQUARE_SIZE, (7 - chess.square_rank(start)) * SQUARE_SIZE)
    end_pos = (chess.square_file(end) * SQUARE_SIZE, (7 - chess.square_rank(end)) * SQUARE_SIZE)
    frames = 10
    for i in range(1, frames + 1):
        pygame.time.delay(30)
        draw_board(screen, board)
        interp = i / frames
        pos = (start_pos[0] + (end_pos[0] - start_pos[0]) * interp,
               start_pos[1] + (end_pos[1] - start_pos[1]) * interp)
        screen.blit(img, pos)
        pygame.display.flip()


def plot_board(board):
    fig = go.Figure()
    squares = []
    colors = []
    for rank in range(8):
        for file in range(8):
            squares.append((file, rank))
            colors.append('white' if (rank + file) % 2 == 0 else 'gray')
    x, y = zip(*squares)
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=SQUARE_SIZE, color=colors)))
    fig.update_layout(width=BOARD_SIZE, height=BOARD_SIZE, yaxis=dict(scaleanchor='x', autorange='reversed'))
    fig.show()


def play(model_path):
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
    pygame.display.set_caption('Chess AI')
    env = ChessEnv()
    policy = PolicyNetwork()
    policy.load_state_dict(torch.load(model_path))
    state = env.reset()
    turn_white = True
    selected = None
    running = True
    while running:
        # compute potential target squares for highlighting
        targets = []
        if selected is not None:
            for uci in env.legal_moves:
                mv = chess.Move.from_uci(uci)
                if mv.from_square == selected:
                    targets.append(mv.to_square)

        draw_board(screen, env.board, selected, targets)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and turn_white:
                x, y = event.pos
                file = x // SQUARE_SIZE
                rank = 7 - (y // SQUARE_SIZE)
                square = chess.square(file, rank)
                piece = env.board.piece_at(square)

                if selected is None:
                    if piece and piece.color == chess.WHITE:
                        selected = square
                else:
                    move = chess.Move(selected, square)
                    if move.uci() in env.legal_moves:
                        animate_move(screen, env.board, move)
                        state, reward, done = env.step(move.uci())
                        selected = None
                        turn_white = False
                        if done:
                            running = False
                    elif piece and piece.color == chess.WHITE:
                        selected = square
                    else:
                        selected = None

        if not turn_white and running:
            move_uci = select_move(policy, state, env.legal_moves)
            move = chess.Move.from_uci(move_uci)
            animate_move(screen, env.board, move)
            state, reward, done = env.step(move_uci)
            turn_white = True
            if done:
                running = False

        if not running:
            draw_board(screen, env.board)
            pygame.display.flip()
            plot_board(env.board)
    pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    play(args.model)

