import chess
import numpy as np

class ChessEnv:
    """Simple wrapper around python-chess."""

    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self.get_state()

    def get_state(self):
        return self.board.fen()

    def legal_moves(self):
        return [move.uci() for move in self.board.legal_moves]

    def step(self, move_uci: str):
        move = chess.Move.from_uci(move_uci)
        if move not in self.board.legal_moves:
            raise ValueError("Illegal move")
        self.board.push(move)
        done = self.board.is_game_over()
        reward = 0.0
        if done:
            result = self.board.result()
            if result == '1-0':
                reward = 1.0
            elif result == '0-1':
                reward = -1.0
        return self.get_state(), reward, done, {}

    @staticmethod
    def board_tensor(board: chess.Board) -> np.ndarray:
        """Return board representation as (12, 8, 8) tensor."""
        mapping = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        for square, piece in board.piece_map().items():
            idx = mapping[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
            row = chess.square_rank(square)
            col = chess.square_file(square)
            tensor[idx, row, col] = 1
        return tensor
