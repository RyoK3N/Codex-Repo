import chess
import numpy as np

class ChessEnv:
    """A minimal chess environment for self-play."""

    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self._get_state()

    def _get_state(self):
        # Simple board encoding: piece values on board
        board_state = np.zeros(64, dtype=np.int8)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = piece.piece_type if piece.color == chess.WHITE else -piece.piece_type
                board_state[square] = value
        return board_state

    def step(self, move_uci):
        move = chess.Move.from_uci(move_uci)
        if move not in self.board.legal_moves:
            raise ValueError("Illegal move")
        self.board.push(move)
        done = self.board.is_game_over()
        reward = 0.0
        if done:
            result = self.board.result()
            if result == "1-0":
                reward = 1.0
            elif result == "0-1":
                reward = -1.0
        return self._get_state(), reward, done

    @property
    def legal_moves(self):
        return [m.uci() for m in self.board.legal_moves]
