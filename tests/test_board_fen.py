from __future__ import annotations

import chess
import pytest

from dialectical_chess.board import START_FEN, OwnedBoard, OwnedMove


def ep_field(fen: str) -> str:
    return fen.split()[3]


def apply_moves(board: OwnedBoard, moves: tuple[str, ...]) -> OwnedBoard:
    for move in moves:
        board = board.apply(OwnedMove.from_uci(move))
    return board


def oracle_after(fen: str, moves: tuple[str, ...]) -> chess.Board:
    board = chess.Board(fen)
    for move in moves:
        board.push(chess.Move.from_uci(move))
    return board


@pytest.mark.parametrize(
    ("fen", "moves"),
    [
        (START_FEN, ("b2b3", "h7h5")),
        (START_FEN, ("e2e4",)),
        (START_FEN, ("e2e4", "d7d5")),
        ("4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1", ("e2e4",)),
    ],
)
def test_double_pawn_push_en_passant_field_matches_python_chess(
    fen: str,
    moves: tuple[str, ...],
) -> None:
    owned = apply_moves(OwnedBoard.from_fen(fen), moves)
    oracle = oracle_after(fen, moves)

    assert ep_field(owned.fen()) == ep_field(oracle.fen())
