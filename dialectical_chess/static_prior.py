"""Disjoint static prior for Phase-2 opinion-valued move arguments."""

from __future__ import annotations

import math

from dialectical_chess.arguments import MoveProbe
from dialectical_chess.board import (
    OwnedBoard,
    file_of,
    opposite,
    piece_color,
    rank_of,
    square_index,
)
from dialectical_chess.search import OWNED_PIECE_VALUE

TAU_SCALE: float = 400.0
TAU_CLAMP: tuple[float, float] = (0.01, 0.99)


def static_prior(probe: MoveProbe) -> float:
    """Return a centipawn-scale prior from the post-move board only.

    This function deliberately ignores ``probe.score`` and every evidence label
    tuple. If a synthetic probe has no post-move board snapshot, there is no
    disjoint board state to read, so the honest residual prior is 0.
    """
    if probe.post_fen is None:
        return 0.0
    board = OwnedBoard.from_fen(probe.post_fen)
    mover = opposite(board.turn)
    return float(
        material_balance_for(board, mover)
        + positional_geometry(board, mover)
        - positional_geometry(board, board.turn)
    )


def squash(prior: float) -> float:
    raw = 0.5 + 0.5 * math.tanh(prior / TAU_SCALE)
    lo, hi = TAU_CLAMP
    return max(lo, min(hi, raw))


def material_balance_for(board: OwnedBoard, color: str) -> int:
    total = 0
    for piece in board.squares:
        if piece is None:
            continue
        value = OWNED_PIECE_VALUE[piece.lower()]
        total += value if piece_color(piece) == color else -value
    return total


def positional_geometry(board: OwnedBoard, color: str) -> int:
    return (
        central_pawn_presence(board, color)
        + minor_development(board, color)
        + open_file_pressure(board, color)
        + supported_outposts(board, color)
        + passed_pawn_score(board, color)
        + king_safety_geometry(board, color)
    )


def central_pawn_presence(board: OwnedBoard, color: str) -> int:
    pawn = "P" if color == "w" else "p"
    return 8 * sum(
        1
        for square in (square_index("d4"), square_index("e4"), square_index("d5"), square_index("e5"))
        if board.piece_at(square) == pawn
    )


def minor_development(board: OwnedBoard, color: str) -> int:
    home_rank = 0 if color == "w" else 7
    return 10 * sum(
        1
        for square, piece in enumerate(board.squares)
        if piece is not None
        and piece_color(piece) == color
        and piece.lower() in {"n", "b"}
        and rank_of(square) != home_rank
    )


def open_file_pressure(board: OwnedBoard, color: str) -> int:
    return 8 * sum(
        1
        for square, piece in enumerate(board.squares)
        if piece is not None
        and piece_color(piece) == color
        and piece.lower() in {"r", "q"}
        and controls_open_file(board, square)
    )


def controls_open_file(board: OwnedBoard, square: int) -> bool:
    file_index = file_of(square)
    return all(
        piece is None or piece.lower() != "p"
        for index, piece in enumerate(board.squares)
        if file_of(index) == file_index
    )


def supported_outposts(board: OwnedBoard, color: str) -> int:
    return 12 * sum(
        1
        for square, piece in enumerate(board.squares)
        if piece is not None
        and piece_color(piece) == color
        and piece.lower() == "n"
        and is_supported_outpost(board, square, color)
    )


def is_supported_outpost(board: OwnedBoard, square: int, color: str) -> bool:
    rank = rank_of(square)
    if color == "w" and rank < 3:
        return False
    if color == "b" and rank > 4:
        return False
    support_rank = rank - 1 if color == "w" else rank + 1
    support_piece = "P" if color == "w" else "p"
    for file_delta in (-1, 1):
        support_file = file_of(square) + file_delta
        if not 0 <= support_file < 8:
            continue
        support_square = support_rank * 8 + support_file
        if 0 <= support_square < 64 and board.piece_at(support_square) == support_piece:
            return True
    return False


def passed_pawn_score(board: OwnedBoard, color: str) -> int:
    pawn = "P" if color == "w" else "p"
    return 12 * sum(
        1
        for square, piece in enumerate(board.squares)
        if piece == pawn and is_passed_pawn(board, square, color)
    )


def is_passed_pawn(board: OwnedBoard, square: int, color: str) -> bool:
    opponent_pawn = "p" if color == "w" else "P"
    start_rank = rank_of(square) + (1 if color == "w" else -1)
    stop_rank = 8 if color == "w" else -1
    step = 1 if color == "w" else -1
    for file_index in range(max(0, file_of(square) - 1), min(7, file_of(square) + 1) + 1):
        for rank_index in range(start_rank, stop_rank, step):
            if board.piece_at(rank_index * 8 + file_index) == opponent_pawn:
                return False
    return True


def king_safety_geometry(board: OwnedBoard, color: str) -> int:
    king_square = board.king_square(color)
    score = 0
    if king_square in castled_king_squares(color):
        score += 20
    score += 6 * pawn_shield_count(board, color, king_square)
    return score


def castled_king_squares(color: str) -> frozenset[int]:
    if color == "w":
        return frozenset({square_index("g1"), square_index("c1")})
    return frozenset({square_index("g8"), square_index("c8")})


def pawn_shield_count(board: OwnedBoard, color: str, king_square: int) -> int:
    pawn = "P" if color == "w" else "p"
    shield_rank = rank_of(king_square) + (1 if color == "w" else -1)
    if not 0 <= shield_rank < 8:
        return 0
    king_file = file_of(king_square)
    count = 0
    for shield_file in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
        if board.piece_at(shield_rank * 8 + shield_file) == pawn:
            count += 1
    return count
