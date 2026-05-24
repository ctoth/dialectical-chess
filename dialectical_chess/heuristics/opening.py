"""Opening heuristic label/evidence producers."""

from __future__ import annotations

from typing import Any

from dialectical_chess.board import (
    OwnedBoard,
    file_of,
    opposite,
    piece_color,
    rank_of,
    square_index,
    square_name,
)
from dialectical_chess.evidence import (
    ArgumentEvidence,
    DefeaterKind,
    EvidenceWorld,
    ObjectionKind,
    SupportKind,
    defeater_evidence,
    defeater_strength,
    material_cost_objection_strength,
)
from dialectical_chess.heuristics.evidence import (
    EvidenceLabels,
    defended_piece_support_strength,
    display_evidence,
    material_support_strength,
    objection,
    support,
)
from dialectical_chess.search import OWNED_PIECE_VALUE, owned_is_threefold_repetition
from dialectical_chess.smt import moved_piece_attacks_square
from dialectical_chess.tuning import (
    KING_ESCAPE_SCORE,
    MAJOR_PIECE_VALUE,
    MOVED_PIECE_DEFENDED_SCORE,
    QUEEN_BLUNDER_EXCHANGE_THRESHOLD,
    UNSUPPORTED_MAJOR_DRIFT_PENALTY,
)


def opening_development_objections(
    board: OwnedBoard,
    move: Any,
    *,
    captured_value: int,
    gives_check: bool,
) -> EvidenceLabels:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return EvidenceLabels(())
    color = piece_color(piece)
    kind = piece.lower()
    undeveloped_minors = undeveloped_minor_count(board, color)
    if (
        kind in {"n", "b"}
        and gives_check
        and captured_value == 0
        and board.fullmove_number <= 10
        and undeveloped_minors >= 2
    ):
        label = f"opening:premature_minor_check:{move.uci()}:undeveloped_minors:{undeveloped_minors}"
        return (
            EvidenceLabels(
                (label,),
                (
                    objection(
                        label,
                        kind=ObjectionKind.OPENING_PREMATURE_MINOR_CHECK,
                        strength=1,
                        objection_magnitude=undeveloped_minors,
                    ),
                ),
                -900,
            )
        )
    if kind not in {"q", "r"}:
        return EvidenceLabels(())
    if captured_value >= OWNED_PIECE_VALUE["n"]:
        return EvidenceLabels(())
    if kind == "r" and captured_value == 0 and board.fullmove_number <= 20:
        label = f"opening:premature_rook:{move.uci()}:undeveloped_minors:{undeveloped_minors}"
        return EvidenceLabels(
            (label,),
            (
                objection(
                    label,
                    kind=ObjectionKind.OPENING_PREMATURE_ROOK,
                    strength=1,
                    objection_magnitude=undeveloped_minors,
                ),
            ),
            -250,
        )
    if kind != "q" or board.fullmove_number > 10 or undeveloped_minors < 2:
        return EvidenceLabels(())
    label = f"opening:premature_queen:{move.uci()}:undeveloped_minors:{undeveloped_minors}"
    return EvidenceLabels(
        (label,),
        (
            objection(
                label,
                kind=ObjectionKind.OPENING_PREMATURE_QUEEN,
                strength=1,
                objection_magnitude=undeveloped_minors,
            ),
        ),
        -1_200,
    )

def undeveloped_minor_count(board: OwnedBoard, color: str) -> int:
    home_squares = (
        ("b1", "g1", "c1", "f1")
        if color == "w"
        else ("b8", "g8", "c8", "f8")
    )
    expected = ("N", "N", "B", "B") if color == "w" else ("n", "n", "b", "b")
    return sum(
        1
        for square, piece in zip(home_squares, expected, strict=True)
        if board.piece_at(square) == piece
    )

def opening_minor_retreat_objections(
    board: OwnedBoard,
    move: Any,
    *,
    captured_value: int,
    gives_check: bool,
) -> EvidenceLabels:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.lower() not in {"n", "b"}:
        return EvidenceLabels(())
    if board.fullmove_number > 20 or captured_value > 0 or gives_check:
        return EvidenceLabels(())
    color = piece_color(piece)
    if is_minor_home_square(move.from_square, piece):
        return EvidenceLabels(())
    to_rank = rank_of(move.to_square)
    retreats_to_home_ranks = to_rank <= 1 if color == "w" else to_rank >= 6
    if not retreats_to_home_ranks:
        return EvidenceLabels(())
    label = f"opening:minor_retreat:{move.uci()}"
    return EvidenceLabels(
        (label,),
        (
            objection(
                label,
                kind=ObjectionKind.OPENING_MINOR_RETREAT,
                strength=1,
            ),
        ),
        -400,
    )

def is_minor_home_square(square: int, piece: str) -> bool:
    if piece == "N":
        return square in {square_index("b1"), square_index("g1")}
    if piece == "B":
        return square in {square_index("c1"), square_index("f1")}
    if piece == "n":
        return square in {square_index("b8"), square_index("g8")}
    if piece == "b":
        return square in {square_index("c8"), square_index("f8")}
    return False

def opening_king_safety_objections(
    board: OwnedBoard,
    move: Any,
    *,
    captured_value: int = 0,
) -> EvidenceLabels:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.lower() != "k":
        return EvidenceLabels(())
    if move.kind == "castle" or board.fullmove_number > 20:
        return EvidenceLabels(())
    color = piece_color(piece)
    if board.in_check(color):
        if captured_value == 0 and not king_stays_on_home_rank(color, move.to_square):
            label = f"opening:king_center_flight:{move.uci()}"
            return EvidenceLabels(
                (label,),
                (
                    objection(
                        label,
                        kind=ObjectionKind.OPENING_KING_CENTER_FLIGHT,
                        strength=1,
                    ),
                ),
                -400,
            )
        return EvidenceLabels(())
    label = f"opening:king_walk:{move.uci()}"
    return EvidenceLabels(
        (label,),
        (
            objection(
                label,
                kind=ObjectionKind.OPENING_KING_WALK,
                strength=1,
            ),
        ),
        -400,
    )

def king_stays_on_home_rank(color: str, square: int) -> bool:
    return rank_of(square) == (0 if color == "w" else 7)
