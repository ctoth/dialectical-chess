"""Positional heuristic label/evidence producers."""

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


def positional_reason_labels(board: OwnedBoard, move: Any, child: OwnedBoard) -> EvidenceLabels:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return EvidenceLabels(())
    labels: list[str] = []
    evidence: list[ArgumentEvidence] = []
    move_text = move.uci()
    kind = piece.lower()
    color = piece_color(piece)
    from_rank = rank_of(move.from_square)
    to_rank = rank_of(move.to_square)

    if kind == "p" and file_of(move.from_square) in {3, 4} and abs(to_rank - from_rank) == 2:
        label = f"development:{move_text}:center_pawn"
        labels.append(label)
        evidence.append(support(label, world=EvidenceWorld.POSITIONAL, counts_as_positional=True, argument_value="positional", strength=1, support_kind=SupportKind.DEVELOPMENT))
    if kind in {"n", "b"} and from_rank == (0 if color == "w" else 7):
        label = f"development:{move_text}:minor_piece"
        labels.append(label)
        evidence.append(support(label, world=EvidenceWorld.POSITIONAL, counts_as_positional=True, argument_value="positional", strength=1, support_kind=SupportKind.DEVELOPMENT))
    if move.kind == "castle":
        label = f"king_safety:{move_text}:castle"
        labels.append(label)
        evidence.append(support(label, world=EvidenceWorld.POSITIONAL, counts_as_positional=True, argument_value="positional", strength=1))

    center_count = moved_piece_center_control(child, move.to_square, piece)
    if center_count:
        label = f"center_control:{move_text}:{center_count}"
        labels.append(label)
        evidence.append(support(label, world=EvidenceWorld.POSITIONAL, counts_as_positional=True, argument_value="positional", strength=1))
    activity_gain = moved_piece_activity_gain(board, child, move.from_square, move.to_square, piece)
    if activity_gain > 0:
        label = f"piece_activity:{move_text}:mobility_gain:{activity_gain}"
        labels.append(label)
        evidence.append(support(label, world=EvidenceWorld.POSITIONAL, counts_as_positional=True, argument_value="positional", strength=1))
    if kind == "p" and is_passed_pawn(child, move.to_square, color):
        label = f"pawn_structure:{move_text}:passed_pawn"
        labels.append(label)
        evidence.append(support(label, world=EvidenceWorld.POSITIONAL, counts_as_positional=True, argument_value="positional", strength=1))
    if kind in {"r", "q"} and controls_open_file(child, move.to_square):
        label = f"file_control:{move_text}:open_file"
        labels.append(label)
        evidence.append(support(label, world=EvidenceWorld.POSITIONAL, counts_as_positional=True, argument_value="positional", strength=1))
    if kind == "n" and is_supported_outpost(child, move.to_square, color):
        label = f"outpost:{move_text}:supported"
        labels.append(label)
        evidence.append(support(label, world=EvidenceWorld.POSITIONAL, counts_as_positional=True, argument_value="positional", strength=1))
    return EvidenceLabels(tuple(labels), tuple(evidence))

def moved_piece_center_control(board: OwnedBoard, source_square: int, piece: str) -> int:
    return sum(
        1
        for target in (
            square_index("d4"),
            square_index("e4"),
            square_index("d5"),
            square_index("e5"),
        )
        if moved_piece_attacks_square(board, source_square, target, piece)
    )

def controls_open_file(board: OwnedBoard, square: int) -> bool:
    file_index = file_of(square)
    return all(
        piece is None or piece.lower() != "p"
        for index, piece in enumerate(board.squares)
        if file_of(index) == file_index
    )

def moved_piece_activity_gain(
    before: OwnedBoard,
    after: OwnedBoard,
    from_square: int,
    to_square: int,
    piece: str,
) -> int:
    before_activity = moved_piece_activity(before, from_square, piece)
    after_activity = moved_piece_activity(after, to_square, piece)
    return after_activity - before_activity

def moved_piece_activity(board: OwnedBoard, square: int, piece: str) -> int:
    return sum(
        1
        for target in range(64)
        if target != square and moved_piece_attacks_square(board, square, target, piece)
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
        if 0 <= support_file < 8:
            support_square = support_rank * 8 + support_file
            if 0 <= support_square < 64 and board.piece_at(support_square) == support_piece:
                return True
    return False
