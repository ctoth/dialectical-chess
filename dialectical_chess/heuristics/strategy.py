"""Strategy heuristic label/evidence producers."""

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


def unsupported_major_drift_objections(
    board: OwnedBoard,
    move: Any,
    *,
    captured_value: int,
    gives_check: bool,
    reason_evidence: list[ArgumentEvidence],
) -> EvidenceLabels:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.lower() != "q":
        return EvidenceLabels(())
    if board.fullmove_number > 35 or captured_value > 0 or gives_check:
        return EvidenceLabels(())
    if any(reason.supports_argument and reason.counts_as_tactical for reason in reason_evidence):
        return EvidenceLabels(())
    label = f"strategy:unsupported_major_drift:{move.uci()}"
    return EvidenceLabels(
        (label,),
        (
            objection(
                label,
                kind=ObjectionKind.UNSUPPORTED_MAJOR_DRIFT,
                strength=1,
            ),
        ),
        UNSUPPORTED_MAJOR_DRIFT_PENALTY,
    )

def draw_objections(
    move: Any,
    *,
    child: OwnedBoard,
    position_history: tuple[str, ...],
) -> EvidenceLabels:
    labels: list[str] = []
    evidence: list[ArgumentEvidence] = []
    move_text = move.uci()
    if owned_is_threefold_repetition(child, position_history=position_history):
        label = f"strategy:threefold_repetition:{move_text}"
        labels.append(label)
        evidence.append(objection(label, kind=ObjectionKind.THREEFOLD_REPETITION))
    if child.is_fifty_move_draw():
        label = f"strategy:fifty_move_draw:{move_text}"
        labels.append(label)
        evidence.append(objection(label, kind=ObjectionKind.FIFTY_MOVE_DRAW))
    return EvidenceLabels(tuple(labels), tuple(evidence))
