"""Reply heuristic label/evidence producers."""

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


def has_reply_mate_in_one_objection(objections: list[ArgumentEvidence]) -> bool:
    return any(
        objection.objection_kind == ObjectionKind.REPLY_MATE_IN_ONE
        for objection in objections
    )
