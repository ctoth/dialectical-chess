"""Forks heuristic label/evidence producers."""

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


def fork_witness_labels(
    witness: Any,
    gives_check: bool,
) -> tuple[tuple[str, ...], tuple[ArgumentEvidence, ...], tuple[str, ...], tuple[ArgumentEvidence, ...], int]:
    target_label = f"smt:fork:targets:{witness.target_count}:value:{witness.target_value}"
    labels = [
        target_label,
        f"smt:fork:piece:{witness.piece}",
        f"smt:fork:net:{witness.net_value}",
    ]
    if gives_check:
        labels.append("smt:fork:gives_check")
    reason_evidence = [
        support(
            target_label,
            world=EvidenceWorld.SMT,
            counts_as_tactical=True,
            argument_value="tactical",
            strength=4,
            support_magnitude=witness.target_value,
            support_kind=SupportKind.SMT_FORK,
        ),
        *(display_evidence(label, world=EvidenceWorld.SMT) for label in labels[1:]),
    ]
    if witness.piece in {"q", "r"} and not gives_check:
        objection_label = f"smt:fork:high_value_piece:{witness.piece}"
        return (
            tuple(labels),
            tuple(reason_evidence),
            (objection_label,),
            (
                objection(
                    objection_label,
                    kind=ObjectionKind.SMT_FORK_HIGH_VALUE,
                    strength=3,
                    world=EvidenceWorld.SMT,
                    argument_value="tactical",
                ),
            ),
            0,
        )
    if witness.moved_piece_en_pris_value:
        objection_label = f"smt:fork:moved_piece_en_pris:{witness.moved_piece_en_pris_value}"
        objections = (objection_label,)
        objection_evidence = (
            objection(
                objection_label,
                kind=ObjectionKind.SMT_FORK_MOVED_PIECE_EN_PRIS,
                strength=3,
                world=EvidenceWorld.SMT,
                moved_piece_en_pris_value=witness.moved_piece_en_pris_value,
                argument_value="tactical",
            ),
        )
        if witness.net_value <= 0 and not gives_check:
            return tuple(labels), tuple(reason_evidence), objections, objection_evidence, 0
    compatibility = f"smt:fork:{witness.target_count}:{witness.target_value}"
    return (
        (compatibility, *labels),
        (
            support(
                compatibility,
                world=EvidenceWorld.SMT,
                counts_as_tactical=True,
                argument_value="tactical",
                strength=4,
            ),
            *reason_evidence,
        ),
        (),
        (),
        max(0, witness.net_value),
    )
