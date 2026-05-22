"""Opinion-valued Phase-2 move decider."""

from __future__ import annotations

from dataclasses import dataclass

import chess
from doxa import Opinion
from doxa.argumentation import evaluate

from dialectical_chess.arguments import MoveProbe
from dialectical_chess.evidence import (
    ObjectionKind,
    ObjectionEvidence,
    forced_mate_refutation_distance,
    has_search_refutation_at_most,
)
from dialectical_chess.loss_mining import has_forced_mate
from dialectical_chess.opinion_graph import (
    MoveArgumentationArtifacts,
    build_argumentation_artifacts,
)
from dialectical_chess.skeptical_filter import skeptical_survivors

SLOWEST_LOSS_MAX_MATE_DEPTH = 4


@dataclass(frozen=True)
class ArgumentationDecision:
    """The opinion-valued decision over one legal-move probe set."""

    selected: MoveProbe
    empty_survivors: bool
    move_opinion: dict[str, Opinion]


def choose_move_argumentation(probes: list[MoveProbe]) -> ArgumentationDecision:
    """Return the Phase-2 argumentation decision for the input probes."""
    if not probes:
        raise ValueError("position has no legal moves")
    artifacts: MoveArgumentationArtifacts = build_argumentation_artifacts(probes)
    opinions = evaluate(artifacts.graph.graph)
    survivors = skeptical_survivors(artifacts)
    empty_survivors = not survivors
    pool = survivors if survivors else {probe.uci for probe in probes}
    selected = max(
        (probe for probe in probes if probe.uci in pool),
        key=(
            (lambda probe: empty_survivors_selection_key(probe, artifacts, opinions))
            if empty_survivors
            else (lambda probe: expectation_selection_key(probe, artifacts, opinions))
        ),
    )
    return ArgumentationDecision(
        selected=selected,
        empty_survivors=empty_survivors,
        move_opinion={
            uci: opinions[argument]
            for uci, argument in artifacts.move_arg.items()
        },
    )


def expectation_selection_key(
    probe: MoveProbe,
    artifacts: MoveArgumentationArtifacts,
    opinions: dict[str, Opinion],
) -> tuple[float, float, str]:
    expectation = opinions[artifacts.move_arg[probe.uci]].expectation()
    return (
        expectation - material_safety_selection_penalty(probe),
        expectation,
        probe.uci,
    )


def empty_survivors_selection_key(
    probe: MoveProbe,
    artifacts: MoveArgumentationArtifacts,
    opinions: dict[str, Opinion],
) -> tuple[int, float, str]:
    return (
        slowest_loss_distance(probe),
        opinions[artifacts.move_arg[probe.uci]].expectation(),
        probe.uci,
    )


def slowest_loss_distance(probe: MoveProbe) -> int:
    if probe.post_fen is not None:
        board = chess.Board(probe.post_fen)
        for mate_depth in range(1, SLOWEST_LOSS_MAX_MATE_DEPTH + 1):
            if has_forced_mate(board, mate_depth=mate_depth):
                return mate_depth
    distances = [
        distance
        for evidence in (*probe.objection_evidence, *probe.reply_attack_evidence)
        if (distance := forced_mate_refutation_distance(evidence)) is not None
    ]
    return min(distances, default=0)


def material_safety_selection_penalty(probe: MoveProbe) -> float:
    if (
        has_search_refutation_at_most(probe.objection_evidence, -300)
        and has_moved_piece_en_pris_objection(probe)
        and has_ignored_hanging_piece_objection(probe)
    ):
        return 1.0
    if not has_search_refutation_at_most(probe.objection_evidence, -400):
        return 0.0
    for evidence in probe.objection_evidence:
        if not isinstance(evidence, ObjectionEvidence):
            continue
        if evidence.objection_kind == ObjectionKind.IGNORED_HANGING_PIECE:
            return 1.0
        if (
            evidence.objection_kind == ObjectionKind.MOVED_PIECE_EN_PRIS
            and evidence.moved_piece_en_pris_value is not None
            and evidence.moved_piece_en_pris_value >= 300
        ):
            return 1.0
    return 0.0


def has_moved_piece_en_pris_objection(probe: MoveProbe) -> bool:
    return any(
        isinstance(evidence, ObjectionEvidence)
        and evidence.objection_kind == ObjectionKind.MOVED_PIECE_EN_PRIS
        and evidence.moved_piece_en_pris_value is not None
        and evidence.moved_piece_en_pris_value >= 300
        for evidence in probe.objection_evidence
    )


def has_ignored_hanging_piece_objection(probe: MoveProbe) -> bool:
    return any(
        isinstance(evidence, ObjectionEvidence)
        and evidence.objection_kind == ObjectionKind.IGNORED_HANGING_PIECE
        for evidence in probe.objection_evidence
    )
