"""Opinion-valued move decider — the lexicographic FACT-then-graded key.

The decider is a single lexicographic key (design D2, modelled on
dialectical-checkers' decider): the FACT-tier term — the worst proven
material loss a move walks into — is ordered strictly before the graded term
— the move's opinion-valued ``expectation()``. A FACT decision always
dominates a graded one.

The FACT material-loss term is supplied by the chess
:mod:`~dialectical_chess.suppression` policy (``fact_material_loss``); this
module names no chess objection kind. Chess's ``material_safety`` penalties
used to be smuggled into the opinion base-rate and a flat argmax penalty;
this decider consults the loss as an honest, ordered FACT key term instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
from doxa import Opinion
from doxa.argumentation import evaluate

from dialectical_chess.arguments import MoveProbe
from dialectical_chess.evidence import forced_mate_refutation_distance
from dialectical_chess.loss_mining import has_forced_mate
from dialectical_chess.opinion_graph import (
    MoveArgumentationArtifacts,
    build_argumentation_artifacts,
)
from dialectical_chess.skeptical_filter import skeptical_survivors
from dialectical_chess.suppression import fact_material_loss

SLOWEST_LOSS_MAX_MATE_DEPTH = 4


@dataclass(frozen=True)
class ArgumentationDecision:
    """The opinion-valued decision over one legal-move probe set."""

    selected: MoveProbe
    empty_survivors: bool
    move_opinion: dict[str, Opinion]


def choose_move_argumentation(probes: list[MoveProbe]) -> ArgumentationDecision:
    """Return the argumentation decision for the input probes.

    Builds the opinion graph and the Dung filter, takes the grounded crisp
    survivors (or — when every move is hard-refuted — falls back to all
    moves), and picks the survivor maximising the lexicographic selection key.
    """
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
) -> tuple[int, float, str]:
    """The lexicographic selection key for a crisp survivor (design D2).

    The key is consumed by ``max`` — larger is better. Its terms, in order:

    1. the FACT term — the negated worst proven material loss
       (:func:`~dialectical_chess.suppression.fact_material_loss`). A move
       with no proven loss scores 0 here and outranks every move that walks
       into one; among moves that do, the smaller loss outranks the larger.
       This term is the FACT-tier prefix of the key: it dominates the graded
       term completely (design D2 — fact-as-highest-value).
    2. the graded term — the move's opinion-valued ``expectation()`` over the
       crisp survivors.
    3. the deterministic tiebreak — the move UCI (the lexicographically
       largest UCI wins an exact tie).
    """
    expectation = opinions[artifacts.move_arg[probe.uci]].expectation()
    return (
        -fact_material_loss(probe),
        expectation,
        probe.uci,
    )


def empty_survivors_selection_key(
    probe: MoveProbe,
    artifacts: MoveArgumentationArtifacts,
    opinions: dict[str, Opinion],
) -> tuple[int, float, str]:
    """The selection key for the empty-survivor fallback (design v2 §5d).

    When every legal move is hard-refuted there is no clean choice; the
    decider picks the least-bad move — the slowest proven loss, then the
    highest graded ``expectation()``, then the largest UCI.
    """
    return (
        slowest_loss_distance(probe),
        opinions[artifacts.move_arg[probe.uci]].expectation(),
        probe.uci,
    )


def slowest_loss_distance(probe: MoveProbe) -> int:
    """Return the proven mate distance a hard-refuted move walks into.

    A larger distance is a slower loss — better, under the empty-survivor
    fallback. Reads the post-move board (when the probe carries one) to scan
    for a forced mate, falling back to the forced-mate distance encoded in the
    move's objection / reply-attack evidence.
    """
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
