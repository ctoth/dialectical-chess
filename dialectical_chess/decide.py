"""Opinion-valued Phase-2 move decider."""

from __future__ import annotations

from dataclasses import dataclass

from doxa import Opinion
from doxa.argumentation import evaluate

from dialectical_chess.arguments import MoveProbe
from dialectical_chess.opinion_graph import (
    MoveArgumentationArtifacts,
    build_argumentation_artifacts,
)
from dialectical_chess.skeptical_filter import skeptical_survivors


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
        key=lambda probe: (
            opinions[artifacts.move_arg[probe.uci]].expectation(),
            probe.uci,
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
