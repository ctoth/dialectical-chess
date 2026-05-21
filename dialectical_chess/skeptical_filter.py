"""Dung skeptical hard-filter for Phase-2 move arguments."""

from __future__ import annotations

from argumentation.dung import grounded_extension

from dialectical_chess.opinion_graph import MoveArgumentationArtifacts


def skeptical_survivors(artifacts: MoveArgumentationArtifacts) -> set[str]:
    """Return move UCIs whose move arguments survive the filter framework."""
    grounded = grounded_extension(artifacts.filter_af)
    return {
        uci
        for uci, move_arg in artifacts.move_arg.items()
        if move_arg in grounded
    }
