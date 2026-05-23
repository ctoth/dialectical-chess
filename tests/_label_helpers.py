"""Test helpers for chunk-F chess MoveProbe assertions.

The chess MoveProbe (chunk E) keeps chess-flavour labels on subclass
extension fields ``reason_evidence`` / ``objection_evidence`` /
``reply_attack_evidence`` while the inherited core fields hold
core-taxonomy strings. Tests that need to assert against chess labels
read from the evidence tuples through this helper so the assertion
still reads as ``"chess:label" in labels_of(probe.reason_evidence)``.
"""

from __future__ import annotations

from typing import Iterable

from dialectical_chess.evidence import ArgumentEvidence


def labels_of(evidence: Iterable[ArgumentEvidence]) -> tuple[str, ...]:
    """Return the chess-flavour ``label`` strings on an evidence tuple."""
    return tuple(item.label for item in evidence)


__all__ = ["labels_of"]
