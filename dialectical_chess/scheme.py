"""Argumentation scheme primitives shared by the generic graph layer.

The generic argumentation machinery (``opinion_graph``, ``decide``) reads
exactly one game-agnostic discriminant off a piece of evidence: its
:class:`Tier`. ``Tier`` is the explicit, typed form of chess's formerly
implicit FACT/graded split — proven facts (forced-mate refutations, material
loss) versus positional judgements.

This module is the chess-side analogue of ``dialectical_checkers.scheme``: a
closed taxonomy the generic layer keys off, with zero chess-specific rule
names. The chess value vocabulary (``ObjectionKind`` / ``DefeaterKind`` /
``SupportKind``) stays in ``evidence.py`` — it is cartridge, not scheme.
"""

from __future__ import annotations

from enum import Enum


class Tier(Enum):
    """Whether a piece of evidence is a proven fact or a positional judgement.

    ``FACT`` evidence is resolver- or terminal-proven — a forced-mate
    refutation, or a material loss the search has confirmed. ``HEURISTIC``
    evidence is a positional judgement (development, centre control, a soft
    objection). The generic decider orders every FACT term strictly before
    every graded (HEURISTIC-derived) term: a FACT decision always dominates a
    graded one (the Bench-Capon fact-as-highest-value bridge).
    """

    FACT = "fact"
    HEURISTIC = "heuristic"
