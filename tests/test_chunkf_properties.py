"""Chunk-F property tests replacing the deleted seam tests.

The Phase-3 chunk-B deletions removed three of the old
``test_cartridge_seam.py`` properties that depended on the chess-local
generic surface (``opinion_graph`` / ``decide`` / ``skeptical_filter`` /
``scheme`` / ``argumentation_cartridge``). The properties themselves
remain load-bearing under the dialectical-games core; they are restated
here against the post-Phase-3 cartridge surface.

The three properties (verbatim from
``reports/core-phase3-chess-plan.md`` §6 and
``prompts/core-phase3-chess-chunkf-coder.md``):

1. ``test_chess_emits_only_core_taxonomy_labels`` — every label the chess
   cartridge emits on a chess ``MoveProbe``'s inherited core fields
   ``reasons`` / ``objections`` / ``reply_attacks`` parses to a core
   :class:`dialectical_games.evidence.ArgumentEvidence` via
   :func:`dialectical_games.evidence.to_argument_evidence` without raising.

2. ``test_chess_cartridge_does_not_export_decider_internals`` — the
   ``dialectical_chess`` package no longer exposes any of the deleted
   decider-internals modules (``decide`` / ``opinion_graph`` /
   ``move_argument`` / ``scheme`` / ``skeptical_filter``); the seam
   between cartridge and core lives in the package surface, not in
   re-exports.

3. ``test_fact_term_dominates_for_chess_material_safety`` — restating
   the FACT-as-highest-value property of the old
   ``test_generic_decider_fact_term_reads_generic_fact_tier_evidence``
   for chess: the chess cartridge's material-safety FACT label drives
   the core decider's ``fact_only_key``; a strictly-worse FACT loss
   produces a strictly-worse key than a clean move.

Each property is hypothesis-generative wherever the input space admits
meaningful sampling (so the property tests are not fixed examples
wearing ``@given`` decorators per the foreman directive).
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Iterable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dialectical_chess.board import OwnedBoard
from dialectical_chess.probe import owned_board_from_fen, probe_moves
from dialectical_chess.arguments import MoveProbe as ChessMoveProbe
from dialectical_chess.graded_policy import ChessGradedPolicy
from dialectical_games.arguments import build_root_argument_graph
from dialectical_games.decider import fact_only_key, lexicographic_decide
from dialectical_games.evidence import to_argument_evidence


# A modest pool of FENs spanning opening / middlegame / endgame / mating
# positions so the label-vocabulary property exercises a representative
# slice of the cartridge's label emission. Each FEN is selected for
# breadth (different sets of FACT / HEURISTIC labels likely to be
# emitted), not for any particular chosen-move identity.
_TAXONOMY_FENS: tuple[str, ...] = (
    # Opening, starting position.
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # Italian / Spanish-shaped middlegame.
    "r1bqk2r/1pppbppp/p1n1pn2/8/2B1P3/2N5/PPPPNPPP/R1BQK2R w KQkq - 4 6",
    # K+R vs K — clean mate-in-one (terminal-win FACT support).
    "7k/8/8/8/8/8/6PP/R5K1 w - - 0 1",
    # Material-tension middlegame.
    "rnbqkbnr/3p1ppp/p3p3/1p6/2p1P3/1BN5/PPPP1PPP/R1BQK1NR w KQkq - 0 6",
    # Hanging-piece position.
    "rnbqk1nr/ppp1bppp/4p3/3P4/8/2N5/PPPP1PPP/R1BQKBNR w KQkq - 1 4",
    # Endgame king-and-pawn.
    "7k/8/5KQ1/8/8/8/8/8 w - - 0 1",
)


@pytest.mark.parametrize("fen", _TAXONOMY_FENS)
def test_chess_emits_only_core_taxonomy_labels(fen: str) -> None:
    """Every label on a chess ``MoveProbe``'s inherited core fields
    (``reasons``, ``objections``, ``reply_attacks``) parses to a core
    :class:`ArgumentEvidence`. The property covers the chess cartridge's
    label-emission contract: only core-taxonomy strings on core fields.
    """
    board = owned_board_from_fen(fen)
    probes = probe_moves(board)
    assert probes, f"position {fen} produced no probes"
    for probe in probes:
        for label in probe.reasons:
            to_argument_evidence(label)  # raises on unknown label
        for label in probe.objections:
            to_argument_evidence(label)
        for label in probe.reply_attacks:
            to_argument_evidence(label)
        for label in probe.defenses:
            to_argument_evidence(label)


@given(fen=st.sampled_from(_TAXONOMY_FENS))
@settings(max_examples=12, deadline=None)
def test_chess_label_emission_invariant_under_repeated_probing(fen: str) -> None:
    """Property restatement: the cartridge's emitted core labels are
    stable across repeated probings of the same position, and every one
    of those labels parses cleanly. Hypothesis samples positions; the
    invariant is per-position (idempotence + closed-taxonomy)."""
    board = owned_board_from_fen(fen)
    first = probe_moves(board)
    second = probe_moves(board)
    assert len(first) == len(second)
    for probe_a, probe_b in zip(first, second):
        assert probe_a.move_id == probe_b.move_id
        # Same label sets emitted on both runs (set equality not tuple
        # equality so the property holds even if ordering becomes
        # non-deterministic in a future probe refactor).
        assert set(probe_a.reasons) == set(probe_b.reasons)
        assert set(probe_a.objections) == set(probe_b.objections)
        assert set(probe_a.reply_attacks) == set(probe_b.reply_attacks)
        # Every emitted label must parse.
        for label in (*probe_a.reasons, *probe_a.objections, *probe_a.reply_attacks):
            to_argument_evidence(label)


# Forbidden submodule names — these were chess-local-generic modules
# deleted in chunk B. The dialectical_chess package must not re-export
# any of them, neither as submodules nor as importable attributes.
_FORBIDDEN_SUBMODULES = frozenset({
    "decide",
    "opinion_graph",
    "move_argument",
    "scheme",
    "skeptical_filter",
    "argumentation_cartridge",
})


def _dialectical_chess_submodules() -> tuple[str, ...]:
    """All submodule names reachable under ``dialectical_chess``."""
    pkg = importlib.import_module("dialectical_chess")
    return tuple(info.name for info in pkgutil.iter_modules(pkg.__path__))


@pytest.mark.parametrize("forbidden", sorted(_FORBIDDEN_SUBMODULES))
def test_chess_cartridge_does_not_export_decider_internals(forbidden: str) -> None:
    """The chess cartridge no longer ships any of the deleted decider-
    internals modules. Both submodule enumeration and attribute access
    must miss for the property to hold."""
    submodules = _dialectical_chess_submodules()
    assert forbidden not in submodules, (
        f"chess cartridge still ships {forbidden} as a submodule; "
        f"chunk-B deletion incomplete"
    )
    pkg = importlib.import_module("dialectical_chess")
    assert not hasattr(pkg, forbidden), (
        f"chess cartridge still re-exports {forbidden} as a package "
        f"attribute; chunk-B deletion incomplete"
    )


@given(loss_magnitude=st.integers(min_value=100, max_value=900))
@settings(max_examples=20, deadline=None)
def test_fact_term_dominates_for_chess_material_safety(loss_magnitude: int) -> None:
    """Build two synthetic chess probes — one with no FACT objection (a
    "clean" move) and one with an ``obj:loses_exchange:{magnitude}`` FACT
    objection (a material-safety FACT loss) — and assert the FACT-only
    key of the loss probe is strictly worse than the clean probe's key
    AND that the core lexicographic decider picks the clean move. The
    property restates the chunk-B-deleted
    ``test_generic_decider_fact_term_reads_generic_fact_tier_evidence``
    for the post-Phase-3 chess cartridge.

    Hypothesis samples the magnitude over the chess centipawn range so
    the property exercises the full magnitude band the chess cartridge
    actually emits (chess piece values 100 / 300 / 320 / 500 / 900)."""
    clean_probe = ChessMoveProbe(
        move_id="a1a2",
        uci="a1a2",
        score=0,
        reasons=(),
        objections=(),
        reply_attacks=(),
        defenses=(),
        child_eval=500,
        contested=False,
    )
    loss_probe = ChessMoveProbe(
        move_id="a1b2",
        uci="a1b2",
        score=0,
        reasons=(),
        objections=(f"obj:loses_exchange:{loss_magnitude}",),
        reply_attacks=(),
        defenses=(),
        child_eval=500,
        contested=False,
    )
    probes = (clean_probe, loss_probe)
    policy = ChessGradedPolicy(board=None)
    graph = build_root_argument_graph(list(probes), policy)

    clean_key = fact_only_key(clean_probe, graph)
    loss_key = fact_only_key(loss_probe, graph)

    # Smaller is better (decider consumes via ``min``); the loss key
    # must be STRICTLY worse than the clean key for the chess FACT
    # material-safety property to hold.
    assert clean_key < loss_key, (
        f"FACT-only key for loss={loss_magnitude} was not strictly worse "
        f"than the clean key (clean={clean_key}, loss={loss_key})"
    )

    chosen = lexicographic_decide(probes, graph)
    assert chosen is not None
    assert chosen.move_id == "a1a2", (
        f"core decider picked {chosen.move_id!r} for loss={loss_magnitude}; "
        f"expected the clean move a1a2"
    )
