"""Phase 1 — the cut cartridge seam and the explicit ``Tier`` (core extraction).

These tests pin the Phase-1 cleanup: chess now has an explicit, typed
:class:`~dialectical_chess.scheme.Tier`, and the generic argumentation
machinery (``opinion_graph``, ``decide``) is cleanly seamed off from the
chess-specific tactics — it reads only generic typed evidence, never a chess
objection-kind name. The chess suppression / material-safety policy lives in
``dialectical_chess.suppression``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dialectical_chess.evidence import (
    EvidenceWorld,
    ObjectionKind,
    objection_evidence,
    objection_tier,
    reply_evidence,
    support_evidence,
)
from dialectical_chess.suppression import fact_material_loss, suppressing_defeaters
from dialectical_games.scheme import Tier

_PACKAGE = Path(__file__).resolve().parents[1] / "dialectical_chess"


# ==========================================================================
# D1 — the explicit Tier.
# ==========================================================================


@pytest.mark.unit
def test_tier_has_exactly_fact_and_heuristic() -> None:
    """D1 — ``Tier`` is the two-member FACT / HEURISTIC enum, modelled on
    dialectical-checkers' ``Tier``."""
    assert {tier.name for tier in Tier} == {"FACT", "HEURISTIC"}
    assert Tier.FACT.value == "fact"
    assert Tier.HEURISTIC.value == "heuristic"


@pytest.mark.unit
def test_forced_mate_and_search_objections_are_fact_tier() -> None:
    """D1 — a proven loss (a forced-mate / search refutation) is FACT-tier."""
    for kind in (
        ObjectionKind.SEARCH_REFUTATION,
        ObjectionKind.REPLY_MATE_IN_ONE,
        ObjectionKind.REPLY_FORCED_MATE,
    ):
        assert objection_tier(kind) is Tier.FACT


@pytest.mark.unit
def test_material_safety_objections_are_fact_tier() -> None:
    """D1 / D2 — chess's material-safety objections are honest FACT-tier
    evidence (the reframing of the smuggled ``material_safety`` penalties)."""
    for kind in (
        ObjectionKind.MOVED_PIECE_EN_PRIS,
        ObjectionKind.IGNORED_HANGING_PIECE,
        ObjectionKind.QUEEN_BLUNDER,
        ObjectionKind.QUEEN_FLANK_INVASION,
    ):
        assert objection_tier(kind) is Tier.FACT


@pytest.mark.unit
def test_positional_objections_are_heuristic_tier() -> None:
    """D1 — a positional judgement (opening play, flank-pawn structure,
    drift) is HEURISTIC-tier, not FACT."""
    for kind in (
        ObjectionKind.OPENING_KING_WALK,
        ObjectionKind.OPENING_PREMATURE_QUEEN,
        ObjectionKind.FLANK_PAWN_WEAKENING,
        ObjectionKind.FLANK_PAWN_LUNGE,
        ObjectionKind.UNSUPPORTED_MAJOR_DRIFT,
        ObjectionKind.NO_IMMEDIATE_TACTICAL_WARRANT,
    ):
        assert objection_tier(kind) is Tier.HEURISTIC


@pytest.mark.unit
def test_objection_factory_sets_tier_from_kind() -> None:
    """D1 — the ``objection_evidence`` factory tags each objection with the
    tier of its kind; the tier is carried on the typed evidence."""
    fact = objection_evidence(
        "safety:moved_piece_en_pris:900",
        world=EvidenceWorld.MATERIAL,
        objection_kind=ObjectionKind.MOVED_PIECE_EN_PRIS,
        objection_strength=97,
        moved_piece_en_pris_value=900,
    )
    heuristic = objection_evidence(
        "opening:king_walk:e1e2",
        world=EvidenceWorld.POSITIONAL,
        objection_kind=ObjectionKind.OPENING_KING_WALK,
        objection_strength=1,
    )
    assert fact.tier is Tier.FACT
    assert heuristic.tier is Tier.HEURISTIC


@pytest.mark.unit
def test_reply_evidence_tier_tracks_forced_mate_proof() -> None:
    """D1 — a reply that proves a forced mate is FACT-tier; a soft reply
    attack is a HEURISTIC judgement."""
    proven = reply_evidence(
        "reply_mate:undefended:a1a2",
        reply_attack_strength=7,
        forced_mate_distance=1,
    )
    soft = reply_evidence("reply_captures:a1a2", reply_attack_strength=1)
    assert proven.tier is Tier.FACT
    assert soft.tier is Tier.HEURISTIC


@pytest.mark.unit
def test_support_evidence_defaults_to_heuristic_tier() -> None:
    """D1 — a support reason is a positional judgement: HEURISTIC by default."""
    reason = support_evidence(
        "development:e2e4:center_pawn",
        world=EvidenceWorld.POSITIONAL,
        counts_as_positional=True,
        support_strength=1,
    )
    assert reason.tier is Tier.HEURISTIC


# ==========================================================================
# D3 — the defeater-suppression policy is the chess cartridge.
# ==========================================================================


@pytest.mark.unit
def test_suppression_policy_suppresses_defended_reply() -> None:
    """D3 — a defended reply suppresses itself; an undefended one does not."""
    defended = reply_evidence(
        "reply_mate:defended:a1a2", reply_attack_strength=7, defense_strength=13
    )
    undefended = reply_evidence(
        "reply_mate:undefended:a1a2", reply_attack_strength=7
    )
    probe = _probe_with(reply_attacks=(defended, undefended))
    assert suppressing_defeaters(probe, defended) == (defended,)
    assert suppressing_defeaters(probe, undefended) == ()


@pytest.mark.unit
def test_suppression_policy_leaves_undefeated_objection_unsuppressed() -> None:
    """D3 — an objection with no firing chess suppression rule is returned
    unsuppressed (no defeaters)."""
    objection = objection_evidence(
        "opening:king_walk:e1e2",
        world=EvidenceWorld.POSITIONAL,
        objection_kind=ObjectionKind.OPENING_KING_WALK,
        objection_strength=1,
    )
    probe = _probe_with(objections=(objection,))
    assert suppressing_defeaters(probe, objection) == ()


# ==========================================================================
# D2 — fact_material_loss is the honest FACT material-safety term.
# ==========================================================================


@pytest.mark.unit
def test_fact_material_loss_zero_for_a_clean_move() -> None:
    """D2 — a move with no material-safety objection has zero FACT loss."""
    probe = _probe_with(objections=())
    assert fact_material_loss(probe) == 0


@pytest.mark.unit
def test_fact_material_loss_scores_search_refuted_en_pris() -> None:
    """D2 — a search-refuted en-pris piece is a proven material loss; the
    FACT term scores it (scaled by the lost material)."""
    probe = _probe_with(
        objections=(
            objection_evidence(
                "safety:moved_piece_en_pris:500",
                world=EvidenceWorld.MATERIAL,
                objection_kind=ObjectionKind.MOVED_PIECE_EN_PRIS,
                objection_strength=17,
                moved_piece_en_pris_value=500,
                search_refutation_score=-600,
            ),
        )
    )
    assert fact_material_loss(probe) > 0


@pytest.mark.unit
def test_fact_material_loss_ignores_un_refuted_en_pris() -> None:
    """D2 — an en-pris objection with no search refutation is not yet a
    *proven* loss; the FACT term stays 0 (the graded layer still sees it)."""
    probe = _probe_with(
        objections=(
            objection_evidence(
                "safety:moved_piece_en_pris:500",
                world=EvidenceWorld.MATERIAL,
                objection_kind=ObjectionKind.MOVED_PIECE_EN_PRIS,
                objection_strength=17,
                moved_piece_en_pris_value=500,
            ),
        )
    )
    assert fact_material_loss(probe) == 0


# --- helpers ---------------------------------------------------------------


def _probe_with(*, objections=(), reply_attacks=()):
    """A minimal MoveProbe carrying the given typed objection / reply evidence."""
    from dialectical_chess.arguments import MoveProbe

    return MoveProbe(
        uci="e2e4",
        san="e4",
        score=0,
        is_checkmate=False,
        gives_check=False,
        is_capture=False,
        captured_value=0,
        promotion_value=0,
        reasons=(),
        objections=tuple(ev.label for ev in objections),
        reply_attacks=tuple(ev.label for ev in reply_attacks),
        objection_evidence=tuple(objections),
        reply_attack_evidence=tuple(reply_attacks),
    )
