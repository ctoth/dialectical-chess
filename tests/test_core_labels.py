"""Property tests for the chess->core label translator (Core Phase 3 chunk D).

The translator at ``dialectical_chess.core_labels`` is the cartridge boundary
where chess-typed witnesses become core-taxonomy label strings. The invariants
this module pins:

* Every emitted core label must round-trip through
  ``dialectical_games.evidence.to_argument_evidence`` (i.e. parse cleanly).
* A FACT chess objection that carries a positive material magnitude always
  produces an ``obj:loses_exchange:{magnitude}`` core label with the same
  magnitude (within the closed FACT material-safety kind set).
* A chess reply with ``forced_mate_distance`` set always produces a
  ``reply:terminal_loss`` core label.
* Chess HEURISTIC objections never produce a core label (foreman
  directive 3 — chess HEURISTIC stays out of core this cycle).
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dialectical_games.evidence import to_argument_evidence
from dialectical_games.scheme import Tier as CoreTier
from dialectical_games.scheme import Value as CoreValue

from dialectical_chess.core_labels import (
    core_labels_for_probe,
    core_objection_label,
    core_reason_label,
    core_reply_attack_label,
)
from dialectical_chess.evidence import (
    EvidenceWorld,
    ObjectionKind,
    objection_evidence,
    reply_evidence,
    support_evidence,
)


_FACT_MATERIAL_SAFETY_KINDS = (
    ObjectionKind.IGNORED_HANGING_PIECE,
    ObjectionKind.MOVED_PIECE_EN_PRIS,
    ObjectionKind.QUEEN_BLUNDER,
    ObjectionKind.QUEEN_FLANK_INVASION,
)

_FACT_REPLY_KINDS = (
    ObjectionKind.REPLY_MATE_IN_ONE,
    ObjectionKind.REPLY_FORCED_MATE,
)

_HEURISTIC_KINDS = (
    ObjectionKind.OPENING_KING_WALK,
    ObjectionKind.OPENING_PREMATURE_QUEEN,
    ObjectionKind.OPENING_PREMATURE_ROOK,
    ObjectionKind.OPENING_MINOR_RETREAT,
    ObjectionKind.FLANK_PAWN_WEAKENING,
    ObjectionKind.FLANK_PAWN_LUNGE,
    ObjectionKind.UNSUPPORTED_MAJOR_DRIFT,
    ObjectionKind.NO_IMMEDIATE_TACTICAL_WARRANT,
    ObjectionKind.THREEFOLD_REPETITION,
    ObjectionKind.FIFTY_MOVE_DRAW,
)


@pytest.mark.property
@given(
    magnitude=st.integers(min_value=1, max_value=10_000),
    kind=st.sampled_from(_FACT_MATERIAL_SAFETY_KINDS),
)
def test_fact_material_safety_translates_to_obj_loses_exchange(
    magnitude: int, kind: ObjectionKind
) -> None:
    """Every FACT-tier chess material-safety objection with a positive
    magnitude becomes ``obj:loses_exchange:{magnitude}`` in the core
    taxonomy, and that label parses to FACT MATERIAL with the same
    magnitude."""
    ev = objection_evidence(
        f"safety:{kind.value}:{magnitude}",
        world=EvidenceWorld.MATERIAL,
        objection_kind=kind,
        objection_strength=1,
        moved_piece_en_pris_value=magnitude,
    )
    label = core_objection_label(ev)
    assert label == f"obj:loses_exchange:{magnitude}"
    assert label is not None
    parsed = to_argument_evidence(label)
    assert parsed.tier is CoreTier.FACT
    assert parsed.value is CoreValue.MATERIAL
    assert parsed.magnitude == magnitude


@pytest.mark.property
@given(
    distance=st.integers(min_value=1, max_value=8),
    kind=st.sampled_from(_FACT_REPLY_KINDS),
)
def test_fact_reply_kinds_translate_to_reply_terminal_loss(
    distance: int, kind: ObjectionKind
) -> None:
    """Every reply that proves a forced mate becomes ``reply:terminal_loss``
    in the core taxonomy; the label parses to FACT WINNING."""
    ev = objection_evidence(
        f"tactical:allows_reply_forced_mate_in_{distance}:e2e4",
        world=EvidenceWorld.TACTICAL,
        objection_kind=kind,
        objection_strength=1,
        forced_mate_distance=distance,
    )
    label = core_objection_label(ev)
    assert label == "reply:terminal_loss"
    assert label is not None
    parsed = to_argument_evidence(label)
    assert parsed.tier is CoreTier.FACT
    assert parsed.value is CoreValue.WINNING


@pytest.mark.property
@given(kind=st.sampled_from(_HEURISTIC_KINDS))
def test_heuristic_chess_objections_do_not_enter_core(kind: ObjectionKind) -> None:
    """Per foreman directive 3, no chess HEURISTIC objection enters the
    core taxonomy this cycle. The translator returns ``None`` for every
    HEURISTIC kind regardless of magnitude / strength."""
    ev = objection_evidence(
        f"heuristic:{kind.value}:1",
        world=EvidenceWorld.POSITIONAL,
        objection_kind=kind,
        objection_strength=3,
    )
    assert core_objection_label(ev) is None


@pytest.mark.property
@given(
    score=st.integers(min_value=-10_000, max_value=-1),
)
def test_search_refutation_translates_to_obj_loses_exchange(score: int) -> None:
    """A negative-score search refutation becomes ``obj:loses_exchange:{|score|}``
    in the core taxonomy; the label parses to FACT MATERIAL with magnitude
    ``|score|``."""
    ev = objection_evidence(
        f"search_refutes:negamax:{score}",
        world=EvidenceWorld.SEARCH,
        objection_kind=ObjectionKind.SEARCH_REFUTATION,
        objection_strength=1,
        search_refutation_score=score,
    )
    label = core_objection_label(ev)
    assert label == f"obj:loses_exchange:{-score}"
    assert label is not None
    parsed = to_argument_evidence(label)
    assert parsed.tier is CoreTier.FACT
    assert parsed.value is CoreValue.MATERIAL
    assert parsed.magnitude == -score


@pytest.mark.property
@given(distance=st.integers(min_value=1, max_value=8))
def test_proven_reply_mate_translates_to_reply_terminal_loss(distance: int) -> None:
    """A reply with a proven ``forced_mate_distance`` becomes
    ``reply:terminal_loss`` in the core taxonomy."""
    ev = reply_evidence(
        f"reply_mate:undefended:a1a2",
        reply_attack_strength=7,
        forced_mate_distance=distance,
    )
    label = core_reply_attack_label(ev)
    assert label == "reply:terminal_loss"
    assert label is not None
    parsed = to_argument_evidence(label)
    assert parsed.tier is CoreTier.FACT


@pytest.mark.property
@given(strength=st.integers(min_value=1, max_value=100))
def test_soft_reply_attacks_do_not_enter_core(strength: int) -> None:
    """A reply attack without a forced mate proof is HEURISTIC; the
    translator returns ``None`` (no core label) for it."""
    ev = reply_evidence(
        "reply_captures:a1a2",
        reply_attack_strength=strength,
    )
    assert core_reply_attack_label(ev) is None


@pytest.mark.property
@given(magnitude=st.integers(min_value=100, max_value=10_000))
def test_material_capture_support_translates_to_pro_material(magnitude: int) -> None:
    """A chess ``material:capture:{n}`` support reason becomes
    ``pro:material:{n}`` in the core taxonomy; the label parses to FACT
    MATERIAL with that magnitude."""
    ev = support_evidence(
        f"material:capture:{magnitude}",
        world=EvidenceWorld.MATERIAL,
        counts_as_tactical=True,
        argument_value="tactical",
        support_strength=1,
    )
    label = core_reason_label(ev)
    assert label == f"pro:material:{magnitude}"
    assert label is not None
    parsed = to_argument_evidence(label)
    assert parsed.tier is CoreTier.FACT
    assert parsed.value is CoreValue.MATERIAL
    assert parsed.magnitude == magnitude


@pytest.mark.property
@given(
    magnitudes=st.lists(
        st.integers(min_value=1, max_value=10_000), min_size=0, max_size=5
    )
)
def test_core_labels_for_probe_dedupes_and_round_trips(
    magnitudes: list[int],
) -> None:
    """Across a probe's full witness set, every emitted core label parses
    cleanly through ``to_argument_evidence`` and the result tuples never
    contain duplicates (the core builder treats duplicates as one
    witness, so the translator deduplicates eagerly)."""
    objection_evidences = tuple(
        objection_evidence(
            f"safety:moved_piece_en_pris:{m}",
            world=EvidenceWorld.MATERIAL,
            objection_kind=ObjectionKind.MOVED_PIECE_EN_PRIS,
            objection_strength=1,
            moved_piece_en_pris_value=m,
        )
        for m in magnitudes
    )
    reasons, objections, reply_attacks = core_labels_for_probe(
        reason_evidence=(),
        objection_evidence=objection_evidences,
        reply_attack_evidence=(),
    )
    # No duplicates.
    assert len(objections) == len(set(objections))
    # Every label parses.
    for label in objections:
        parsed = to_argument_evidence(label)
        assert parsed.tier is CoreTier.FACT
    # Reasons / reply_attacks should be empty (no inputs).
    assert reasons == ()
    assert reply_attacks == ()
