"""Property tests for the chess->core label translator (Core Phase 3).

The translator at ``dialectical_chess.core_labels`` is the cartridge boundary
where chess-typed witnesses become core-taxonomy label strings. The invariants
this module pins (chunks D + G.1):

* Every emitted core label must round-trip through
  ``dialectical_games.evidence.to_argument_evidence`` (i.e. parse cleanly).
* A FACT chess objection that carries a positive material magnitude always
  produces an ``obj:loses_exchange:{magnitude}`` core label with the same
  magnitude (within the closed FACT material-safety kind set).
* A chess reply with ``forced_mate_distance`` set always produces a
  ``reply:terminal_loss`` core label.
* Chunk G.1: chess HEURISTIC objections, supports, and the
  ``ADVANCED_FLANK_PAWN_RESPONSE`` defeater now translate into the core
  HEURISTIC taxonomy. Each branch's round-trip is pinned below.
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
    DefeaterKind,
    EvidenceWorld,
    ObjectionKind,
    defeater_evidence,
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

# Pre-chunk-G these kinds had no core analogue. Chunk-G.1 lifts the chess
# HEURISTIC family into core via ``obj:opening:*`` / ``obj:king_safety:*`` /
# ``obj:strategy:*``. ``NO_IMMEDIATE_TACTICAL_WARRANT`` and ``NONE`` have no
# chunk-G mapping (still out-of-core HEURISTIC).
_HEURISTIC_KINDS_FIXED: tuple[tuple[ObjectionKind, str], ...] = (
    (ObjectionKind.OPENING_MINOR_RETREAT, "obj:opening:minor_retreat"),
    (ObjectionKind.OPENING_KING_WALK, "obj:opening:king_walk"),
    (ObjectionKind.OPENING_KING_CENTER_FLIGHT, "obj:opening:king_center_flight"),
    (ObjectionKind.CASTLED_FLANK_PAWN_WEAKENING, "obj:king_safety:castled_flank_pawn_weakening"),
    (ObjectionKind.FLANK_PAWN_WEAKENING, "obj:king_safety:flank_pawn_weakening"),
    (ObjectionKind.FLANK_PAWN_LUNGE, "obj:king_safety:flank_pawn_lunge"),
    (ObjectionKind.UNANSWERED_ADVANCED_FLANK_PAWN, "obj:king_safety:unanswered_advanced_flank_pawn"),
    (ObjectionKind.QUEEN_FLANK_INVASION, "obj:king_safety:queen_flank_invasion"),
    (ObjectionKind.UNSUPPORTED_MAJOR_DRIFT, "obj:strategy:unsupported_major_drift"),
    (ObjectionKind.THREEFOLD_REPETITION, "obj:strategy:threefold_repetition"),
    (ObjectionKind.FIFTY_MOVE_DRAW, "obj:strategy:fifty_move_draw"),
    (ObjectionKind.SMT_FORK_HIGH_VALUE, "obj:smt:fork:high_value_piece"),
)

_HEURISTIC_KINDS_OPENING_UNDEV: tuple[tuple[ObjectionKind, str], ...] = (
    (ObjectionKind.OPENING_PREMATURE_MINOR_CHECK, "obj:opening:premature_minor_check"),
    (ObjectionKind.OPENING_PREMATURE_ROOK, "obj:opening:premature_rook"),
    (ObjectionKind.OPENING_PREMATURE_QUEEN, "obj:opening:premature_queen"),
)

# These kinds genuinely have NO chunk-G mapping.
_HEURISTIC_KINDS_STILL_OUT_OF_CORE: tuple[ObjectionKind, ...] = (
    ObjectionKind.NO_IMMEDIATE_TACTICAL_WARRANT,
    ObjectionKind.NONE,
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
@given(kind=st.sampled_from(_HEURISTIC_KINDS_STILL_OUT_OF_CORE))
def test_residual_heuristic_kinds_have_no_core_label(kind: ObjectionKind) -> None:
    """``NO_IMMEDIATE_TACTICAL_WARRANT`` and ``NONE`` have no chunk-G mapping.

    They are the only ``ObjectionKind`` members left without a core analogue
    after chunk-G.1. The translator returns ``None`` for them regardless of
    magnitude / strength.
    """
    ev = objection_evidence(
        f"heuristic:{kind.value}:1",
        world=EvidenceWorld.POSITIONAL,
        objection_kind=kind,
        objection_strength=3,
    )
    assert core_objection_label(ev) is None


@pytest.mark.property
@given(data=st.data())
def test_heuristic_fixed_objection_kinds_translate_to_obj_key(data: st.DataObject) -> None:
    """Every chunk-G.1 fixed-HEURISTIC ``ObjectionKind`` maps to its core key.

    The label is round-tripped through ``to_argument_evidence`` and asserted
    HEURISTIC.
    """
    kind, expected = data.draw(st.sampled_from(_HEURISTIC_KINDS_FIXED))
    move = data.draw(st.sampled_from(["e2e4", "g1f3", "f1c4", "d7d5"]))
    ev = objection_evidence(
        f"heuristic:{kind.value}:{move}",
        world=EvidenceWorld.POSITIONAL,
        objection_kind=kind,
        objection_strength=1,
    )
    label = core_objection_label(ev)
    assert label == expected, f"{kind!r} -> {label!r} (expected {expected!r})"
    assert label is not None
    parsed = to_argument_evidence(label)
    assert parsed.tier is CoreTier.HEURISTIC
    assert parsed.magnitude is None


@pytest.mark.property
@given(
    data=st.data(),
    undeveloped=st.integers(min_value=1, max_value=4),
)
def test_heuristic_opening_undeveloped_minors_translate_with_magnitude(
    data: st.DataObject, undeveloped: int
) -> None:
    """The opening:premature_* HEURISTIC objections carry the
    ``:undeveloped_minors:{n}`` magnitude through to core."""
    kind, prefix = data.draw(st.sampled_from(_HEURISTIC_KINDS_OPENING_UNDEV))
    move = data.draw(st.sampled_from(["e2e4", "g1f3", "f1c4"]))
    label_in = f"opening:{kind.value}:{move}:undeveloped_minors:{undeveloped}"
    ev = objection_evidence(
        label_in,
        world=EvidenceWorld.POSITIONAL,
        objection_kind=kind,
        objection_strength=1,
    )
    label = core_objection_label(ev)
    expected = f"{prefix}:{undeveloped}"
    assert label == expected, f"{kind!r}/{undeveloped} -> {label!r} (expected {expected!r})"
    assert label is not None
    parsed = to_argument_evidence(label)
    assert parsed.tier is CoreTier.HEURISTIC
    assert parsed.magnitude == undeveloped


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


# ---------------------------------------------------------------------------
# chunk G.1 — chess HEURISTIC support translation
# ---------------------------------------------------------------------------
#
# Each chunk-G.1 HEURISTIC support label class is exercised with the chess
# emitter's actual label shape (``{prefix}:{move}:{suffix-or-magnitude}``).
# The translator must produce the expected core key and the label must
# round-trip through ``to_argument_evidence`` as HEURISTIC.

# Move strategies — hypothesis-generative over UCI move strings and square
# names matching what chess emits.

_MOVE_STRATEGY: st.SearchStrategy[str] = st.sampled_from([
    "e2e4", "g1f3", "f1c4", "d7d5", "b1c3", "e1g1", "c2c4", "f7f5",
])
_SQUARE_STRATEGY: st.SearchStrategy[str] = st.sampled_from([
    "a1", "b2", "c3", "d4", "e5", "f6", "g7", "h8",
])


@pytest.mark.property
@given(move=_MOVE_STRATEGY)
def test_chunk_g_development_center_pawn_translates(move: str) -> None:
    ev = support_evidence(
        f"development:{move}:center_pawn",
        world=EvidenceWorld.POSITIONAL,
        counts_as_positional=True,
        support_strength=1,
    )
    label = core_reason_label(ev)
    assert label == "pro:development:center_pawn"
    parsed = to_argument_evidence(label)
    assert parsed.tier is CoreTier.HEURISTIC
    assert parsed.value is CoreValue.STRUCTURE


@pytest.mark.property
@given(move=_MOVE_STRATEGY)
def test_chunk_g_development_minor_piece_translates(move: str) -> None:
    ev = support_evidence(
        f"development:{move}:minor_piece",
        world=EvidenceWorld.POSITIONAL,
        counts_as_positional=True,
        support_strength=1,
    )
    assert core_reason_label(ev) == "pro:development:minor_piece"


@pytest.mark.property
@given(move=_MOVE_STRATEGY)
def test_chunk_g_king_safety_castle_translates(move: str) -> None:
    ev = support_evidence(
        f"king_safety:{move}:castle",
        world=EvidenceWorld.POSITIONAL,
        counts_as_positional=True,
        support_strength=1,
    )
    assert core_reason_label(ev) == "pro:king_safety:castle"


@pytest.mark.property
@given(move=_MOVE_STRATEGY, count=st.integers(min_value=1, max_value=4))
def test_chunk_g_center_control_translates_with_magnitude(
    move: str, count: int
) -> None:
    ev = support_evidence(
        f"center_control:{move}:{count}",
        world=EvidenceWorld.POSITIONAL,
        counts_as_positional=True,
        support_strength=count,
    )
    label = core_reason_label(ev)
    assert label == f"pro:center_control:{count}"
    assert label is not None
    parsed = to_argument_evidence(label)
    assert parsed.tier is CoreTier.HEURISTIC
    assert parsed.magnitude == count


@pytest.mark.property
@given(move=_MOVE_STRATEGY, gain=st.integers(min_value=1, max_value=20))
def test_chunk_g_piece_activity_translates_to_pro_mobility(
    move: str, gain: int
) -> None:
    ev = support_evidence(
        f"piece_activity:{move}:mobility_gain:{gain}",
        world=EvidenceWorld.POSITIONAL,
        counts_as_positional=True,
        support_strength=1,
    )
    label = core_reason_label(ev)
    assert label == f"pro:mobility:{gain}"
    assert label is not None
    parsed = to_argument_evidence(label)
    assert parsed.tier is CoreTier.HEURISTIC
    assert parsed.value is CoreValue.MOBILITY
    assert parsed.magnitude == gain


@pytest.mark.property
@given(move=_MOVE_STRATEGY)
def test_chunk_g_passed_pawn_translates(move: str) -> None:
    ev = support_evidence(
        f"pawn_structure:{move}:passed_pawn",
        world=EvidenceWorld.POSITIONAL,
        counts_as_positional=True,
        support_strength=1,
    )
    assert core_reason_label(ev) == "pro:pawn_structure:passed_pawn"


@pytest.mark.property
@given(move=_MOVE_STRATEGY)
def test_chunk_g_open_file_translates(move: str) -> None:
    ev = support_evidence(
        f"file_control:{move}:open_file",
        world=EvidenceWorld.POSITIONAL,
        counts_as_positional=True,
        support_strength=1,
    )
    assert core_reason_label(ev) == "pro:file_control:open_file"


@pytest.mark.property
@given(move=_MOVE_STRATEGY)
def test_chunk_g_outpost_supported_translates(move: str) -> None:
    ev = support_evidence(
        f"outpost:{move}:supported",
        world=EvidenceWorld.POSITIONAL,
        counts_as_positional=True,
        support_strength=1,
    )
    assert core_reason_label(ev) == "pro:outpost:supported"


@pytest.mark.property
@given(move=_MOVE_STRATEGY, sq=_SQUARE_STRATEGY)
def test_chunk_g_escape_square_translates(move: str, sq: str) -> None:
    ev = support_evidence(
        f"king_safety:escape_square:{move}:{sq}",
        world=EvidenceWorld.POSITIONAL,
        counts_as_positional=True,
        support_strength=1,
    )
    assert core_reason_label(ev) == "pro:king_safety:escape_square"


@pytest.mark.property
@given(move=_MOVE_STRATEGY)
def test_chunk_g_advanced_flank_pawn_response_support_translates(move: str) -> None:
    """The HEURISTIC support form of ADVANCED_FLANK_PAWN_RESPONSE
    (label-prefix match) translates to the same core key as the defeater
    form (see ``test_chunk_g_defeater_advanced_flank_pawn_response``).
    """
    ev = support_evidence(
        f"king_safety:advanced_flank_pawn_response:{move}",
        world=EvidenceWorld.POSITIONAL,
        counts_as_positional=True,
        support_strength=1,
    )
    assert core_reason_label(ev) == "pro:king_safety:advanced_flank_pawn_response"


@pytest.mark.property
@given(move=_MOVE_STRATEGY, value=st.integers(min_value=100, max_value=900))
def test_chunk_g_piece_safety_defended_translates(move: str, value: int) -> None:
    ev = support_evidence(
        f"piece_safety:defended:{move}:{value}",
        world=EvidenceWorld.MATERIAL,
        counts_as_positional=True,
        support_strength=1,
        defended_piece_value=value,
    )
    label = core_reason_label(ev)
    assert label == f"pro:piece_safety:defended:{value}"
    assert label is not None
    parsed = to_argument_evidence(label)
    assert parsed.tier is CoreTier.HEURISTIC
    assert parsed.value is CoreValue.MATERIAL
    assert parsed.magnitude == value


@pytest.mark.property
@given(
    target_count=st.integers(min_value=1, max_value=8),
    target_value=st.integers(min_value=300, max_value=3000),
)
def test_chunk_g_tactical_threat_translates_with_value(
    target_count: int, target_value: int
) -> None:
    ev = support_evidence(
        f"tactical:threat:targets:{target_count}:value:{target_value}",
        world=EvidenceWorld.TACTICAL,
        counts_as_tactical=True,
        support_strength=1,
        tactical_threat_value=target_value,
    )
    label = core_reason_label(ev)
    assert label == f"pro:tactical:threat:{target_value}"
    assert label is not None
    parsed = to_argument_evidence(label)
    assert parsed.tier is CoreTier.HEURISTIC
    assert parsed.value is CoreValue.MATERIAL
    assert parsed.magnitude == target_value


@pytest.mark.property
@given(move=_MOVE_STRATEGY, gain=st.integers(min_value=1, max_value=20))
def test_chunk_g_checking_exchange_pressure_translates(move: str, gain: int) -> None:
    ev = support_evidence(
        f"tactical:checking_exchange_pressure:{move}:{gain}",
        world=EvidenceWorld.TACTICAL,
        counts_as_tactical=True,
        support_strength=1,
    )
    assert core_reason_label(ev) == "pro:tactical:checking_exchange_pressure"


@pytest.mark.property
@given(
    target_count=st.integers(min_value=1, max_value=4),
    target_value=st.integers(min_value=300, max_value=3000),
)
def test_chunk_g_smt_fork_targets_translates(
    target_count: int, target_value: int
) -> None:
    ev = support_evidence(
        f"smt:fork:targets:{target_count}:value:{target_value}",
        world=EvidenceWorld.SMT,
        counts_as_tactical=True,
        support_strength=1,
    )
    label = core_reason_label(ev)
    assert label == f"pro:smt:fork:{target_value}"
    assert label is not None
    parsed = to_argument_evidence(label)
    assert parsed.tier is CoreTier.HEURISTIC
    assert parsed.magnitude == target_value


@pytest.mark.property
@given(value=st.integers(min_value=100, max_value=900))
def test_chunk_g_smt_fork_moved_piece_en_pris_objection_translates(value: int) -> None:
    """``smt:fork:moved_piece_en_pris:{v}`` arrives via SMT_FORK_HIGH_VALUE
    or no dedicated kind; the prefix-fallback in ``_heuristic_objection_label``
    matches by label and produces the magnitude-carrying obj key."""
    ev = objection_evidence(
        f"smt:fork:moved_piece_en_pris:{value}",
        world=EvidenceWorld.SMT,
        objection_kind=ObjectionKind.NONE,
        objection_strength=1,
    )
    label = core_objection_label(ev)
    assert label == f"obj:smt:fork:moved_piece_en_pris:{value}"
    assert label is not None
    parsed = to_argument_evidence(label)
    assert parsed.tier is CoreTier.HEURISTIC
    assert parsed.magnitude == value


@pytest.mark.property
@given(move=_MOVE_STRATEGY)
def test_chunk_g_defeater_advanced_flank_pawn_response(move: str) -> None:
    """The ADVANCED_FLANK_PAWN_RESPONSE defeater re-channels as a pro:
    support — the core taxonomy has no defeater channel (chunk-G.1 §3).
    """
    ev = defeater_evidence(
        f"king_safety:advanced_flank_pawn_response:{move}",
        world=EvidenceWorld.POSITIONAL,
        defeater_kind=DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE,
        defeater_strength=33,
    )
    assert core_reason_label(ev) == "pro:king_safety:advanced_flank_pawn_response"


@pytest.mark.property
@given(
    kind=st.sampled_from([
        DefeaterKind.COMPENSATING_FORCING_PRESSURE,
        DefeaterKind.COMPENSATING_TACTICAL_PRESSURE,
        DefeaterKind.FORCING_MATERIAL_GAIN,
        DefeaterKind.SEARCH_SUPPORT,
    ]),
)
def test_chunk_g_other_defeaters_have_no_g1_mapping(kind: DefeaterKind) -> None:
    """The non-ADVANCED_FLANK_PAWN_RESPONSE defeaters have no G.1 mapping —
    they remain invisible to the core graded layer this cycle (chunk-G.1
    plan §3 / §7-D documents the F12 defeater-channel deficit)."""
    ev = defeater_evidence(
        f"defeater:{kind.value}",
        world=EvidenceWorld.POSITIONAL,
        defeater_kind=kind,
        defeater_strength=33,
    )
    assert core_reason_label(ev) is None
