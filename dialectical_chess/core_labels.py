"""Translate chess-typed witnesses to core taxonomy labels (Core Phase 3).

The chess cartridge's witness producers (``heuristics/``, ``reply_mate_scan``,
``search``, etc.) emit chess-flavoured label strings (e.g.
``"safety:moved_piece_en_pris:500"``, ``"tactical:allows_reply_forced_mate_in_2:e2e4"``)
on chess-typed :class:`ArgumentEvidence`. The core
``dialectical_games.evidence.to_argument_evidence`` parses only its closed
taxonomy of labels (e.g. ``"obj:loses_exchange:500"``, ``"reply:terminal_loss"``).

This module maps chess-typed evidence -> core-taxonomy label strings. The chess
MoveProbe (a subclass of the core MoveProbe) populates parent-class fields
``reasons`` / ``objections`` / ``reply_attacks`` / ``defenses`` with these
core labels so the core graph builder reads them as typed FACT / HEURISTIC
evidence. The chess-flavoured originals stay on the chess-extension fields
``reason_evidence`` / ``objection_evidence`` / ``reply_attack_evidence`` for
chess-side reasoning and diagnostics.

Chunk G.1 (core Phase 3) lifted the chess HEURISTIC vocabulary into core via
new ``pro:``/``obj:`` rows in ``dialectical_games.evidence`` — this module
now translates the chess HEURISTIC family too. The chess-FACT family
(terminal-win, material captures, search refutation, material safety,
forced-mate replies) stays as before; the new HEURISTIC paths are at the
bottom of ``core_reason_label`` and dispatched by ``ObjectionKind`` in
``core_objection_label``. The defeater channel does not exist in core
taxonomy; ``ADVANCED_FLANK_PAWN_RESPONSE`` is re-channelled as a positive
``pro:`` support for the move it defends (the only chunk-G defeater
re-channel — ``SEARCH_SUPPORT``, ``COMPENSATING_*``, ``FORCING_*`` have no
G.1 mapping; see chunk-G.1 plan §3 §7-D).
"""

from __future__ import annotations

from dialectical_chess.evidence import (
    ArgumentEvidence,
    COMPENSATING_TACTICAL_THREAT_THRESHOLD,
    DefeaterEvidence,
    DefeaterKind,
    ObjectionEvidence,
    ObjectionKind,
    ReplyEvidence,
    SupportEvidence,
    SupportKind,
)


# Chess support labels that map to a core FACT label.
_TERMINAL_CHECKMATE_LABEL = "terminal:checkmate"
_PROCEDURAL_MATE_IN_ONE_LABEL = "procedural:mate_in_one"
_MATERIAL_CAPTURE_PREFIX = "material:capture:"
_MATERIAL_PROMOTION_PREFIX = "material:promotion:"

# Chess HEURISTIC objection ObjectionKind -> core label dispatch (chunk G.1).
# The opening-undeveloped-minor kinds carry an ``:undeveloped_minors:{n}``
# suffix that becomes the magnitude; the fixed kinds map directly.

_OPENING_UNDEV_PREFIXES_BY_KIND: dict[ObjectionKind, str] = {
    ObjectionKind.OPENING_PREMATURE_MINOR_CHECK: "obj:opening:premature_minor_check",
    ObjectionKind.OPENING_PREMATURE_ROOK: "obj:opening:premature_rook",
    ObjectionKind.OPENING_PREMATURE_QUEEN: "obj:opening:premature_queen",
}

_FIXED_OBJECTION_BY_KIND: dict[ObjectionKind, str] = {
    ObjectionKind.OPENING_MINOR_RETREAT: "obj:opening:minor_retreat",
    ObjectionKind.OPENING_KING_CENTER_FLIGHT: "obj:opening:king_center_flight",
    ObjectionKind.OPENING_KING_WALK: "obj:opening:king_walk",
    ObjectionKind.CASTLED_FLANK_PAWN_WEAKENING: "obj:king_safety:castled_flank_pawn_weakening",
    ObjectionKind.FLANK_PAWN_WEAKENING: "obj:king_safety:flank_pawn_weakening",
    ObjectionKind.FLANK_PAWN_LUNGE: "obj:king_safety:flank_pawn_lunge",
    ObjectionKind.UNANSWERED_ADVANCED_FLANK_PAWN: "obj:king_safety:unanswered_advanced_flank_pawn",
    ObjectionKind.QUEEN_FLANK_INVASION: "obj:king_safety:queen_flank_invasion",
    ObjectionKind.UNSUPPORTED_MAJOR_DRIFT: "obj:strategy:unsupported_major_drift",
    ObjectionKind.THREEFOLD_REPETITION: "obj:strategy:threefold_repetition",
    ObjectionKind.FIFTY_MOVE_DRAW: "obj:strategy:fifty_move_draw",
    ObjectionKind.SMT_FORK_HIGH_VALUE: "obj:smt:fork:high_value_piece",
}

_FIXED_SUPPORT_BY_KIND: dict[SupportKind, str] = {
    SupportKind.TERMINAL_WIN: "pro:terminal_win",
    SupportKind.DEVELOPMENT_CENTER_PAWN: "pro:development:center_pawn",
    SupportKind.DEVELOPMENT_MINOR_PIECE: "pro:development:minor_piece",
    SupportKind.KING_SAFETY_CASTLE: "pro:king_safety:castle",
    SupportKind.PASSED_PAWN: "pro:pawn_structure:passed_pawn",
    SupportKind.OPEN_FILE: "pro:file_control:open_file",
    SupportKind.SUPPORTED_OUTPOST: "pro:outpost:supported",
    SupportKind.KING_ESCAPE_SQUARE: "pro:king_safety:escape_square",
    SupportKind.ADVANCED_FLANK_PAWN_RESPONSE: "pro:king_safety:advanced_flank_pawn_response",
    SupportKind.CHECKING_EXCHANGE_PRESSURE: "pro:tactical:checking_exchange_pressure",
}

_MAGNITUDE_SUPPORT_PREFIX_BY_KIND: dict[SupportKind, str] = {
    SupportKind.MATERIAL_GAIN: "pro:material",
    SupportKind.CENTER_CONTROL: "pro:center_control",
    SupportKind.MOBILITY_GAIN: "pro:mobility",
    SupportKind.PIECE_DEFENDED: "pro:piece_safety:defended",
    SupportKind.TACTICAL_THREAT: "pro:tactical:threat",
    SupportKind.SMT_FORK: "pro:smt:fork",
}


def core_reason_label(evidence: ArgumentEvidence) -> str | None:
    """Map chess support evidence to a core-taxonomy reason label.

    Returns ``None`` if it has no core analogue. Chunk-G.1 lifts chess
    HEURISTIC supports into core via the ``pro:`` family at the bottom of
    this function. Chess defeaters re-channel as ``pro:`` supports here
    (the core taxonomy has no ``defeater:`` channel — chunk-G.1 plan §3).
    """
    if isinstance(evidence, DefeaterEvidence):
        return None
    if not isinstance(evidence, SupportEvidence):
        return None
    fixed = _FIXED_SUPPORT_BY_KIND.get(evidence.support_kind)
    if fixed is not None:
        return fixed
    prefix = _MAGNITUDE_SUPPORT_PREFIX_BY_KIND.get(evidence.support_kind)
    if prefix is not None:
        magnitude = _support_magnitude(evidence)
        if magnitude is not None and magnitude > 0:
            return f"{prefix}:{magnitude}"
    return None


def core_objection_label(evidence: ArgumentEvidence) -> str | None:
    """Map a chess objection to a core-taxonomy objection label.

    Chunk-G.1 lifts the chess HEURISTIC objection family into core; the
    chess FACT-tier objection kinds keep their previous mapping. The
    HEURISTIC family is dispatched by ``ObjectionKind`` against the two
    chunk-G.1 dispatch tables at the top of this module.
    """
    if not isinstance(evidence, ObjectionEvidence):
        return None
    kind = evidence.objection_kind
    if kind == ObjectionKind.SEARCH_REFUTATION:
        # A search refutation proves a terminal loss only when it's a
        # mate score; chess emits these as a negative material score, so
        # we encode them as ``obj:loses_exchange:{|score|}`` (a FACT
        # material loss with the refuted magnitude).
        score = evidence.search_refutation_score
        if score is not None and score < 0:
            return f"obj:loses_exchange:{-score}"
        return None
    if kind in (ObjectionKind.REPLY_MATE_IN_ONE, ObjectionKind.REPLY_FORCED_MATE):
        # A reply that proves a forced mate against this move - terminal
        # loss for this move.
        return "reply:terminal_loss"
    if kind in (
        ObjectionKind.IGNORED_HANGING_PIECE,
        ObjectionKind.MOVED_PIECE_EN_PRIS,
        ObjectionKind.QUEEN_BLUNDER,
        ObjectionKind.QUEEN_FLANK_INVASION,
    ):
        magnitude = evidence.moved_piece_en_pris_value
        if magnitude is None:
            # QUEEN_BLUNDER / QUEEN_FLANK_INVASION may not carry an en-pris
            # value; estimate from search_refutation_score if available.
            score = evidence.search_refutation_score
            if score is not None and score < 0:
                magnitude = -score
        if magnitude is not None and magnitude > 0:
            return f"obj:loses_exchange:{magnitude}"
        # Defensive fall-through: every FACT objection kind above SHOULD
        # carry either a ``moved_piece_en_pris_value`` (from the upstream
        # emitter) or a ``search_refutation_score`` (from search refutation).
        # If neither is set, we drop to the HEURISTIC dispatcher so the
        # objection still surfaces as some core label. The
        # ``QUEEN_FLANK_INVASION`` emitter at ``heuristics/king_safety.py``
        # now provides the en-pris value (F11 upstream emitter fix), so the
        # FACT route is live for that kind — the fallthrough here only
        # covers future regressions.
    # Chess HEURISTIC objections (chunk G.1).
    return _heuristic_objection_label(evidence)


def _heuristic_objection_label(evidence: ObjectionEvidence) -> str | None:
    """Map a chess HEURISTIC objection to its core-taxonomy ``obj:`` key.

    Dispatched by ``ObjectionKind``:
    - opening:premature_minor_check / rook / queen carry an
      ``:undeveloped_minors:{n}`` suffix that becomes the magnitude.
    - the other chunk-G.1 HEURISTIC objection kinds map to fixed keys.
    - the SMT moved-piece-en-pris objection has no dedicated kind (the
      only SMT-fork ObjectionKind is SMT_FORK_HIGH_VALUE), so we fall
      back to a label-prefix match.
    """
    kind = evidence.objection_kind
    prefix = _OPENING_UNDEV_PREFIXES_BY_KIND.get(kind)
    if prefix is not None:
        n = evidence.objection_magnitude
        if n is not None and n > 0:
            return f"{prefix}:{n}"
        return None
    fixed = _FIXED_OBJECTION_BY_KIND.get(kind)
    if fixed is not None:
        return fixed
    if kind is ObjectionKind.SMT_FORK_MOVED_PIECE_EN_PRIS:
        n = evidence.objection_magnitude
        if n is not None and n > 0:
            return f"obj:smt:fork:moved_piece_en_pris:{n}"
        return None
    return None


def core_reply_attack_label(evidence: ArgumentEvidence) -> str | None:
    """Map a chess reply-attack evidence to a core-taxonomy reply label,
    or ``None`` if the reply is HEURISTIC-only this cycle.
    """
    if not isinstance(evidence, ReplyEvidence):
        return None
    if evidence.forced_mate_distance is not None:
        # A reply that proves a forced mate - terminal loss for this move.
        return "reply:terminal_loss"
    # Soft reply attacks are HEURISTIC; no core analogue this cycle.
    return None


def core_labels_for_probe(
    *,
    reason_evidence: tuple[ArgumentEvidence, ...],
    objection_evidence: tuple[ArgumentEvidence, ...],
    reply_attack_evidence: tuple[ArgumentEvidence, ...],
    gives_check: bool = False,
    captured_value: int = 0,
    promotion_value: int = 0,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return the core-taxonomy label tuples for the chess witnesses on one probe.

    The tuple shape is ``(reasons, objections, reply_attacks, defenses)``.
    label tuples for the chess witnesses on one probe. Duplicates are
    suppressed so the core graph builder sees each FACT label at most
    once per probe (the core builder treats duplicates as a single
    witness)."""
    reasons: list[str] = []
    for evidence in reason_evidence:
        label = core_reason_label(evidence)
        if label is not None and label not in reasons:
            reasons.append(label)
    objections: list[str] = []
    for evidence in objection_evidence:
        label = core_objection_label(evidence)
        if label is not None and label not in objections:
            objections.append(label)
    reply_attacks: list[str] = []
    for evidence in reply_attack_evidence:
        label = core_reply_attack_label(evidence)
        if label is not None and label not in reply_attacks:
            reply_attacks.append(label)
    defenses = core_defense_labels_for_probe(
        reason_evidence=reason_evidence,
        objection_evidence=objection_evidence,
        reply_attack_evidence=reply_attack_evidence,
        gives_check=gives_check,
        captured_value=captured_value,
        promotion_value=promotion_value,
    )
    return tuple(reasons), tuple(objections), tuple(reply_attacks), defenses


def core_defense_labels_for_probe(
    *,
    reason_evidence: tuple[ArgumentEvidence, ...],
    objection_evidence: tuple[ArgumentEvidence, ...],
    reply_attack_evidence: tuple[ArgumentEvidence, ...],
    gives_check: bool = False,
    captured_value: int = 0,
    promotion_value: int = 0,
) -> tuple[str, ...]:
    """Return keyed core defense labels produced by typed chess defeaters."""
    del reply_attack_evidence
    defenses: list[str] = []
    for objection in objection_evidence:
        if not isinstance(objection, ObjectionEvidence):
            continue
        answered = core_objection_label(objection)
        if answered is None:
            continue
        for defeater_kind in _defeaters_for_objection(
            objection,
            reason_evidence=reason_evidence,
            gives_check=gives_check,
            material_gain=captured_value + promotion_value,
        ):
            del defeater_kind
            label = f"defense:heuristic_suppression@{answered}"
            if label not in defenses:
                defenses.append(label)
    return tuple(defenses)


def _defeaters_for_objection(
    objection: ObjectionEvidence,
    *,
    reason_evidence: tuple[ArgumentEvidence, ...],
    gives_check: bool,
    material_gain: int,
) -> tuple[DefeaterKind, ...]:
    kind = objection.objection_kind
    defeaters: list[DefeaterKind] = []
    if kind is ObjectionKind.QUEEN_BLUNDER and _has_compensating_forcing_pressure(
        reason_evidence, gives_check=gives_check, material_gain=material_gain
    ):
        defeaters.append(DefeaterKind.COMPENSATING_FORCING_PRESSURE)
    if (
        kind is ObjectionKind.MOVED_PIECE_EN_PRIS
        and objection.moved_piece_en_pris_value is not None
        and objection.moved_piece_en_pris_value >= 300
    ):
        if _has_compensating_tactical_pressure(reason_evidence):
            defeaters.append(DefeaterKind.COMPENSATING_TACTICAL_PRESSURE)
        if gives_check and material_gain > 0:
            defeaters.append(DefeaterKind.FORCING_MATERIAL_GAIN)
    if kind is ObjectionKind.OPENING_PREMATURE_MINOR_CHECK and _has_reason_defeater(
        reason_evidence, DefeaterKind.SEARCH_SUPPORT
    ):
        defeaters.append(DefeaterKind.SEARCH_SUPPORT)
    if kind in {
        ObjectionKind.FLANK_PAWN_WEAKENING,
        ObjectionKind.FLANK_PAWN_LUNGE,
    } and _has_reason_defeater(
        reason_evidence, DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE
    ):
        defeaters.append(DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE)
    return tuple(defeaters)


def _has_compensating_forcing_pressure(
    evidence_items: tuple[ArgumentEvidence, ...],
    *,
    gives_check: bool,
    material_gain: int,
) -> bool:
    return _has_compensating_tactical_pressure(evidence_items) and (
        gives_check or material_gain > 0
    )


def _has_compensating_tactical_pressure(
    evidence_items: tuple[ArgumentEvidence, ...],
) -> bool:
    return any(
        isinstance(evidence, SupportEvidence | DefeaterEvidence)
        and evidence.tactical_threat_value >= COMPENSATING_TACTICAL_THREAT_THRESHOLD
        for evidence in evidence_items
    )


def _has_reason_defeater(
    evidence_items: tuple[ArgumentEvidence, ...],
    kind: DefeaterKind,
) -> bool:
    return any(
        isinstance(evidence, DefeaterEvidence)
        and evidence.defeater_kind is kind
        for evidence in evidence_items
    )


def _support_magnitude(evidence: SupportEvidence) -> int | None:
    if evidence.support_magnitude is not None:
        return evidence.support_magnitude
    if evidence.defended_piece_value is not None:
        return evidence.defended_piece_value
    if evidence.tactical_threat_value > 0:
        return evidence.tactical_threat_value
    return evidence.search_support_score


def _parse_int_suffix(label: str, prefix: str) -> int | None:
    """Parse a base-10 integer suffix from ``label`` after ``prefix``."""
    tail = label[len(prefix):]
    if not tail.isascii() or not tail.isdecimal():
        return None
    return int(tail)


def _parse_int_after_last_colon(label: str) -> int | None:
    """Parse the integer after the final ``:`` of ``label`` (or ``None``)."""
    tail = label.rpartition(":")[2]
    if not tail.isascii() or not tail.isdecimal():
        return None
    return int(tail)


__all__ = [
    "core_labels_for_probe",
    "core_objection_label",
    "core_reason_label",
    "core_reply_attack_label",
]
