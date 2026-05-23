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
    DefeaterEvidence,
    DefeaterKind,
    ObjectionEvidence,
    ObjectionKind,
    ReplyEvidence,
    SupportEvidence,
)


# Chess support labels that map to a core FACT label.
_TERMINAL_CHECKMATE_LABEL = "terminal:checkmate"
_PROCEDURAL_MATE_IN_ONE_LABEL = "procedural:mate_in_one"
_MATERIAL_CAPTURE_PREFIX = "material:capture:"
_MATERIAL_PROMOTION_PREFIX = "material:promotion:"

# Chess HEURISTIC support label prefixes / suffixes (chunk G.1).
_DEV_CENTER_PAWN_SUFFIX = ":center_pawn"
_DEV_MINOR_PIECE_SUFFIX = ":minor_piece"
_KS_CASTLE_SUFFIX = ":castle"
_CENTER_CONTROL_PREFIX = "center_control:"
_PIECE_ACTIVITY_PREFIX = "piece_activity:"
_PIECE_ACTIVITY_MOBILITY_INFIX = ":mobility_gain:"
_PAWN_STRUCTURE_PASSED_SUFFIX = ":passed_pawn"
_FILE_CONTROL_OPEN_SUFFIX = ":open_file"
_OUTPOST_SUPPORTED_SUFFIX = ":supported"
_KS_ESCAPE_SQUARE_PREFIX = "king_safety:escape_square:"
_KS_ADV_FLANK_RESP_PREFIX = "king_safety:advanced_flank_pawn_response:"
_PIECE_SAFETY_DEFENDED_PREFIX = "piece_safety:defended:"
_TACTICAL_THREAT_PREFIX = "tactical:threat:"
_TACTICAL_CHECK_EXCH_PREFIX = "tactical:checking_exchange_pressure:"
_SMT_FORK_TARGETS_PREFIX = "smt:fork:targets:"
_SMT_FORK_MOVED_PIECE_EN_PRIS_PREFIX = "smt:fork:moved_piece_en_pris:"


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


def core_reason_label(evidence: ArgumentEvidence) -> str | None:
    """Map chess support evidence to a core-taxonomy reason label.

    Returns ``None`` if it has no core analogue. Chunk-G.1 lifts chess
    HEURISTIC supports into core via the ``pro:`` family at the bottom of
    this function. Chess defeaters re-channel as ``pro:`` supports here
    (the core taxonomy has no ``defeater:`` channel — chunk-G.1 plan §3).
    """
    if isinstance(evidence, DefeaterEvidence):
        # Defeater evidence re-channelled as a positive pro: support for the
        # move it defends (chunk G.1 — only ADVANCED_FLANK_PAWN_RESPONSE
        # maps; the others have no G.1 mapping). See chunk-G.1 plan §3.
        return _core_defeater_label(evidence)
    if not isinstance(evidence, SupportEvidence):
        return None
    label = evidence.label
    if label == _TERMINAL_CHECKMATE_LABEL or label == _PROCEDURAL_MATE_IN_ONE_LABEL:
        return "pro:terminal_win"
    if label.startswith(_MATERIAL_CAPTURE_PREFIX):
        magnitude = _parse_int_suffix(label, _MATERIAL_CAPTURE_PREFIX)
        if magnitude is not None and magnitude > 0:
            return f"pro:material:{magnitude}"
        return None
    if label.startswith(_MATERIAL_PROMOTION_PREFIX):
        magnitude = _parse_int_suffix(label, _MATERIAL_PROMOTION_PREFIX)
        if magnitude is not None and magnitude > 0:
            return f"pro:material:{magnitude}"
        return None
    # Chess HEURISTIC support family (chunk G.1). Each branch is the
    # smallest mechanical translation from the chess emitter's stringly-
    # typed label to a core-taxonomy ``pro:`` key.
    if label.startswith("development:") and label.endswith(_DEV_CENTER_PAWN_SUFFIX):
        return "pro:development:center_pawn"
    if label.startswith("development:") and label.endswith(_DEV_MINOR_PIECE_SUFFIX):
        return "pro:development:minor_piece"
    if label.startswith("king_safety:") and label.endswith(_KS_CASTLE_SUFFIX):
        return "pro:king_safety:castle"
    if label.startswith(_CENTER_CONTROL_PREFIX):
        n = _parse_int_after_last_colon(label)
        if n is not None and n > 0:
            return f"pro:center_control:{n}"
        return None
    if label.startswith(_PIECE_ACTIVITY_PREFIX) and _PIECE_ACTIVITY_MOBILITY_INFIX in label:
        n = _parse_int_after_last_colon(label)
        if n is not None and n > 0:
            return f"pro:mobility:{n}"
        return None
    if label.startswith("pawn_structure:") and label.endswith(_PAWN_STRUCTURE_PASSED_SUFFIX):
        return "pro:pawn_structure:passed_pawn"
    if label.startswith("file_control:") and label.endswith(_FILE_CONTROL_OPEN_SUFFIX):
        return "pro:file_control:open_file"
    if label.startswith("outpost:") and label.endswith(_OUTPOST_SUPPORTED_SUFFIX):
        return "pro:outpost:supported"
    if label.startswith(_KS_ESCAPE_SQUARE_PREFIX):
        return "pro:king_safety:escape_square"
    if label.startswith(_KS_ADV_FLANK_RESP_PREFIX):
        return "pro:king_safety:advanced_flank_pawn_response"
    if label.startswith(_PIECE_SAFETY_DEFENDED_PREFIX):
        n = _parse_int_after_last_colon(label)
        if n is not None and n > 0:
            return f"pro:piece_safety:defended:{n}"
        return None
    if label.startswith(_TACTICAL_THREAT_PREFIX):
        # "tactical:threat:targets:{c}:value:{v}" — take v as magnitude.
        n = _parse_int_after_last_colon(label)
        if n is not None and n > 0:
            return f"pro:tactical:threat:{n}"
        return None
    if label.startswith(_TACTICAL_CHECK_EXCH_PREFIX):
        return "pro:tactical:checking_exchange_pressure"
    if label.startswith(_SMT_FORK_TARGETS_PREFIX):
        # "smt:fork:targets:{n}:value:{v}" — take v as magnitude.
        n = _parse_int_after_last_colon(label)
        if n is not None and n > 0:
            return f"pro:smt:fork:{n}"
        return None
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
        # Chunk-G.1: when the FACT route is dead (no magnitude), fall
        # through to the HEURISTIC dispatcher so QUEEN_FLANK_INVASION still
        # gets the HEURISTIC obj label (the FACT-route fix at the upstream
        # emitter — heuristics/king_safety.py:209-217 — is deferred).
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
        # opening:premature_*:{move}:undeveloped_minors:{n} -> obj:.. :{n}.
        n = _parse_int_after_last_colon(evidence.label)
        if n is not None and n > 0:
            return f"{prefix}:{n}"
        return None
    fixed = _FIXED_OBJECTION_BY_KIND.get(kind)
    if fixed is not None:
        return fixed
    # smt:fork:moved_piece_en_pris:{n} arrives with no dedicated kind —
    # SMT_FORK_HIGH_VALUE is the only fork ObjectionKind. Differentiate
    # by label prefix.
    if evidence.label.startswith(_SMT_FORK_MOVED_PIECE_EN_PRIS_PREFIX):
        n = _parse_int_after_last_colon(evidence.label)
        if n is not None and n > 0:
            return f"obj:smt:fork:moved_piece_en_pris:{n}"
        return None
    return None


def _core_defeater_label(evidence: DefeaterEvidence) -> str | None:
    """Re-channel a chess defeater as a pro: support label (chunk G.1).

    The core taxonomy has no ``defeater:`` channel; chess defeaters become
    positive supports for the move they defend. Only
    ``ADVANCED_FLANK_PAWN_RESPONSE`` has a G.1 mapping; the others
    (``SEARCH_SUPPORT``, ``COMPENSATING_TACTICAL_PRESSURE``,
    ``COMPENSATING_FORCING_PRESSURE``, ``FORCING_MATERIAL_GAIN``) stay
    invisible to the core graded layer for this cycle — see chunk-G.1
    plan §7-D for the F12 defeater-channel deficit.
    """
    kind = evidence.defeater_kind
    if kind is DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE:
        return "pro:king_safety:advanced_flank_pawn_response"
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
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return the ``(reasons, objections, reply_attacks)`` core-taxonomy
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
    return tuple(reasons), tuple(objections), tuple(reply_attacks)


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
