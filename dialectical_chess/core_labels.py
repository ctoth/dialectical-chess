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

Per the Phase-3 foreman directive 3: chess HEURISTIC vocabulary does NOT
enter the core graded layer in this cycle. Chess HEURISTIC objections (opening
play, flank pawns, positional drift) are NOT translated — they are absent
from the core ``objections`` tuple. Chess FACT objections (forced-mate, search
refutation, material safety) ARE translated. Chess support reasons are
translated when they fit the FACT taxonomy (terminal-win, material captures).
"""

from __future__ import annotations

from dialectical_chess.evidence import (
    ArgumentEvidence,
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


def core_reason_label(evidence: ArgumentEvidence) -> str | None:
    """Map chess support evidence to a core-taxonomy reason label, or
    ``None`` if it has no core analogue this cycle."""
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
    return None


def core_objection_label(evidence: ArgumentEvidence) -> str | None:
    """Map a chess objection to a core-taxonomy objection label, or
    ``None`` for a chess HEURISTIC objection that has no core analogue
    (per foreman directive 3, chess HEURISTIC labels stay out of core
    this cycle).

    Per the chess ``objection_tier`` table, the FACT-tier chess objection
    kinds are SEARCH_REFUTATION, REPLY_MATE_IN_ONE, REPLY_FORCED_MATE,
    IGNORED_HANGING_PIECE, MOVED_PIECE_EN_PRIS, QUEEN_BLUNDER,
    QUEEN_FLANK_INVASION. The first three are reply / refutation labels
    that map to ``reply:terminal_loss`` or ``obj:terminal_loss`` (they
    prove a terminal loss); the material-safety ones map to
    ``obj:loses_exchange:{n}`` carrying the material magnitude.
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
        return None
    # Chess HEURISTIC objections (opening play, flank pawn, drift, etc.)
    # stay out of the core graded layer this cycle.
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


__all__ = [
    "core_labels_for_probe",
    "core_objection_label",
    "core_reason_label",
    "core_reply_attack_label",
]
