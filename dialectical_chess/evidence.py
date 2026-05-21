"""Evidence-label comorphisms between chess worlds and argumentation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EvidenceWorld(str, Enum):
    POSITIONAL = "positional"
    TACTICAL = "tactical"
    SEARCH = "search"
    SMT = "smt"
    MATERIAL = "material"
    TERMINAL = "terminal"
    PROCEDURAL = "procedural"
    REPLY = "reply"
    UNKNOWN = "unknown"


class ObjectionKind(str, Enum):
    NONE = "none"
    NO_IMMEDIATE_TACTICAL_WARRANT = "no_immediate_tactical_warrant"
    SEARCH_REFUTATION = "search_refutation"
    SMT_FORK_HIGH_VALUE = "smt_fork_high_value"
    REPLY_MATE_IN_ONE = "reply_mate_in_one"
    REPLY_FORCED_MATE = "reply_forced_mate"
    QUEEN_BLUNDER = "queen_blunder"
    IGNORED_HANGING_PIECE = "ignored_hanging_piece"
    MOVED_PIECE_EN_PRIS = "moved_piece_en_pris"
    QUEEN_FLANK_INVASION = "queen_flank_invasion"
    UNANSWERED_ADVANCED_FLANK_PAWN = "unanswered_advanced_flank_pawn"
    UNSUPPORTED_MAJOR_DRIFT = "unsupported_major_drift"
    IMMEDIATE_REPETITION = "immediate_repetition"
    THREEFOLD_REPETITION = "threefold_repetition"
    FIFTY_MOVE_DRAW = "fifty_move_draw"
    OPENING_KING_WALK = "opening_king_walk"
    OPENING_KING_CENTER_FLIGHT = "opening_king_center_flight"
    OPENING_PREMATURE_QUEEN = "opening_premature_queen"
    OPENING_PREMATURE_ROOK = "opening_premature_rook"
    OPENING_MINOR_RETREAT = "opening_minor_retreat"
    OPENING_PREMATURE_MINOR_CHECK = "opening_premature_minor_check"
    FLANK_PAWN_WEAKENING = "flank_pawn_weakening"
    CASTLED_FLANK_PAWN_WEAKENING = "castled_flank_pawn_weakening"
    FLANK_PAWN_LUNGE = "flank_pawn_lunge"


class DefeaterKind(str, Enum):
    COMPENSATING_FORCING_PRESSURE = "compensating_forcing_pressure"
    COMPENSATING_TACTICAL_PRESSURE = "compensating_tactical_pressure"
    FORCING_MATERIAL_GAIN = "forcing_material_gain"
    SEARCH_SUPPORT = "search_support"
    ADVANCED_FLANK_PAWN_RESPONSE = "advanced_flank_pawn_response"


@dataclass(frozen=True)
class ArgumentEvidence:
    label: str
    world: EvidenceWorld
    supports_argument: bool
    counts_as_positional: bool
    counts_as_tactical: bool
    argument_value: str = "procedural"
    support_strength: int = 0
    objection_kind: ObjectionKind = ObjectionKind.NONE
    objection_strength: int = 0
    reply_attack_strength: int = 0
    defense_strength: int = 0
    defeater_kind: DefeaterKind | None = None
    defeater_strength: int = 0
    defended_piece_value: int | None = None
    moved_piece_en_pris_value: int | None = None
    tactical_threat_value: int = 0
    search_refutation_score: int | None = None
    search_support_score: int | None = None


ARGUMENT_POSITIONAL_REASON_PREFIXES = (
    "center_control:",
    "development:",
    "file_control:",
    "king_safety:",
    "outpost:",
    "pawn_structure:",
    "piece_activity:",
    "piece_safety:",
)
REPORT_POSITIONAL_REASON_PREFIXES = (
    "center_control:",
    "development:",
    "file_control:",
    "king_safety:",
    "outpost:",
    "pawn_structure:",
    "piece_activity:",
)
TACTICAL_REASON_PREFIXES = (
    "terminal:",
    "tactical:",
    "material:",
    "procedural:",
    "smt:",
    "search:",
    "search_support:",
)
COMPENSATING_TACTICAL_THREAT_THRESHOLD = 700
LARGE_SEARCH_REFUTATION_THRESHOLD = -500

OBJECTION_STRENGTHS = {
    ObjectionKind.SEARCH_REFUTATION: 1,
    ObjectionKind.SMT_FORK_HIGH_VALUE: 3,
    ObjectionKind.REPLY_MATE_IN_ONE: 6,
    ObjectionKind.REPLY_FORCED_MATE: 3,
    ObjectionKind.QUEEN_BLUNDER: 2,
    ObjectionKind.IGNORED_HANGING_PIECE: 1,
    ObjectionKind.MOVED_PIECE_EN_PRIS: 1,
    ObjectionKind.QUEEN_FLANK_INVASION: 2,
    ObjectionKind.UNANSWERED_ADVANCED_FLANK_PAWN: 4,
    ObjectionKind.UNSUPPORTED_MAJOR_DRIFT: 1,
    ObjectionKind.IMMEDIATE_REPETITION: 2,
    ObjectionKind.THREEFOLD_REPETITION: 2,
    ObjectionKind.FIFTY_MOVE_DRAW: 2,
    ObjectionKind.OPENING_KING_WALK: 1,
    ObjectionKind.OPENING_KING_CENTER_FLIGHT: 1,
    ObjectionKind.OPENING_PREMATURE_QUEEN: 1,
    ObjectionKind.OPENING_PREMATURE_ROOK: 1,
    ObjectionKind.OPENING_MINOR_RETREAT: 1,
    ObjectionKind.OPENING_PREMATURE_MINOR_CHECK: 1,
    ObjectionKind.FLANK_PAWN_WEAKENING: 1,
    ObjectionKind.CASTLED_FLANK_PAWN_WEAKENING: 1,
    ObjectionKind.FLANK_PAWN_LUNGE: 1,
}

DEFEATER_STRENGTHS = {
    DefeaterKind.SEARCH_SUPPORT: 97,
    DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE: 33,
    DefeaterKind.COMPENSATING_FORCING_PRESSURE: 33,
    DefeaterKind.FORCING_MATERIAL_GAIN: 33,
    DefeaterKind.COMPENSATING_TACTICAL_PRESSURE: 17,
}


def to_argument_evidence(label: str) -> ArgumentEvidence:
    world = evidence_world(label)
    objection_kind = classify_objection(label)
    search_refutation = search_refutation_score(label)
    defended_value = defended_piece_value(label)
    moved_piece_value = moved_piece_en_pris_value(label)
    defeater_kind = classify_defeater(label)
    positional = is_argument_positional_reason(label)
    tactical = is_tactical_reason(label)
    supports_argument = objection_kind == ObjectionKind.NONE and (positional or tactical)
    return ArgumentEvidence(
        label=label,
        world=world,
        supports_argument=supports_argument,
        counts_as_positional=positional,
        counts_as_tactical=tactical,
        argument_value=argument_value_for(label, world, objection_kind, defeater_kind),
        support_strength=support_strength(label, world, supports_argument),
        objection_kind=objection_kind,
        objection_strength=objection_strength(label, objection_kind, search_refutation, moved_piece_value),
        reply_attack_strength=reply_attack_strength(label),
        defense_strength=defense_strength(label),
        defeater_kind=defeater_kind,
        defeater_strength=0 if defeater_kind is None else DEFEATER_STRENGTHS[defeater_kind],
        defended_piece_value=defended_value,
        moved_piece_en_pris_value=moved_piece_value,
        tactical_threat_value=tactical_threat_value(label),
        search_refutation_score=search_refutation,
        search_support_score=search_support_score(label),
    )


def evidence_world(label: str) -> EvidenceWorld:
    if label.startswith("defeater:"):
        return EvidenceWorld.PROCEDURAL
    if label.startswith("reply_"):
        return EvidenceWorld.REPLY
    if label.startswith("search"):
        return EvidenceWorld.SEARCH
    if label.startswith("smt:"):
        return EvidenceWorld.SMT
    if label.startswith("safety:"):
        return EvidenceWorld.MATERIAL
    if label.startswith("material:"):
        return EvidenceWorld.MATERIAL
    if label.startswith("terminal:"):
        return EvidenceWorld.TERMINAL
    if label.startswith("procedural:"):
        return EvidenceWorld.PROCEDURAL
    if label.startswith("tactical:"):
        return EvidenceWorld.TACTICAL
    if is_argument_positional_reason(label):
        return EvidenceWorld.POSITIONAL
    return EvidenceWorld.UNKNOWN


def classify_objection(label: str) -> ObjectionKind:
    if label == "objection:no_immediate_tactical_warrant":
        return ObjectionKind.NO_IMMEDIATE_TACTICAL_WARRANT
    if label.startswith("search_refutes:"):
        return ObjectionKind.SEARCH_REFUTATION
    if label.startswith("smt:fork:high_value_piece:"):
        return ObjectionKind.SMT_FORK_HIGH_VALUE
    if label.startswith("tactical:allows_reply_mate_in_one:"):
        return ObjectionKind.REPLY_MATE_IN_ONE
    if label.startswith("tactical:allows_reply_forced_mate_in_"):
        return ObjectionKind.REPLY_FORCED_MATE
    if label.startswith("safety:queen_blunder:"):
        return ObjectionKind.QUEEN_BLUNDER
    if label.startswith("safety:ignored_hanging_piece:"):
        return ObjectionKind.IGNORED_HANGING_PIECE
    if label.startswith("safety:moved_piece_en_pris:"):
        return ObjectionKind.MOVED_PIECE_EN_PRIS
    if label.startswith("king_safety:queen_flank_invasion:"):
        return ObjectionKind.QUEEN_FLANK_INVASION
    if label.startswith("king_safety:unanswered_advanced_flank_pawn:"):
        return ObjectionKind.UNANSWERED_ADVANCED_FLANK_PAWN
    if label.startswith("strategy:unsupported_major_drift:"):
        return ObjectionKind.UNSUPPORTED_MAJOR_DRIFT
    if label.startswith("strategy:immediate_repetition:"):
        return ObjectionKind.IMMEDIATE_REPETITION
    if label.startswith("strategy:threefold_repetition:"):
        return ObjectionKind.THREEFOLD_REPETITION
    if label.startswith("strategy:fifty_move_draw:"):
        return ObjectionKind.FIFTY_MOVE_DRAW
    if label.startswith("opening:king_walk:"):
        return ObjectionKind.OPENING_KING_WALK
    if label.startswith("opening:king_center_flight:"):
        return ObjectionKind.OPENING_KING_CENTER_FLIGHT
    if label.startswith("opening:premature_queen:"):
        return ObjectionKind.OPENING_PREMATURE_QUEEN
    if label.startswith("opening:premature_rook:"):
        return ObjectionKind.OPENING_PREMATURE_ROOK
    if label.startswith("opening:minor_retreat:"):
        return ObjectionKind.OPENING_MINOR_RETREAT
    if label.startswith("opening:premature_minor_check:"):
        return ObjectionKind.OPENING_PREMATURE_MINOR_CHECK
    if label.startswith("king_safety:flank_pawn_weakening:"):
        return ObjectionKind.FLANK_PAWN_WEAKENING
    if label.startswith("king_safety:castled_flank_pawn_weakening:"):
        return ObjectionKind.CASTLED_FLANK_PAWN_WEAKENING
    if label.startswith("king_safety:flank_pawn_lunge:"):
        return ObjectionKind.FLANK_PAWN_LUNGE
    return ObjectionKind.NONE


def classify_defeater(label: str) -> DefeaterKind | None:
    prefix = "defeater:"
    if not label.startswith(prefix):
        if label.startswith("king_safety:advanced_flank_pawn_response:"):
            return DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE
        if label.startswith("search_support:"):
            return DefeaterKind.SEARCH_SUPPORT
        return None
    value = label.removeprefix(prefix)
    try:
        return DefeaterKind(value)
    except ValueError:
        return None


def is_argument_positional_reason(reason: str) -> bool:
    return reason.startswith(ARGUMENT_POSITIONAL_REASON_PREFIXES)


def is_report_positional_reason(reason: str) -> bool:
    return reason.startswith(REPORT_POSITIONAL_REASON_PREFIXES)


def is_tactical_reason(reason: str) -> bool:
    if reason.startswith("smt:fork:"):
        parts = reason.split(":")
        return len(parts) == 4 and parts[2].isdigit() and parts[3].lstrip("-").isdigit()
    if reason.startswith("search_line:") or reason.startswith("material:exchange_nonnegative:"):
        return False
    return reason.startswith(TACTICAL_REASON_PREFIXES)


def tactical_threat_value(reason: str) -> int:
    prefix = "tactical:threat:targets:"
    if not reason.startswith(prefix):
        return 0
    parts = reason.split(":")
    if len(parts) != 6 or parts[4] != "value":
        return 0
    try:
        return int(parts[5])
    except ValueError:
        return 0


def support_strength(label: str, world: EvidenceWorld, supports_argument: bool) -> int:
    if not supports_argument:
        return 0
    if label.startswith("material:promotion:"):
        return 17
    if label.startswith("material:capture:"):
        return material_support_strength(label)
    if label.startswith("king_safety:advanced_flank_pawn_response:"):
        return 13
    if label.startswith("piece_safety:defended:"):
        return defended_piece_support_strength(label)
    if label == "tactical:check":
        return 7
    if tactical_threat_value(label) >= COMPENSATING_TACTICAL_THREAT_THRESHOLD:
        return 6
    if world in {EvidenceWorld.TERMINAL, EvidenceWorld.PROCEDURAL}:
        return 9
    if world in {EvidenceWorld.SMT, EvidenceWorld.SEARCH}:
        return 4
    if world == EvidenceWorld.TACTICAL:
        return 3
    return 1


def material_support_strength(label: str) -> int:
    parts = label.split(":")
    if len(parts) != 3:
        return 4
    try:
        value = int(parts[2])
    except ValueError:
        return 4
    if value >= 500:
        return 9
    if value >= 300:
        return 6
    if value > 0:
        return 3
    return 1


def defended_piece_support_strength(label: str) -> int:
    value = defended_piece_value(label)
    if value is None:
        return 1
    if value >= 900:
        return 4
    if value >= 500:
        return 3
    return 1


def objection_strength(
    label: str,
    objection_kind: ObjectionKind,
    refutation_score: int | None,
    moved_value: int | None,
) -> int:
    if objection_kind == ObjectionKind.SEARCH_REFUTATION:
        if refutation_score is not None and refutation_score <= -100_000:
            return 6
        if refutation_score is not None and refutation_score <= LARGE_SEARCH_REFUTATION_THRESHOLD:
            return OBJECTION_STRENGTHS[ObjectionKind.SEARCH_REFUTATION]
        return 0
    if objection_kind == ObjectionKind.REPLY_FORCED_MATE:
        return 6 if forced_mate_depth(label) == 2 else OBJECTION_STRENGTHS[objection_kind]
    if objection_kind == ObjectionKind.MOVED_PIECE_EN_PRIS and (moved_value is None or moved_value < 300):
        return 0
    return OBJECTION_STRENGTHS.get(objection_kind, 0)


def reply_attack_strength(label: str) -> int:
    if label.startswith("reply_mate:"):
        return 7
    return 1 if label.startswith("reply_") else 0


def defense_strength(label: str) -> int:
    if is_defensible_reply_attack(label):
        return 13
    return 0


def moved_piece_en_pris_value(label: str) -> int | None:
    prefix = "safety:moved_piece_en_pris:"
    if not label.startswith(prefix):
        return None
    try:
        return int(label.removeprefix(prefix))
    except ValueError:
        return None


def defended_piece_value(label: str) -> int | None:
    prefix = "piece_safety:defended:"
    if not label.startswith(prefix):
        return None
    parts = label.split(":")
    if len(parts) != 4:
        return None
    try:
        return int(parts[3])
    except ValueError:
        return None


def forced_mate_depth(label: str) -> int | None:
    prefix = "tactical:allows_reply_forced_mate_in_"
    if not label.startswith(prefix):
        return None
    depth_text = label.removeprefix(prefix).split(":", 1)[0]
    try:
        return int(depth_text)
    except ValueError:
        return None


def is_forced_mate_refutation(evidence: ArgumentEvidence) -> bool:
    if evidence.search_refutation_score is not None:
        return evidence.search_refutation_score <= -100_000
    return evidence.objection_kind in {
        ObjectionKind.REPLY_MATE_IN_ONE,
        ObjectionKind.REPLY_FORCED_MATE,
    }


def is_large_search_refutation(evidence: ArgumentEvidence) -> bool:
    return (
        evidence.search_refutation_score is not None
        and evidence.search_refutation_score <= LARGE_SEARCH_REFUTATION_THRESHOLD
    )


def search_refutation_score(label: str) -> int | None:
    prefix = "search_refutes:"
    if not label.startswith(prefix):
        return None
    parts = label.split(":")
    if len(parts) != 3:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def search_support_score(label: str) -> int | None:
    prefix = "search_support:"
    if not label.startswith(prefix):
        return None
    parts = label.split(":")
    if len(parts) != 3:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def is_undefended_reply_capture(label: str) -> bool:
    return label.startswith("reply_captures_moved_piece:undefended:")


def is_defensible_reply_attack(label: str) -> bool:
    return label.startswith("reply_captures_moved_piece:defended:")


def argument_value_for(
    label: str,
    world: EvidenceWorld,
    objection_kind: ObjectionKind,
    defeater_kind: DefeaterKind | None,
) -> str:
    if defeater_kind == DefeaterKind.SEARCH_SUPPORT:
        return "search"
    if defeater_kind == DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE:
        return "positional"
    if defeater_kind in {
        DefeaterKind.COMPENSATING_FORCING_PRESSURE,
        DefeaterKind.FORCING_MATERIAL_GAIN,
    }:
        return "material_safety"
    if defeater_kind == DefeaterKind.COMPENSATING_TACTICAL_PRESSURE:
        return "tactical"
    if objection_kind in {
        ObjectionKind.QUEEN_BLUNDER,
        ObjectionKind.IGNORED_HANGING_PIECE,
        ObjectionKind.MOVED_PIECE_EN_PRIS,
    }:
        return "material_safety"
    if objection_kind == ObjectionKind.SEARCH_REFUTATION:
        return "search"
    if objection_kind in {ObjectionKind.REPLY_MATE_IN_ONE, ObjectionKind.REPLY_FORCED_MATE}:
        return "reply_refutation"
    if world == EvidenceWorld.TERMINAL:
        return "terminal"
    if world == EvidenceWorld.SEARCH:
        return "search"
    if world in {EvidenceWorld.TACTICAL, EvidenceWorld.SMT, EvidenceWorld.MATERIAL}:
        return "tactical"
    if world == EvidenceWorld.POSITIONAL:
        return "positional"
    if world == EvidenceWorld.REPLY:
        if "reply_mate:" in label or "reply_captures_moved_piece:undefended:" in label:
            return "reply_refutation"
        return "tactical"
    return "procedural"
