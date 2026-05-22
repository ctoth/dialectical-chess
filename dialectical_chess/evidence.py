"""Evidence-label comorphisms between chess worlds and argumentation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


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


class EvidenceRole(str, Enum):
    SUPPORT = "support"
    OBJECTION = "objection"
    DEFEATER = "defeater"
    REPLY = "reply"


class SupportKind(str, Enum):
    GENERIC = "generic"
    DEVELOPMENT = "development"


@dataclass(frozen=True)
class SupportEvidence:
    label: str
    world: EvidenceWorld
    counts_as_positional: bool
    counts_as_tactical: bool
    argument_value: str = "procedural"
    support_strength: int = 0
    tactical_threat_value: int = 0
    defended_piece_value: int | None = None
    search_support_score: int | None = None
    support_kind: SupportKind = SupportKind.GENERIC
    role: EvidenceRole = EvidenceRole.SUPPORT

    @property
    def supports_argument(self) -> bool:
        return self.support_strength > 0

    @property
    def objection_kind(self) -> ObjectionKind:
        return ObjectionKind.NONE

    @property
    def objection_strength(self) -> int:
        return 0

    @property
    def reply_attack_strength(self) -> int:
        return 0

    @property
    def defense_strength(self) -> int:
        return 0

    @property
    def defeater_kind(self) -> DefeaterKind | None:
        return None

    @property
    def defeater_strength(self) -> int:
        return 0

    @property
    def moved_piece_en_pris_value(self) -> int | None:
        return None

    @property
    def search_refutation_score(self) -> int | None:
        return None

    @property
    def forced_mate_distance(self) -> int | None:
        return None


@dataclass(frozen=True)
class ObjectionEvidence:
    label: str
    world: EvidenceWorld
    objection_kind: ObjectionKind = ObjectionKind.NONE
    objection_strength: int = 0
    moved_piece_en_pris_value: int | None = None
    search_refutation_score: int | None = None
    forced_mate_distance: int | None = None
    argument_value: str = "procedural"
    role: EvidenceRole = EvidenceRole.OBJECTION

    @property
    def supports_argument(self) -> bool:
        return False

    @property
    def counts_as_positional(self) -> bool:
        return False

    @property
    def counts_as_tactical(self) -> bool:
        return False

    @property
    def support_strength(self) -> int:
        return 0

    @property
    def support_kind(self) -> SupportKind:
        return SupportKind.GENERIC

    @property
    def reply_attack_strength(self) -> int:
        return 0

    @property
    def defense_strength(self) -> int:
        return 0

    @property
    def defeater_kind(self) -> DefeaterKind | None:
        return None

    @property
    def defeater_strength(self) -> int:
        return 0

    @property
    def defended_piece_value(self) -> int | None:
        return None

    @property
    def tactical_threat_value(self) -> int:
        return 0

    @property
    def search_support_score(self) -> int | None:
        return None


@dataclass(frozen=True)
class DefeaterEvidence:
    label: str
    world: EvidenceWorld
    defeater_kind: DefeaterKind
    defeater_strength: int
    counts_as_positional: bool = False
    counts_as_tactical: bool = False
    argument_value: str = "procedural"
    support_strength: int = 0
    tactical_threat_value: int = 0
    search_support_score: int | None = None
    support_kind: SupportKind = SupportKind.GENERIC
    role: EvidenceRole = EvidenceRole.DEFEATER

    @property
    def supports_argument(self) -> bool:
        return self.support_strength > 0

    @property
    def objection_kind(self) -> ObjectionKind:
        return ObjectionKind.NONE

    @property
    def objection_strength(self) -> int:
        return 0

    @property
    def reply_attack_strength(self) -> int:
        return 0

    @property
    def defense_strength(self) -> int:
        return 0

    @property
    def defended_piece_value(self) -> int | None:
        return None

    @property
    def moved_piece_en_pris_value(self) -> int | None:
        return None

    @property
    def search_refutation_score(self) -> int | None:
        return None

    @property
    def forced_mate_distance(self) -> int | None:
        return None


@dataclass(frozen=True)
class ReplyEvidence:
    label: str
    world: EvidenceWorld
    reply_attack_strength: int = 0
    defense_strength: int = 0
    forced_mate_distance: int | None = None
    argument_value: str = "reply_refutation"
    role: EvidenceRole = EvidenceRole.REPLY

    @property
    def supports_argument(self) -> bool:
        return False

    @property
    def counts_as_positional(self) -> bool:
        return False

    @property
    def counts_as_tactical(self) -> bool:
        return False

    @property
    def support_strength(self) -> int:
        return 0

    @property
    def support_kind(self) -> SupportKind:
        return SupportKind.GENERIC

    @property
    def objection_kind(self) -> ObjectionKind:
        return ObjectionKind.NONE

    @property
    def objection_strength(self) -> int:
        return 0

    @property
    def defeater_kind(self) -> DefeaterKind | None:
        return None

    @property
    def defeater_strength(self) -> int:
        return 0

    @property
    def defended_piece_value(self) -> int | None:
        return None

    @property
    def moved_piece_en_pris_value(self) -> int | None:
        return None

    @property
    def tactical_threat_value(self) -> int:
        return 0

    @property
    def search_refutation_score(self) -> int | None:
        return None

    @property
    def search_support_score(self) -> int | None:
        return None


def support_evidence(
    label: str,
    *,
    world: EvidenceWorld,
    counts_as_positional: bool = False,
    counts_as_tactical: bool = False,
    argument_value: str = "procedural",
    support_strength: int = 0,
    tactical_threat_value: int = 0,
    defended_piece_value: int | None = None,
    search_support_score: int | None = None,
    support_kind: SupportKind = SupportKind.GENERIC,
) -> SupportEvidence:
    return SupportEvidence(
        label=label,
        world=world,
        counts_as_positional=counts_as_positional,
        counts_as_tactical=counts_as_tactical,
        argument_value=argument_value,
        support_strength=support_strength,
        tactical_threat_value=tactical_threat_value,
        defended_piece_value=defended_piece_value,
        search_support_score=search_support_score,
        support_kind=support_kind,
    )


def objection_evidence(
    label: str,
    *,
    world: EvidenceWorld,
    objection_kind: ObjectionKind,
    objection_strength: int,
    moved_piece_en_pris_value: int | None = None,
    search_refutation_score: int | None = None,
    forced_mate_distance: int | None = None,
    argument_value: str = "procedural",
) -> ObjectionEvidence:
    return ObjectionEvidence(
        label=label,
        world=world,
        objection_kind=objection_kind,
        objection_strength=objection_strength,
        moved_piece_en_pris_value=moved_piece_en_pris_value,
        search_refutation_score=search_refutation_score,
        forced_mate_distance=forced_mate_distance,
        argument_value=argument_value,
    )


def defeater_evidence(
    label: str,
    *,
    world: EvidenceWorld,
    defeater_kind: DefeaterKind,
    defeater_strength: int,
    counts_as_positional: bool = False,
    counts_as_tactical: bool = False,
    argument_value: str = "procedural",
    support_strength: int = 0,
    tactical_threat_value: int = 0,
    search_support_score: int | None = None,
    support_kind: SupportKind = SupportKind.GENERIC,
) -> DefeaterEvidence:
    return DefeaterEvidence(
        label=label,
        world=world,
        defeater_kind=defeater_kind,
        defeater_strength=defeater_strength,
        counts_as_positional=counts_as_positional,
        counts_as_tactical=counts_as_tactical,
        argument_value=argument_value,
        support_strength=support_strength,
        tactical_threat_value=tactical_threat_value,
        search_support_score=search_support_score,
        support_kind=support_kind,
    )


def reply_evidence(
    label: str,
    *,
    reply_attack_strength: int,
    defense_strength: int = 0,
    forced_mate_distance: int | None = None,
    argument_value: str = "reply_refutation",
) -> ReplyEvidence:
    return ReplyEvidence(
        label=label,
        world=EvidenceWorld.REPLY,
        reply_attack_strength=reply_attack_strength,
        defense_strength=defense_strength,
        forced_mate_distance=forced_mate_distance,
        argument_value=argument_value,
    )


ArgumentEvidence: TypeAlias = (
    SupportEvidence | ObjectionEvidence | DefeaterEvidence | ReplyEvidence
)


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


def material_cost_objection_strength(value: int | None) -> int:
    if value is None or value < 300:
        return 0
    if value >= 900:
        return 97
    return 17


def base_objection_strength(objection_kind: ObjectionKind) -> int:
    match objection_kind:
        case ObjectionKind.SEARCH_REFUTATION:
            return 1
        case ObjectionKind.SMT_FORK_HIGH_VALUE:
            return 3
        case ObjectionKind.REPLY_MATE_IN_ONE:
            return 6
        case ObjectionKind.REPLY_FORCED_MATE:
            return 3
        case ObjectionKind.QUEEN_BLUNDER:
            return 2
        case ObjectionKind.UNANSWERED_ADVANCED_FLANK_PAWN:
            return 4
        case ObjectionKind.THREEFOLD_REPETITION | ObjectionKind.FIFTY_MOVE_DRAW:
            return 2
        case (
            ObjectionKind.IGNORED_HANGING_PIECE
            | ObjectionKind.MOVED_PIECE_EN_PRIS
            | ObjectionKind.UNSUPPORTED_MAJOR_DRIFT
            | ObjectionKind.OPENING_KING_WALK
            | ObjectionKind.OPENING_KING_CENTER_FLIGHT
            | ObjectionKind.OPENING_PREMATURE_QUEEN
            | ObjectionKind.OPENING_PREMATURE_ROOK
            | ObjectionKind.OPENING_MINOR_RETREAT
            | ObjectionKind.OPENING_PREMATURE_MINOR_CHECK
            | ObjectionKind.FLANK_PAWN_WEAKENING
            | ObjectionKind.CASTLED_FLANK_PAWN_WEAKENING
            | ObjectionKind.FLANK_PAWN_LUNGE
        ):
            return 1
        case ObjectionKind.QUEEN_FLANK_INVASION:
            return 9
        case _:
            return 0


def defeater_strength(defeater_kind: DefeaterKind) -> int:
    match defeater_kind:
        case DefeaterKind.SEARCH_SUPPORT:
            return 97
        case (
            DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE
            | DefeaterKind.COMPENSATING_FORCING_PRESSURE
            | DefeaterKind.FORCING_MATERIAL_GAIN
        ):
            return 33
        case DefeaterKind.COMPENSATING_TACTICAL_PRESSURE:
            return 17


def is_forced_mate_refutation(evidence: ArgumentEvidence) -> bool:
    if evidence.search_refutation_score is not None:
        return evidence.search_refutation_score <= -100_000
    return evidence.forced_mate_distance is not None


def forced_mate_refutation_distance(evidence: ArgumentEvidence) -> int | None:
    """Return the proven mate distance for a hard refutation, when encoded."""
    return evidence.forced_mate_distance


def is_large_search_refutation(evidence: ArgumentEvidence) -> bool:
    return (
        evidence.search_refutation_score is not None
        and evidence.search_refutation_score <= LARGE_SEARCH_REFUTATION_THRESHOLD
    )
