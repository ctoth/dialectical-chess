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
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ArgumentEvidence:
    label: str
    world: EvidenceWorld
    supports_argument: bool
    counts_as_positional: bool
    counts_as_tactical: bool
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


def to_argument_evidence(label: str) -> ArgumentEvidence:
    world = evidence_world(label)
    positional = is_argument_positional_reason(label)
    tactical = is_tactical_reason(label)
    return ArgumentEvidence(
        label=label,
        world=world,
        supports_argument=positional or tactical,
        counts_as_positional=positional,
        counts_as_tactical=tactical,
        tactical_threat_value=tactical_threat_value(label),
        search_refutation_score=search_refutation_score(label),
        search_support_score=search_support_score(label),
    )


def evidence_world(label: str) -> EvidenceWorld:
    if label.startswith("search"):
        return EvidenceWorld.SEARCH
    if label.startswith("smt:"):
        return EvidenceWorld.SMT
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
