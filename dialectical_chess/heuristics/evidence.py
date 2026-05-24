"""Evidence construction helpers for move probing."""

from __future__ import annotations

from dataclasses import dataclass

from dialectical_chess.evidence import (
    ArgumentEvidence,
    EvidenceWorld,
    ObjectionKind,
    SupportKind,
    base_objection_strength,
    objection_evidence,
    support_evidence,
)
from dialectical_chess.tuning import (
    LARGE_SEARCH_REFUTATION_THRESHOLD,
    MAJOR_PIECE_VALUE,
    MINOR_PIECE_VALUE,
    QUEEN_VALUE,
    REPLY_MATE_REFUTATION_SCORE,
)

@dataclass(frozen=True)
class EvidenceLabels:
    labels: tuple[str, ...]
    evidence: tuple[ArgumentEvidence, ...] = ()
    score: int = 0


def support(
    label: str,
    *,
    world: EvidenceWorld,
    strength: int,
    counts_as_positional: bool = False,
    counts_as_tactical: bool = False,
    argument_value: str = "procedural",
    tactical_threat_value: int = 0,
    defended_piece_value: int | None = None,
    search_support_score: int | None = None,
    support_magnitude: int | None = None,
    support_kind: SupportKind = SupportKind.GENERIC,
) -> ArgumentEvidence:
    return support_evidence(
        label,
        world=world,
        counts_as_positional=counts_as_positional,
        counts_as_tactical=counts_as_tactical,
        argument_value=argument_value,
        support_strength=strength,
        tactical_threat_value=tactical_threat_value,
        defended_piece_value=defended_piece_value,
        search_support_score=search_support_score,
        support_magnitude=support_magnitude,
        support_kind=support_kind,
    )


def display_evidence(label: str, *, world: EvidenceWorld = EvidenceWorld.PROCEDURAL) -> ArgumentEvidence:
    return support_evidence(label, world=world)


def objection(
    label: str,
    *,
    kind: ObjectionKind,
    strength: int | None = None,
    world: EvidenceWorld = EvidenceWorld.UNKNOWN,
    moved_piece_en_pris_value: int | None = None,
    search_refutation_score: int | None = None,
    forced_mate_distance: int | None = None,
    argument_value: str = "procedural",
) -> ArgumentEvidence:
    return objection_evidence(
        label,
        world=world,
        objection_kind=kind,
        objection_strength=base_objection_strength(kind) if strength is None else strength,
        moved_piece_en_pris_value=moved_piece_en_pris_value,
        search_refutation_score=search_refutation_score,
        forced_mate_distance=forced_mate_distance,
        argument_value=argument_value,
    )


def search_refutation_strength(score: int) -> int:
    if score <= REPLY_MATE_REFUTATION_SCORE:
        return 6
    if score <= LARGE_SEARCH_REFUTATION_THRESHOLD:
        return 1
    return 0


def material_support_strength(value: int) -> int:
    if value >= MAJOR_PIECE_VALUE:
        return 9
    if value >= MINOR_PIECE_VALUE:
        return 6
    if value > 0:
        return 3
    return 1


def defended_piece_support_strength(value: int) -> int:
    if value >= QUEEN_VALUE:
        return 4
    if value >= MAJOR_PIECE_VALUE:
        return 3
    return 1


