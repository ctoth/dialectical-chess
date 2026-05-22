"""Move-probe data and the Phase-2 argumentation decision hook."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dialectical_chess.evidence import ArgumentEvidence


@dataclass(frozen=True)
class MoveProbe:
    uci: str
    san: str
    score: int
    is_checkmate: bool
    gives_check: bool
    is_capture: bool
    captured_value: int
    promotion_value: int
    reasons: tuple[str, ...]
    objections: tuple[str, ...]
    reply_attacks: tuple[str, ...] = ()
    search_score: int | None = None
    search_line: tuple[str, ...] = ()
    smt_witnesses: tuple[str, ...] = ()
    post_fen: str | None = None
    reason_evidence: tuple[ArgumentEvidence, ...] = ()
    objection_evidence: tuple[ArgumentEvidence, ...] = ()
    reply_attack_evidence: tuple[ArgumentEvidence, ...] = ()


def choose_move(
    probes: list[MoveProbe],
    *,
    deadline: float | None = None,
) -> MoveProbe:
    from dialectical_chess.argumentation_cartridge import choose_move_argumentation

    decision = choose_move_argumentation(probes, deadline=deadline)
    return decision.selected


def build_argument_payload(probes: list[MoveProbe]) -> dict[str, Any]:
    return {
        "move_scores": [probe_payload(probe) for probe in probes],
    }


def probe_payload(probe: MoveProbe) -> dict[str, Any]:
    payload = asdict(probe)
    payload.pop("reason_evidence", None)
    payload.pop("objection_evidence", None)
    payload.pop("reply_attack_evidence", None)
    return payload
