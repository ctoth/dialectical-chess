"""Unified engine API for dialectical chess move decisions."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

from dialectical_chess.arguments import (
    MoveProbe,
    choose_move,
)
from dialectical_chess.evidence import (
    LARGE_SEARCH_REFUTATION_THRESHOLD,
    ArgumentEvidence,
    EvidenceWorld,
    ObjectionKind,
    forced_mate_refutation_distance,
    has_search_refutation_at_most,
    objection_evidence,
)
from dialectical_chess.loss_mining import has_forced_mate
from dialectical_chess.probe import ensure_owned_board, probe_moves
from dialectical_chess.search import ReplyAnalysisSettings

SELECTED_REPLY_MATE_LOW_CLOCK_LEGAL_LIMIT = 20


@dataclass(frozen=True)
class EngineSettings:
    dialectic_depth: int = 1
    search_depth: int = 0
    search_backend: str = "negamax"
    smt_mate: bool = True
    smt_fork: bool = True
    positional_reasons: bool = True
    reply_mate_scan: bool = True
    reply_analysis: ReplyAnalysisSettings = ReplyAnalysisSettings()
    position_history: tuple[str, ...] = ()
    deadline: float | None = None


@dataclass(frozen=True)
class EngineDecision:
    move_uci: str
    selected: MoveProbe | None

    @property
    def score(self) -> int | None:
        return None if self.selected is None else self.selected.score


@dataclass(frozen=True)
class EngineAnalysis:
    probes: tuple[MoveProbe, ...]
    decision: EngineDecision


class DialecticalChessEngine:
    """Reusable engine surface used by UCI, benchmarks, and probe adapters."""

    def __init__(self, settings: EngineSettings | None = None) -> None:
        self.settings = settings or EngineSettings()

    def analyze(self, board: Any) -> EngineAnalysis:
        board = ensure_owned_board(board)
        probes = list(
            probe_moves(
                board,
                dialectic_depth=self.settings.dialectic_depth,
                search_depth=self.settings.search_depth,
                search_backend=self.settings.search_backend,
                smt_mate=self.settings.smt_mate,
                smt_fork=self.settings.smt_fork,
                positional_reasons=self.settings.positional_reasons,
                reply_mate_scan=self.settings.reply_mate_scan,
                reply_analysis=self.settings.reply_analysis,
                position_history=self.settings.position_history,
                deadline=self.settings.deadline,
            )
        )
        selected = choose_move(probes) if probes else None
        if uses_selected_reply_mate_refutation(self.settings):
            probes, selected = selected_reply_mate_refutation_fixpoint(
                board,
                probes,
                selected,
                allow_mate_four=self.settings.reply_mate_scan,
                deadline=self.settings.deadline,
        )
        decision = EngineDecision(
            move_uci="0000" if selected is None else selected.uci,
            selected=selected,
        )
        return EngineAnalysis(probes=tuple(probes), decision=decision)

    def choose_move(self, board: Any) -> EngineDecision:
        return self.analyze(board).decision


def selected_reply_mate_refutation_fixpoint(
    board: Any,
    probes: list[MoveProbe],
    selected: MoveProbe | None,
    *,
    allow_mate_four: bool,
    deadline: float | None = None,
) -> tuple[list[MoveProbe], MoveProbe | None]:
    move_by_uci = {move.uci(): move for move in board.legal_moves()}
    if not allow_mate_four and len(move_by_uci) > SELECTED_REPLY_MATE_LOW_CLOCK_LEGAL_LIMIT:
        return probes, selected
    refuted: set[str] = set()
    while selected is not None and selected.uci not in refuted:
        if deadline is not None and time.monotonic() >= deadline:
            break
        refutation = selected_reply_mate_refutation(
            board,
            move_by_uci,
            selected,
            mate_depths=selected_reply_mate_depths(selected, allow_mate_four=allow_mate_four),
            deadline=deadline,
        )
        if refutation is None:
            break
        refuted.add(selected.uci)
        objection_label, objection = refutation
        probes = [
            replace(
                probe,
                objections=probe.objections + (objection_label,),
                objection_evidence=probe.objection_evidence + (objection,),
            )
            if probe.uci == selected.uci and objection_label not in probe.objections
            else probe
            for probe in probes
        ]
        selected = choose_move(probes) if probes else None
    return probes, selected


def selected_reply_mate_depths(
    selected: MoveProbe,
    *,
    allow_mate_four: bool,
) -> tuple[int, ...]:
    if allow_mate_four and selected_has_large_search_refutation(selected):
        return (2, 3, 4)
    return (2, 3)


def selected_has_large_search_refutation(selected: MoveProbe) -> bool:
    return has_search_refutation_at_most(
        selected.objection_evidence,
        LARGE_SEARCH_REFUTATION_THRESHOLD,
    )


def uses_selected_reply_mate_refutation(settings: EngineSettings) -> bool:
    if settings.search_depth == 1:
        return True
    return settings.search_depth == 0 and not settings.positional_reasons


def selected_reply_mate_refutation(
    board: Any,
    move_by_uci: dict[str, Any],
    selected: MoveProbe,
    *,
    mate_depths: tuple[int, ...],
    deadline: float | None = None,
) -> tuple[str, ArgumentEvidence] | None:
    if selected.is_checkmate:
        return None
    if any(
        forced_mate_refutation_distance(objection) is not None
        for objection in selected.objection_evidence
    ):
        return None
    move = move_by_uci.get(selected.uci)
    if move is None:
        return None
    child = board.apply(move)
    for mate_depth in mate_depths:
        if deadline is not None and time.monotonic() >= deadline:
            break
        if has_forced_mate(child, mate_depth=mate_depth, deadline=deadline):
            label = f"tactical:allows_reply_forced_mate_in_{mate_depth}:{selected.uci}"
            return (
                label,
                objection_evidence(
                    label,
                    world=EvidenceWorld.TACTICAL,
                    objection_kind=ObjectionKind.REPLY_FORCED_MATE,
                    objection_strength=6 if mate_depth == 2 else 3,
                    forced_mate_distance=mate_depth,
                    argument_value="reply_refutation",
                ),
            )
    return None
