"""Unified engine API for dialectical chess move decisions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from dialectical_chess.arguments import (
    LARGE_SEARCH_REFUTATION_THRESHOLD,
    MoveProbe,
    RootArgumentGraph,
    SELECTOR_MODES,
    build_root_argument_graph,
    choose_move,
)
from dialectical_chess.evidence import to_argument_evidence
from dialectical_chess.loss_mining import has_forced_mate
from dialectical_chess.probe import ensure_owned_board, probe_moves
from dialectical_chess.search import ReplyAnalysisSettings


@dataclass(frozen=True)
class EngineSettings:
    dialectic_depth: int = 1
    search_depth: int = 0
    search_backend: str = "negamax"
    smt_mate: bool = True
    smt_fork: bool = True
    selector_mode: str = "argument"
    positional_reasons: bool = True
    reply_mate_scan: bool = True
    reply_analysis: ReplyAnalysisSettings = ReplyAnalysisSettings()
    recent_own_move: str | None = None

    def __post_init__(self) -> None:
        if self.selector_mode not in SELECTOR_MODES:
            raise ValueError(f"unknown selector_mode: {self.selector_mode}")


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
    graph: RootArgumentGraph
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
                recent_own_move=self.settings.recent_own_move,
            )
        )
        graph = build_root_argument_graph(probes)
        selected = choose_move(probes, graph, selector_mode=self.settings.selector_mode) if probes else None
        if uses_selected_reply_mate_refutation(self.settings):
            probes, graph, selected = selected_reply_mate_refutation_fixpoint(
                board,
                probes,
                graph,
                selected,
                selector_mode=self.settings.selector_mode,
            )
        decision = EngineDecision(
            move_uci="0000" if selected is None else selected.uci,
            selected=selected,
        )
        return EngineAnalysis(probes=tuple(probes), graph=graph, decision=decision)

    def choose_move(self, board: Any) -> EngineDecision:
        return self.analyze(board).decision


def selected_reply_mate_refutation_fixpoint(
    board: Any,
    probes: list[MoveProbe],
    graph: RootArgumentGraph,
    selected: MoveProbe | None,
    *,
    selector_mode: str,
) -> tuple[list[MoveProbe], RootArgumentGraph, MoveProbe | None]:
    move_by_uci = {move.uci(): move for move in board.legal_moves()}
    refuted: set[str] = set()
    while selected is not None and selected.uci not in refuted:
        objection = selected_reply_mate_refutation(
            board,
            move_by_uci,
            selected,
            mate_depths=selected_reply_mate_depths(selected),
        )
        if objection is None:
            break
        refuted.add(selected.uci)
        probes = [
            replace(
                probe,
                objections=probe.objections + (objection,),
            )
            if probe.uci == selected.uci and objection not in probe.objections
            else probe
            for probe in probes
        ]
        graph = build_root_argument_graph(probes)
        selected = choose_move(probes, graph, selector_mode=selector_mode) if probes else None
    return probes, graph, selected


def selected_reply_mate_depths(
    selected: MoveProbe,
) -> tuple[int, ...]:
    if selected_has_large_search_refutation(selected):
        return (2, 3, 4)
    return (2, 3)


def selected_has_large_search_refutation(selected: MoveProbe) -> bool:
    for objection in selected.objections:
        score = to_argument_evidence(objection).search_refutation_score
        if score is not None and score <= LARGE_SEARCH_REFUTATION_THRESHOLD:
            return True
    return False


def uses_selected_reply_mate_refutation(settings: EngineSettings) -> bool:
    if settings.search_depth == 1:
        return True
    return settings.reply_mate_scan and settings.search_depth == 0 and not settings.positional_reasons


def selected_reply_mate_refutation(
    board: Any,
    move_by_uci: dict[str, Any],
    selected: MoveProbe,
    *,
    mate_depths: tuple[int, ...],
) -> str | None:
    if selected.is_checkmate:
        return None
    if any(objection.startswith("tactical:allows_reply_forced_mate_in_") for objection in selected.objections):
        return None
    move = move_by_uci.get(selected.uci)
    if move is None:
        return None
    child = board.apply(move)
    for mate_depth in mate_depths:
        if has_forced_mate(child, mate_depth=mate_depth):
            return f"tactical:allows_reply_forced_mate_in_{mate_depth}:{selected.uci}"
    return None
