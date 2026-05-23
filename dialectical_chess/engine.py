"""Chess cartridge: ``Cartridge`` Protocol impl + engine driver.

The chess cartridge implements the core ``dialectical_games.engine.Cartridge``
Protocol (``probe_moves`` + ``make_graded_policy``) and drives moves through
the core orchestrator ``dialectical_games.engine.analyze``. The chess-specific
reply-mate refutation fixpoint is encoded as a ``PostDecisionHook`` reading
its config off ``PostDecisionContext.cartridge_settings``.

Re-exports the core's ``EngineDecision`` / ``EngineAnalysis`` (the chess
cartridge adds a one-line ``move_uci`` alias on the chess-side wrapper for
backwards compat with 77 chess test sites).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, cast

from dialectical_games.arguments import (
    MoveProbe as CoreMoveProbe,
    build_root_argument_graph,
)
from dialectical_games.decider import lexicographic_decide
from dialectical_games.engine import (
    EngineAnalysis as CoreEngineAnalysis,
    EngineDecision as CoreEngineDecision,
    EngineSettings as CoreEngineSettings,
    PostDecisionContext,
    PostDecisionResult,
    analyze as core_analyze,
)

from dialectical_chess.arguments import MoveProbe
from dialectical_chess.evidence import (
    LARGE_SEARCH_REFUTATION_THRESHOLD,
    ArgumentEvidence,
    EvidenceWorld,
    ObjectionKind,
    forced_mate_refutation_distance,
    has_search_refutation_at_most,
    objection_evidence,
)
from dialectical_chess.graded_policy import ChessGradedPolicy
from dialectical_chess.loss_mining import has_forced_mate
from dialectical_chess.probe import ensure_owned_board, probe_moves
from dialectical_chess.search import ReplyAnalysisSettings

SELECTED_REPLY_MATE_LOW_CLOCK_LEGAL_LIMIT = 20


@dataclass(frozen=True)
class EngineSettings:
    """Chess cartridge engine settings.

    Carried opaquely on the core ``EngineSettings.cartridge_settings`` so the
    chess post-decision hook can read it. Mirrors the pre-Phase-3 chess
    EngineSettings.
    """

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
    """Chess engine decision — wraps the chosen probe.

    Carries ``move_uci`` (the chess-canonical name; 77 chess test sites
    read ``decision.move_uci``) plus a ``move_id`` alias matching the
    core's vocabulary.
    """

    move_uci: str
    selected: MoveProbe | None

    @property
    def move_id(self) -> str:
        """Alias matching ``dialectical_games.engine.EngineDecision.move_id``."""
        return self.move_uci

    @property
    def score(self) -> int | None:
        return None if self.selected is None else self.selected.score


@dataclass(frozen=True)
class EngineAnalysis:
    """Chess engine analysis — probes + decision.

    Mirrors the pre-Phase-3 chess EngineAnalysis surface; drops the core
    ``graph`` field (the core's RootArgumentGraph) which chess callers do
    not read today. Add it back if a chess caller needs the graded layer.
    """

    probes: tuple[MoveProbe, ...]
    decision: EngineDecision


class DialecticalChessEngine:
    """Chess engine — implements the core ``Cartridge`` Protocol.

    ``probe_moves(board)`` returns chess MoveProbes (subclass of core
    MoveProbe) populated with both core taxonomy labels (inherited fields,
    read by the core graph builder) and chess-flavoured evidence (chess
    extension fields, used by the chess reply-mate hook and other
    cartridge-side diagnostics).

    ``make_graded_policy(board)`` returns a fresh ``ChessGradedPolicy``
    bound to ``board``.
    """

    def __init__(self, settings: EngineSettings | None = None) -> None:
        self.settings = settings or EngineSettings()

    # --- Cartridge Protocol ------------------------------------------------

    def probe_moves(self, board: Any) -> tuple[CoreMoveProbe, ...]:
        owned = ensure_owned_board(board)
        return tuple(
            probe_moves(
                owned,
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

    def make_graded_policy(self, board: Any) -> ChessGradedPolicy:
        return ChessGradedPolicy(board=board)

    # --- chess-facing driver -----------------------------------------------

    def analyze(self, board: Any) -> EngineAnalysis:
        owned = ensure_owned_board(board)
        post_decision = (
            _reply_mate_post_decision
            if uses_selected_reply_mate_refutation(self.settings)
            else None
        )
        core_settings = CoreEngineSettings(
            search_backend=self.settings.search_backend,
            deadline=self.settings.deadline,
            cartridge_settings=self.settings,
        )
        analysis: CoreEngineAnalysis = core_analyze(
            owned,
            cartridge=self,
            settings=core_settings,
            post_decision=post_decision,
        )
        selected_probe = (
            cast(MoveProbe, analysis.decision.selected)
            if analysis.decision.selected is not None
            else None
        )
        chess_decision = EngineDecision(
            move_uci="0000" if selected_probe is None else selected_probe.uci,
            selected=selected_probe,
        )
        chess_probes = tuple(cast(MoveProbe, p) for p in analysis.probes)
        return EngineAnalysis(probes=chess_probes, decision=chess_decision)

    def choose_move(self, board: Any) -> EngineDecision:
        return self.analyze(board).decision


# --- post-decision hook: reply-mate refutation fixpoint --------------------


def _reply_mate_post_decision(
    context: PostDecisionContext,
    probes: tuple[CoreMoveProbe, ...],
    selected: CoreMoveProbe | None,
) -> PostDecisionResult:
    """Reply-mate refutation fixpoint as a core ``PostDecisionHook``.

    Walks the selected move; if a forced reply-mate refutation can be
    proved against it, appends the refutation objection (in core
    taxonomy: ``reply:terminal_loss``) to the chess MoveProbe and asks
    the core decider to re-select. Iterates until either no probe is
    selected, every selected probe has been refuted, the deadline elapses,
    or no further refutation can be proved.
    """
    settings: EngineSettings = context.cartridge_settings
    allow_mate_four = settings.reply_mate_scan
    board = context.board
    move_by_uci = {m.uci(): m for m in board.legal_moves()}
    if (
        not allow_mate_four
        and len(move_by_uci) > SELECTED_REPLY_MATE_LOW_CLOCK_LEGAL_LIMIT
    ):
        return PostDecisionResult(probes=probes, selected=selected)

    refuted: set[str] = set()
    current_probes = list(probes)
    current_selected = selected
    while current_selected is not None:
        chess_selected = cast(MoveProbe, current_selected)
        if chess_selected.uci in refuted:
            break
        if context.deadline is not None and time.monotonic() >= context.deadline:
            break
        refutation = selected_reply_mate_refutation(
            board,
            move_by_uci,
            chess_selected,
            mate_depths=selected_reply_mate_depths(
                chess_selected, allow_mate_four=allow_mate_four
            ),
            deadline=context.deadline,
        )
        if refutation is None:
            break
        refuted.add(chess_selected.uci)
        chess_label, chess_evidence = refutation
        # Translate to a core taxonomy label so the core graph builder
        # parses it as FACT (the chess label is chess-flavoured).
        core_label = "reply:terminal_loss"
        new_probes: list[CoreMoveProbe] = []
        for probe in current_probes:
            cp = cast(MoveProbe, probe)
            if cp.uci == chess_selected.uci and core_label not in cp.objections:
                new_probes.append(
                    replace(
                        cp,
                        objections=cp.objections + (core_label,),
                        objection_evidence=cp.objection_evidence + (chess_evidence,),
                    )
                )
            else:
                new_probes.append(probe)
        current_probes = new_probes
        current_selected = context.redecide(tuple(current_probes))

    return PostDecisionResult(
        probes=tuple(current_probes), selected=current_selected
    )


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
