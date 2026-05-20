from __future__ import annotations

from argparse import Namespace
from io import StringIO

import chess
from hypothesis import given
from hypothesis import strategies as st


@given(
    dialectic_depth=st.integers(min_value=0, max_value=3),
    search_depth=st.integers(min_value=0, max_value=3),
    search_backend=st.sampled_from(("negamax", "alphabeta")),
    smt_mate=st.booleans(),
    smt_fork=st.booleans(),
)
def test_engine_settings_are_plain_serializable(
    dialectic_depth: int,
    search_depth: int,
    search_backend: str,
    smt_mate: bool,
    smt_fork: bool,
) -> None:
    from dataclasses import asdict

    from dialectical_chess.engine import EngineSettings

    settings = EngineSettings(
        dialectic_depth=dialectic_depth,
        search_depth=search_depth,
        search_backend=search_backend,
        smt_mate=smt_mate,
        smt_fork=smt_fork,
    )

    assert asdict(settings) == {
        "dialectic_depth": dialectic_depth,
        "search_depth": search_depth,
        "search_backend": search_backend,
        "smt_mate": smt_mate,
        "smt_fork": smt_fork,
        "selector_mode": "argument",
        "positional_reasons": True,
        "reply_mate_scan": True,
        "reply_analysis": {
            "max_replies": 128,
            "max_defense_nodes": 5000,
            "min_defense_material": 300,
        },
        "recent_own_move": None,
    }


def test_engine_selects_mate_in_one() -> None:
    from dialectical_chess import DialecticalChessEngine as PackageEngine
    from dialectical_chess.engine import DialecticalChessEngine
    from dialectical_chess.probe import owned_board_from_fen

    assert PackageEngine is DialecticalChessEngine
    board = owned_board_from_fen("7k/6pp/8/8/8/8/6PP/R5K1 w - - 0 1")

    decision = DialecticalChessEngine().choose_move(board)

    assert decision.move_uci == "a1a8"
    assert decision.selected is not None
    assert decision.selected.score == 2_001_050
    assert "procedural:mate_in_one" in decision.selected.reasons
    assert "smt:mate_in_one" not in decision.selected.reasons


def test_engine_returns_null_decision_for_no_legal_moves() -> None:
    from dialectical_chess.engine import DialecticalChessEngine
    from dialectical_chess.probe import owned_board_from_fen

    board = owned_board_from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")

    decision = DialecticalChessEngine().choose_move(board)

    assert decision.move_uci == "0000"
    assert decision.selected is None


def test_benchmark_adapter_scores_through_engine(monkeypatch) -> None:
    import dialectical_chess.bench as bench
    from dialectical_chess.arguments import MoveProbe
    from dialectical_chess.engine import EngineDecision

    class FakeEngine:
        def __init__(self, settings):
            self.settings = settings

        def choose_move(self, board):
            selected = MoveProbe(
                uci="a1a8",
                san="a1a8",
                score=123,
                is_checkmate=True,
                gives_check=True,
                is_capture=False,
                captured_value=0,
                promotion_value=0,
                reasons=("fake:engine",),
                objections=(),
                optimizer_trace={"status": "optimal", "selected_candidate": "move:a1a8"},
            )
            return EngineDecision(move_uci="a1a8", selected=selected)

    monkeypatch.setattr(bench, "DialecticalChessEngine", FakeEngine)

    args = Namespace(
        dialectic_depth=1,
        search_depth=0,
        search_backend="negamax",
        smt_mate=True,
        selector_mode="argument",
        positional_reasons=True,
    )

    result = bench.score_board(chess.Board(), {"a1a8"}, args)

    assert result["selected_uci"] == "a1a8"
    assert result["reasons"] == ["fake:engine"]
    assert result["optimizer_trace"]["status"] == "optimal"


def test_uci_adapter_scores_through_engine(monkeypatch) -> None:
    import dialectical_chess.uci as uci
    from dialectical_chess.arguments import MoveProbe
    from dialectical_chess.engine import EngineDecision

    class FakeEngine:
        def __init__(self, settings):
            self.settings = settings

        def choose_move(self, board):
            selected = MoveProbe(
                uci="a1a8",
                san="a1a8",
                score=456,
                is_checkmate=True,
                gives_check=True,
                is_capture=False,
                captured_value=0,
                promotion_value=0,
                reasons=("fake:uci",),
                objections=(),
                optimizer_trace={"status": "optimal", "selected_candidate": "move:a1a8"},
            )
            return EngineDecision(move_uci="a1a8", selected=selected)

    monkeypatch.setattr(uci, "DialecticalChessEngine", FakeEngine)
    output = StringIO()

    move = uci.choose_uci_move(object(), output_stream=output)

    assert move == "a1a8"
    assert "info score cp 456 pv a1a8" in output.getvalue()
    assert "info string optimizer_status=optimal" in output.getvalue()


def test_uci_go_uses_lower_depth_when_clock_is_low() -> None:
    from dialectical_chess.engine import EngineSettings
    from dialectical_chess.probe import owned_board_from_fen
    from dialectical_chess.uci import settings_for_go

    board = owned_board_from_fen("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    settings = EngineSettings(search_depth=2, search_backend="alphabeta")

    adjusted = settings_for_go(settings, board, "go wtime 4500 btime 9000 winc 100 binc 100")

    assert adjusted.search_depth == 1
    assert adjusted.dialectic_depth == 0
    assert adjusted.search_backend == "alphabeta"
    assert not adjusted.reply_mate_scan


def test_uci_go_keeps_depth_when_fast_clock_is_playable() -> None:
    from dialectical_chess.engine import EngineSettings
    from dialectical_chess.probe import owned_board_from_fen
    from dialectical_chess.uci import settings_for_go

    board = owned_board_from_fen("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    settings = EngineSettings(search_depth=2, search_backend="alphabeta")

    adjusted = settings_for_go(settings, board, "go wtime 10000 btime 30000 winc 100 binc 100")

    assert adjusted is settings


def test_uci_go_keeps_depth_at_twenty_five_seconds() -> None:
    from dialectical_chess.engine import EngineSettings
    from dialectical_chess.probe import owned_board_from_fen
    from dialectical_chess.uci import settings_for_go

    board = owned_board_from_fen("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    settings = EngineSettings(search_depth=2, search_backend="alphabeta")

    adjusted = settings_for_go(settings, board, "go wtime 25000 btime 30000 winc 200 binc 200")

    assert adjusted is settings


def test_uci_go_uses_depth_one_when_clock_is_short() -> None:
    from dialectical_chess.engine import EngineSettings
    from dialectical_chess.probe import owned_board_from_fen
    from dialectical_chess.uci import settings_for_go

    board = owned_board_from_fen("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    settings = EngineSettings(search_depth=2, search_backend="alphabeta")

    adjusted = settings_for_go(settings, board, "go wtime 5500 btime 30000 winc 100 binc 100")

    assert adjusted.search_depth == 1
    assert adjusted.dialectic_depth == 0
    assert adjusted.search_backend == "alphabeta"
    assert not adjusted.reply_mate_scan


def test_uci_go_disables_reply_work_when_clock_is_critical() -> None:
    from dialectical_chess.engine import EngineSettings
    from dialectical_chess.probe import owned_board_from_fen
    from dialectical_chess.uci import settings_for_go

    board = owned_board_from_fen("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    settings = EngineSettings(search_depth=2, search_backend="alphabeta")

    adjusted = settings_for_go(settings, board, "go wtime 2400 btime 30000 winc 100 binc 100")

    assert adjusted.search_depth == 0
    assert adjusted.dialectic_depth == 0
    assert adjusted.search_backend == "alphabeta"
    assert not adjusted.smt_mate
    assert not adjusted.smt_fork
    assert not adjusted.positional_reasons
    assert not adjusted.reply_mate_scan


def test_critical_clock_profile_rejects_selected_forced_mate() -> None:
    from dialectical_chess.engine import EngineSettings
    from dialectical_chess.probe import owned_board_from_fen
    from dialectical_chess.uci import choose_uci_move, settings_for_go

    board = owned_board_from_fen("2b2k2/p2n1prp/1rNp2p1/8/4P2P/3B1PP1/P1P1K1q1/8 w - - 1 26")
    settings = settings_for_go(
        EngineSettings(search_depth=2, search_backend="alphabeta"),
        board,
        "go wtime 2400 btime 30000 winc 100 binc 100",
    )

    assert settings.search_depth == 0
    assert not settings.reply_mate_scan
    assert choose_uci_move(board, settings=settings) == "e2e3"


def test_uci_go_uses_lower_depth_when_clock_is_middling() -> None:
    from dialectical_chess.engine import EngineSettings
    from dialectical_chess.probe import owned_board_from_fen
    from dialectical_chess.uci import settings_for_go

    board = owned_board_from_fen("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    settings = EngineSettings(search_depth=2, search_backend="alphabeta")

    adjusted = settings_for_go(settings, board, "go wtime 15000 btime 30000 winc 200 binc 200")

    assert adjusted is settings


def test_uci_go_keeps_depth_when_clock_is_healthy() -> None:
    from dialectical_chess.engine import EngineSettings
    from dialectical_chess.probe import owned_board_from_fen
    from dialectical_chess.uci import settings_for_go

    board = owned_board_from_fen("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    settings = EngineSettings(search_depth=2, search_backend="alphabeta")

    adjusted = settings_for_go(settings, board, "go wtime 28000 btime 30000 winc 200 binc 200")

    assert adjusted is settings
