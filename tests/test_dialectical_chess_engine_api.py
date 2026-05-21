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
        "positional_reasons": True,
        "reply_mate_scan": True,
        "reply_analysis": {
            "max_replies": 128,
            "max_defense_nodes": 5000,
            "min_defense_material": 300,
        },
        "position_history": (),
        "deadline": None,
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

    board = owned_board_from_fen("7k/5KQ1/8/8/8/8/8/8 b - - 0 1")

    decision = DialecticalChessEngine().choose_move(board)

    assert decision.move_uci == "0000"
    assert decision.selected is None


def test_choose_move_raises_value_error_for_empty_probe_list() -> None:
    import pytest

    from dialectical_chess.arguments import choose_move

    with pytest.raises(ValueError, match="position has no legal moves"):
        choose_move([], None)


def test_uci_no_legal_move_position_survives_and_returns_null_move() -> None:
    from dialectical_chess.uci import run_uci

    output = StringIO()
    input_stream = StringIO(
        "position fen 7k/5KQ1/8/8/8/8/8/8 b - - 0 1\n"
        "go\n"
        "quit\n"
    )

    assert run_uci(input_stream, output) == 0
    assert "bestmove 0000" in output.getvalue()


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
            )
            return EngineDecision(move_uci="a1a8", selected=selected)

    monkeypatch.setattr(bench, "DialecticalChessEngine", FakeEngine)

    args = Namespace(
        dialectic_depth=1,
        search_depth=0,
        search_backend="negamax",
        smt_mate=True,
        positional_reasons=True,
    )

    result = bench.score_board(chess.Board(), {"a1a8"}, args)

    assert result["selected_uci"] == "a1a8"
    assert result["reasons"] == ["fake:engine"]


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
            )
            return EngineDecision(move_uci="a1a8", selected=selected)

    monkeypatch.setattr(uci, "DialecticalChessEngine", FakeEngine)
    output = StringIO()

    move = uci.choose_uci_move(object(), output_stream=output)

    assert move == "a1a8"
    assert "info score cp 456 pv a1a8" in output.getvalue()


def test_uci_go_movetime_returns_within_budget_tolerance() -> None:
    import time

    from dialectical_chess.uci import run_uci

    output = StringIO()
    input_stream = StringIO(
        "position fen 7k/6pp/8/8/8/8/6PP/R5K1 w - - 0 1\n"
        "go movetime 200\n"
        "quit\n"
    )

    started = time.perf_counter()
    assert run_uci(input_stream, output) == 0

    assert time.perf_counter() - started < 0.5
    assert output.getvalue().count("bestmove ") == 1


def test_uci_go_low_clock_budget_zero_returns_legal_bestmove() -> None:
    """C1 regression: the budget<=0 low-clock fallback must not crash the UCI
    process. best_available_move treated OwnedBoard.legal_moves as a property;
    it is a method. go wtime 100 / 50 / 10 all drive budget_ms <= 0."""
    import chess

    from dialectical_chess.uci import run_uci

    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    for clock in (100, 50, 10):
        output = StringIO()
        input_stream = StringIO(
            f"position fen {start_fen}\n"
            f"go wtime {clock} btime {clock}\n"
            "quit\n"
        )

        assert run_uci(input_stream, output) == 0

        text = output.getvalue()
        assert text.count("bestmove ") == 1, (clock, text)
        bestmove = text.split("bestmove ")[1].split()[0]
        legal = {move.uci() for move in chess.Board(start_fen).legal_moves}
        assert bestmove in legal, (clock, bestmove)


def test_uci_go_depth_is_honored(monkeypatch) -> None:
    import dialectical_chess.uci as uci
    from dialectical_chess.arguments import MoveProbe
    from dialectical_chess.engine import EngineDecision

    seen_depths = []

    class FakeEngine:
        def __init__(self, settings):
            seen_depths.append(settings.search_depth)

        def choose_move(self, board):
            selected = MoveProbe(
                uci="a2a3",
                san="a2a3",
                score=1,
                is_checkmate=False,
                gives_check=False,
                is_capture=False,
                captured_value=0,
                promotion_value=0,
                reasons=("fake:depth",),
                objections=(),
            )
            return EngineDecision(move_uci="a2a3", selected=selected)

    monkeypatch.setattr(uci, "DialecticalChessEngine", FakeEngine)
    output = StringIO()

    assert uci.run_uci(StringIO("go depth 2\nquit\n"), output) == 0

    assert seen_depths == [2]
    assert output.getvalue().count("bestmove ") == 1


def test_uci_go_infinite_stop_returns_exactly_one_bestmove(monkeypatch) -> None:
    import dialectical_chess.uci as uci
    from dialectical_chess.arguments import MoveProbe
    from dialectical_chess.engine import EngineDecision

    class FakeEngine:
        def __init__(self, settings):
            self.settings = settings

        def choose_move(self, board):
            selected = MoveProbe(
                uci="a2a3",
                san="a2a3",
                score=1,
                is_checkmate=False,
                gives_check=False,
                is_capture=False,
                captured_value=0,
                promotion_value=0,
                reasons=("fake:infinite",),
                objections=(),
            )
            return EngineDecision(move_uci="a2a3", selected=selected)

    monkeypatch.setattr(uci, "DialecticalChessEngine", FakeEngine)
    output = StringIO()

    assert uci.run_uci(StringIO("go infinite\nstop\nquit\n"), output) == 0

    text = output.getvalue()
    assert text.count("bestmove ") == 1
    assert "bestmove a2a3" in text


def test_low_budget_probe_stops_after_best_so_far_plus_one(monkeypatch) -> None:
    import dialectical_chess.probe as probe_module
    from dialectical_chess.probe import owned_board_from_fen, probe_moves

    ticks = iter((0.0, 1.0, 2.0, 3.0))

    monkeypatch.setattr(probe_module.time, "monotonic", lambda: next(ticks, 3.0))

    board = owned_board_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    probes = probe_moves(
        board,
        dialectic_depth=0,
        search_depth=0,
        smt_mate=False,
        smt_fork=False,
        positional_reasons=False,
        reply_mate_scan=False,
        deadline=0.5,
    )

    assert 1 <= len(probes) <= 2
