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
        choose_move([])


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
    import dialectical_chess.scoring as scoring
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

    monkeypatch.setattr(scoring, "DialecticalChessEngine", FakeEngine)

    args = Namespace(
        dialectic_depth=1,
        search_depth=0,
        search_backend="negamax",
        smt_mate=True,
        positional_reasons=True,
    )

    result = scoring.score_board(chess.Board(), {"a1a8"}, args)

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


def test_critical_clock_profile_rejects_selected_forced_mate() -> None:
    """M4 restored behaviour test (ported to parse_go / settings_for_go_request):
    under a critical-clock budget the real engine still selects the sound move
    e2e3 -- the alternative e2e1 allows a reply forced mate in 2. Pins the
    rewritten time-control profile's move selection.

    The move-selection assertion drops the wall-clock `deadline` that
    settings_for_go_request stamps in: the budget is set at parse time, so a
    loaded machine could let it expire mid-probe and the deadline-bounded
    reply-mate search (M3) would then miss the refuting objection. That
    deadline-expiry behaviour is M3's own concern, covered separately by
    test_has_forced_mate_returns_best_so_far_on_expired_deadline. This test
    pins the *profile's decision*, which must be deterministic."""
    from dataclasses import replace

    from dialectical_chess.engine import EngineSettings
    from dialectical_chess.probe import owned_board_from_fen
    from dialectical_chess.uci import choose_uci_move, parse_go, settings_for_go_request

    board = owned_board_from_fen("2b2k2/p2n1prp/1rNp2p1/8/4P2P/3B1PP1/P1P1K1q1/8 w - - 1 26")
    request = parse_go("go wtime 1500 btime 30000 winc 100 binc 100".split())
    settings, budget_ms = settings_for_go_request(
        EngineSettings(search_depth=2, search_backend="alphabeta"),
        board,
        request,
    )

    # The critical-clock budget profile: depth dropped to 0, reply work off.
    assert budget_ms is not None and budget_ms <= 100
    assert settings.search_depth == 0
    assert settings.dialectic_depth == 0
    assert not settings.reply_mate_scan
    assert not settings.positional_reasons
    assert choose_uci_move(board, settings=replace(settings, deadline=None)) == "e2e3"


def test_critical_clock_profile_bounds_selected_forced_mate_in_wide_positions(monkeypatch) -> None:
    """M4 restored behaviour test (ported to parse_go / settings_for_go_request):
    in a wide position the critical-clock profile must NOT run an unbounded
    forced-mate proof, and the engine must still return a non-null move."""
    from dialectical_chess import engine as engine_module
    from dialectical_chess.engine import EngineSettings
    from dialectical_chess.probe import owned_board_from_fen
    from dialectical_chess.uci import choose_uci_move, parse_go, settings_for_go_request

    from dataclasses import replace

    def reject_unbounded_mate_search(*args, **kwargs) -> bool:
        raise AssertionError(
            "critical profile should not run a selected forced-mate proof in wide positions"
        )

    monkeypatch.setattr(engine_module, "has_forced_mate", reject_unbounded_mate_search)
    board = owned_board_from_fen("1rb1kr2/pp1p2pp/2nQ2n1/b5B1/5p2/2P2N2/PPK2P1P/R4B1R b - - 3 25")
    request = parse_go("go btime 1500 wtime 30000 binc 100 winc 100".split())
    settings, budget_ms = settings_for_go_request(
        EngineSettings(search_depth=2, search_backend="alphabeta"),
        board,
        request,
    )

    assert budget_ms is not None and budget_ms <= 100
    assert settings.search_depth == 0
    assert not settings.reply_mate_scan
    assert choose_uci_move(board, settings=replace(settings, deadline=None)) != "0000"


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


def test_uci_go_infinite_with_clock_does_not_self_terminate(monkeypatch) -> None:
    """M1 regression: `go infinite` with wtime/btime present must search until
    `stop`, not self-terminate on a budget. settings_for_go_request computed a
    deadline regardless of request.infinite, so the infinite search ended in
    ~10ms. Assert exactly one bestmove, emitted only after `stop`, and that the
    settings carried into the search have no deadline."""
    import threading

    import dialectical_chess.uci as uci
    from dialectical_chess.arguments import MoveProbe
    from dialectical_chess.engine import EngineDecision

    release = threading.Event()
    seen_deadlines = []

    class FakeEngine:
        def __init__(self, settings):
            seen_deadlines.append(settings.deadline)

        def choose_move(self, board):
            # Block until `stop` joins the search thread -- a real infinite
            # search runs until interrupted, never finishing on its own.
            release.wait(timeout=5.0)
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

    original_finish = uci.finish_search

    def finish_then_check(active_search):
        # The search must still be running when `stop` arrives: a self-
        # terminating budget would have produced a bestmove already.
        assert output.getvalue().count("bestmove ") == 0
        release.set()
        return original_finish(active_search)

    monkeypatch.setattr(uci, "finish_search", finish_then_check)

    assert uci.run_uci(StringIO("go infinite wtime 1000 btime 1000\nstop\nquit\n"), output) == 0

    text = output.getvalue()
    assert text.count("bestmove ") == 1
    assert "bestmove a2a3" in text
    # An infinite search carries no deadline even with a clock present.
    assert seen_deadlines == [None]


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


def test_has_forced_mate_returns_best_so_far_on_expired_deadline() -> None:
    """M3 regression: a single heavy mate search must be deadline-bounded.

    Without the deadline threaded into the recursion, one has_forced_mate call
    can run for seconds and a top-of-loop deadline check cannot interrupt it.
    A deadline already in the past makes the search return best-so-far (no
    proven mate => False) immediately rather than fully expanding the tree."""
    import time

    import chess

    from dialectical_chess.loss_mining import FORCED_MATE_CACHE, has_forced_mate

    FORCED_MATE_CACHE.clear()
    # A wide opening position: a depth-4 forced-mate proof here would expand a
    # large tree. With an already-expired deadline the search must bail out.
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    started = time.perf_counter()
    result = has_forced_mate(board, mate_depth=4, deadline=time.monotonic() - 1.0)
    elapsed = time.perf_counter() - started

    assert result is False
    assert elapsed < 0.5
    FORCED_MATE_CACHE.clear()


def test_reply_mate_fixpoint_bounds_single_iteration_with_deadline(monkeypatch) -> None:
    """M3 regression: the reply-mate fixpoint's heavy per-iteration call
    (selected_reply_mate_refutation -> has_forced_mate) must observe the
    deadline. A slow has_forced_mate must not overrun an expired budget."""
    import time

    from dialectical_chess import engine as engine_module
    from dialectical_chess.arguments import MoveProbe
    from dialectical_chess.engine import selected_reply_mate_refutation_fixpoint
    from dialectical_chess.probe import owned_board_from_fen

    board = owned_board_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    selected = MoveProbe(
        uci="e2e4",
        san="e2e4",
        score=0,
        is_checkmate=False,
        gives_check=False,
        is_capture=False,
        captured_value=0,
        promotion_value=0,
        reasons=(),
        objections=(),
    )

    deadline_seen = []

    def slow_has_forced_mate(child, *, mate_depth, deadline=None):
        deadline_seen.append(deadline)
        # The deadline must be threaded all the way into the mate search.
        assert deadline is not None
        return False

    monkeypatch.setattr(engine_module, "has_forced_mate", slow_has_forced_mate)

    probes = [selected]
    started = time.perf_counter()
    out_probes, out_selected = selected_reply_mate_refutation_fixpoint(
        board,
        probes,
        selected,
        allow_mate_four=True,
        deadline=time.monotonic() + 0.05,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0
    assert out_selected is not None
    # The deadline reached the heavy mate call, not just the loop top.
    assert deadline_seen and all(d is not None for d in deadline_seen)
