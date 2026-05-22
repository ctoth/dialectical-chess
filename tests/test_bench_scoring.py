from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import chess

from dialectical_chess.arguments import MoveProbe
from dialectical_chess.engine import EngineDecision


def bench_args(**overrides):
    values = {
        "dialectic_depth": 1,
        "search_depth": 0,
        "search_backend": "negamax",
        "smt_mate": True,
        "smt_fork": True,
        "positional_reasons": True,
        "reply_max_replies": 128,
        "reply_max_defense_nodes": 5000,
        "reply_min_defense_material": 300,
        "progress_every": 0,
        "fail_fast": False,
        "epd": None,
        "limit": None,
    }
    values.update(overrides)
    return Namespace(**values)


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
            reasons=("fake:bench",),
            objections=(),
        )
        return EngineDecision(move_uci="a1a8", selected=selected)


def test_run_epd_reports_crashed_position_as_errored_not_failed(monkeypatch, tmp_path) -> None:
    import dialectical_chess.bench_epd as bench_epd

    epd = tmp_path / "suite.epd"
    epd.write_text(
        '7k/6pp/8/8/8/8/6PP/R5K1 w - - bm Ra8#; id "ok";\n'
        '7k/6pp/8/8/8/8/6PP/R5K1 w - - bm Ra8#; id "crash";\n',
        encoding="utf-8",
    )

    def score_or_crash(board, expected_uci, args, *, avoid_uci=None):
        if len(calls) == 1:
            raise RuntimeError("engine crashed")
        calls.append(board)
        return {"correct": True, "avoided": None, "selected_uci": "a1a8"}

    calls = []
    monkeypatch.setattr(bench_epd, "score_board", score_or_crash)

    payload = bench_epd.run_epd(bench_args(epd=epd))

    assert not payload["ok"]
    assert payload["total"] == 2
    assert payload["evaluated"] == 1
    assert payload["errored"] == 1
    assert payload["failed"] == 0
    assert payload["solved"] == 1
    assert payload["hit_rate"] == 1.0


def test_score_board_excludes_avoid_rate_when_suite_has_no_am(monkeypatch) -> None:
    import dialectical_chess.scoring as scoring

    monkeypatch.setattr(scoring, "DialecticalChessEngine", FakeEngine)

    result = scoring.score_board(chess.Board(), {"a1a8"}, bench_args())

    assert result["correct"] is True
    assert result["avoided"] is None


def test_score_board_requires_bm_and_am_when_both_present(monkeypatch) -> None:
    import dialectical_chess.scoring as scoring

    monkeypatch.setattr(scoring, "DialecticalChessEngine", FakeEngine)

    result = scoring.score_board(
        chess.Board(), {"a1a8"}, bench_args(), avoid_uci={"a1a8"}
    )

    assert result["correct"] is False
    assert result["avoided"] is False


def test_read_bestmove_times_out_on_hanging_engine_and_kills_process() -> None:
    """M2 regression: a hung engine must trip the watchdog -- TimeoutError and
    process.kill() -- and the single persistent reader must not leak."""
    import threading

    from dialectical_chess.matches import read_bestmove

    hang_forever = threading.Event()

    class HangingStdout:
        def readline(self):
            # Block as a real stdout would on a non-responsive engine, until
            # the process is killed (kill() releases the wait).
            hang_forever.wait(timeout=10.0)
            return ""

    class FakePopen:
        def __init__(self):
            self.stdout = HangingStdout()
            self.killed = False

        def kill(self):
            self.killed = True
            hang_forever.set()

    process = FakePopen()
    threads_before = threading.active_count()

    try:
        read_bestmove(process, timeout_seconds=0.1)  # type: ignore[arg-type]
        assert False, "expected TimeoutError"
    except TimeoutError:
        pass

    assert process.killed is True
    # The single reader thread observes EOF (kill released the wait) and exits;
    # no orphaned readers accumulate.
    for reader in threading.enumerate():
        if reader is not threading.current_thread() and reader.daemon:
            reader.join(timeout=2.0)
    assert threading.active_count() <= threads_before


def test_read_bestmove_reads_lines_in_stream_order() -> None:
    """M2 regression: a chatty engine emitting many info lines before bestmove
    must be read in stream order by the single persistent reader -- no
    concurrent readline() reordering."""
    from dialectical_chess.matches import read_bestmove

    class ChattyStdout:
        def __init__(self):
            self._lines = iter(
                [
                    "info depth 1 score cp 10\n",
                    "info depth 2 score cp 12\n",
                    "info depth 3 score cp 15\n",
                    "bestmove e2e4 ponder e7e5\n",
                ]
            )

        def readline(self):
            return next(self._lines, "")

    class FakePopen:
        def __init__(self):
            self.stdout = ChattyStdout()
            self.killed = False

        def kill(self):
            self.killed = True

    process = FakePopen()

    move = read_bestmove(process, timeout_seconds=2.0)  # type: ignore[arg-type]

    assert move == "e2e4"
    assert process.killed is False


def test_run_uci_match_treats_unparsed_game_count_as_not_ok(monkeypatch) -> None:
    import dialectical_chess.matches as matches

    def fake_which(name: str):
        return "fastchess" if name == "fastchess" else None

    def fake_run(command, capture_output, text, check):
        return SimpleNamespace(
            returncode=0,
            stdout="Score of Dialectical vs Random: 1 - 0 - 0\n",
            stderr="",
        )

    monkeypatch.setattr(matches.shutil, "which", fake_which)
    monkeypatch.setattr(matches.subprocess, "run", fake_run)
    args = Namespace(
        run_uci_match=True,
        match_baseline="random",
        match_tc="1+0.01",
        match_games=2,
        match_openings="openings.epd",
        match_max_plies=40,
        match_pgn_out=None,
        dialectic_depth=1,
        search_depth=0,
        search_backend="negamax",
        reply_max_replies=128,
        reply_max_defense_nodes=5000,
        reply_min_defense_material=300,
        smt_mate=True,
        smt_fork=True,
        positional_reasons=True,
        stockfish_path=None,
        stockfish_elo=1320,
    )

    payload = matches.run_uci_match(args)

    assert payload["ok"] is False
    assert payload["games_played"] is None
