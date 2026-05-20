from __future__ import annotations

from argparse import Namespace
from io import StringIO

import pytest


from dialectical_chess.arguments import (  # noqa: E402
    MoveProbe,
    build_root_argument_graph,
    choose_move,
)
from dialectical_chess.baselines import fastchess_baseline  # noqa: E402


def test_uci_position_parses_startpos_moves() -> None:
    pytest.importorskip("chess")
    from dialectical_chess.uci import parse_uci_position

    board = parse_uci_position("position startpos moves e2e4 e7e5")

    assert board.fen() == "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"


def test_uci_position_tracks_recent_own_move() -> None:
    pytest.importorskip("chess")
    from dialectical_chess.uci import parse_uci_position_state

    board, recent_own_move = parse_uci_position_state("position startpos moves e2e4 e7e5 g1f3 b8c6")

    assert board.fen() == "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    assert recent_own_move == "g1f3"


def test_uci_loop_keeps_recent_own_move_across_fen_only_updates(monkeypatch) -> None:
    pytest.importorskip("chess")
    import dialectical_chess.uci as uci

    seen_recent_own_moves: list[str | None] = []
    chosen_moves = iter(("a5c6", "c6a5"))

    def fake_choose_uci_move(board, *, settings, output_stream):
        seen_recent_own_moves.append(settings.recent_own_move)
        return next(chosen_moves)

    monkeypatch.setattr(uci, "choose_uci_move", fake_choose_uci_move)

    input_stream = StringIO(
        "\n".join(
            (
                "position fen r2qk2r/pppbn1pp/1b2p3/n2pPpN1/1P1P4/2PB4/3N1PPP/1RBQK2R b Kkq - 1 12",
                "go",
                "position fen r2qk2r/pppbn1pp/1bn1p3/1P1pPpN1/3P4/2PB4/3N1PPP/1RBQK2R b Kkq - 0 13",
                "go",
                "quit",
                "",
            )
        )
    )
    output_stream = StringIO()

    assert uci.run_uci(input_stream, output_stream) == 0
    assert seen_recent_own_moves == [None, "a5c6"]
    assert output_stream.getvalue().splitlines() == ["bestmove a5c6", "bestmove c6a5"]


def test_epd_parses_best_and_avoid_moves() -> None:
    pytest.importorskip("chess")
    from dialectical_chess.bench import parse_epd_case

    case = parse_epd_case(
        '7k/6pp/8/8/8/8/6PP/R5K1 w - - bm Ra8#; am h2h3; id "smoke";',
        index=1,
    )

    assert case["id"] == "smoke"
    assert case["expected_uci"] == {"a1a8"}
    assert case["avoid_uci"] == {"h2h3"}


def test_stockfish_baseline_command_uses_strength_limit() -> None:
    args = Namespace(stockfish_path="stockfish", stockfish_elo=1320)

    name, command = fastchess_baseline("stockfish", "uv", args)

    assert name == "StockfishElo1320"
    assert "option.UCI_LimitStrength=true" in command
    assert "option.UCI_Elo=1320" in command


def test_argument_selection_prefers_supported_move_before_score_fallback() -> None:
    supported = MoveProbe(
        uci="a1a8",
        san="a1a8",
        score=10,
        is_checkmate=False,
        gives_check=False,
        is_capture=False,
        captured_value=0,
        promotion_value=0,
        reasons=("terminal:checkmate",),
        objections=(),
    )
    unsupported = MoveProbe(
        uci="h2h3",
        san="h2h3",
        score=9999,
        is_checkmate=False,
        gives_check=False,
        is_capture=False,
        captured_value=0,
        promotion_value=0,
        reasons=(),
        objections=("objection:no_immediate_tactical_warrant",),
    )

    graph = build_root_argument_graph([supported, unsupported])

    assert choose_move([supported, unsupported], graph) == supported


def test_no_legal_moves_returns_uci_null_move() -> None:
    pytest.importorskip("chess")
    from dialectical_chess.uci import choose_uci_move, parse_uci_position

    board = parse_uci_position("position fen 7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")

    assert choose_uci_move(board) == "0000"
