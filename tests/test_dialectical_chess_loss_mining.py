from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import chess
import pytest
from hypothesis import given
from hypothesis import strategies as st


from dialectical_chess.loss_mining import (  # noqa: E402
    LossTurningPoint,
    has_forced_mate,
    mine_loss_turning_points,
    reviewed_epd_lines,
    safe_legal_moves,
)
from dialectical_chess.pgn_diagnostics import pgn_positions  # noqa: E402
import dialectical_chess.matches as matches  # noqa: E402
from dialectical_chess.matches import PROJECT_ROOT, build_fastchess_command, prepare_match_outputs  # noqa: E402


def test_fastchess_command_can_emit_diagnostic_pgn() -> None:
    args = Namespace(
        match_baseline="stockfish",
        match_games=2,
        match_max_plies=400,
        match_openings=Path("dialectical_chess/fixtures/dialectical_chess_openings.epd"),
        match_pgn_out=Path("scratch/losses.pgn"),
        match_tc="30+0.2",
        stockfish_elo=1320,
        stockfish_path="stockfish",
        dialectic_depth=2,
        search_depth=1,
        search_backend="alphabeta",
        selector_mode="optimizer",
        reply_max_replies=64,
        reply_max_defense_nodes=1000,
        reply_min_defense_material=500,
        smt_mate=True,
        smt_fork=False,
        positional_reasons=True,
    )

    command = build_fastchess_command(args, fastchess="fast-chess", uv_executable="uv")

    assert "-pgnout" in command
    assert f"file={PROJECT_ROOT / 'scratch' / 'losses.pgn'}" in command
    assert "notation=uci" in command
    assert "append=false" in command
    assert "args=run dialectical-chess-probe --uci --dialectic-depth 2 --search-depth 1 --search-backend alphabeta --selector-mode optimizer --reply-max-replies 64 --reply-max-defense-nodes 1000 --reply-min-defense-material 500 --no-smt-fork" in command


def test_prepare_match_outputs_creates_relative_pgn_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(matches, "PROJECT_ROOT", tmp_path)
    args = Namespace(match_pgn_out=Path("scratch/losses.pgn"))

    prepare_match_outputs(args)

    assert (tmp_path / "scratch").is_dir()


def test_pgn_diagnostics_filters_engine_positions() -> None:
    pgn = """
[Event "diagnostic"]
[White "StockfishElo2000"]
[Black "Dialectical"]
[Result "1-0"]

1. e4 e6 2. Qh5 Qh4 1-0
"""

    payload = pgn_positions(pgn, engine_name="Dialectical", engine_only=True)

    assert payload == {
        "positions": [
            {
                "game_index": 1,
                "ply": 2,
                "mover": "b",
                "engine_to_move": True,
                "fen_before": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                "move_uci": "e7e6",
                "comment": "",
            },
            {
                "game_index": 1,
                "ply": 4,
                "mover": "b",
                "engine_to_move": True,
                "fen_before": "rnbqkbnr/pppp1ppp/4p3/7Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2",
                "move_uci": "d8h4",
                "comment": "",
            },
        ]
    }


def test_mines_first_engine_move_that_allows_immediate_mate() -> None:
    pgn = """
[Event "loss"]
[White "Dialectical"]
[Black "StockfishElo1320"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1
"""

    points = mine_loss_turning_points(pgn, engine_name="Dialectical", mate_depth=1)

    assert points == [
        LossTurningPoint(
            game_index=1,
            ply=3,
            fen_before="rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2",
            played_move="g2g4",
            side_to_move="w",
            result="0-1",
            reason="allows_mate_in_1",
            suggested_avoid_uci=[
                "g1h3",
                "e1f2",
                "b1c3",
                "b1a3",
                "f3f4",
                "h2h3",
                "g2g3",
                "e2e3",
                "d2d3",
                "c2c3",
                "b2b3",
                "a2a3",
                "h2h4",
                "e2e4",
                "d2d4",
                "c2c4",
                "b2b4",
                "a2a4",
            ],
        )
    ]


def test_mines_first_engine_move_that_allows_forced_mate_in_two() -> None:
    pgn = """
[Event "loss"]
[White "Dialectical"]
[Black "StockfishElo2000"]
[Result "0-1"]
[SetUp "1"]
[FEN "2kr1bnr/1p3ppp/p7/3N1b1Q/P3nP2/2B5/2P2qPP/R3KBNR w - - 4 17"]

17. Kd1 0-1
"""

    points = mine_loss_turning_points(pgn, engine_name="Dialectical", mate_depth=2)

    assert points == [
        LossTurningPoint(
            game_index=1,
            ply=1,
            fen_before="2kr1bnr/1p3ppp/p7/3N1b1Q/P3nP2/2B5/2P2qPP/R3KBNR w - - 4 17",
            played_move="e1d1",
            side_to_move="w",
            result="0-1",
            reason="allows_mate_in_2",
            suggested_avoid_uci=[],
        )
    ]


def test_forced_mate_depth_requires_defender_coverage() -> None:
    board = chess.Board("2kr1bnr/1p3ppp/p7/3N1b1Q/P3nP2/2B5/2P2qPP/R2K1BNR b - - 5 17")

    assert not has_forced_mate(board, mate_depth=1)
    assert has_forced_mate(board, mate_depth=2)


def test_safe_legal_moves_excludes_moves_allowing_forced_mate() -> None:
    safe_moves = safe_legal_moves(
        "4kbnr/3p1ppp/2pP4/q3P3/8/PQN2N1P/5PP1/RBB1R1K1 b k - 0 23",
        mate_depth=2,
    )

    assert "c6c5" in safe_moves
    assert "a5c5" not in safe_moves


@given(
    prefix=st.lists(
        st.sampled_from(["a2a3", "a2a4", "b2b3", "b2b4", "g2g3", "h2h3"]),
        min_size=0,
        max_size=3,
        unique=True,
    )
)
def test_reviewed_epd_lines_escape_ids_and_encode_avoid_moves(prefix: list[str]) -> None:
    points = [
        LossTurningPoint(
            game_index=index + 1,
            ply=1,
            fen_before="8/8/8/8/8/8/8/K6k w - - 0 1",
            played_move=move,
            side_to_move="w",
            result="0-1",
            reason='bad "quoted" move',
            suggested_avoid_uci=["b2b3", "g2g3"],
        )
        for index, move in enumerate(prefix)
    ]

    lines = reviewed_epd_lines(points)

    assert len(lines) == len(prefix)
    for move, line in zip(prefix, lines, strict=True):
        assert f" am {move};" in line
        assert " bm b2b3 g2g3;" in line
        assert '\\"quoted\\"' in line
