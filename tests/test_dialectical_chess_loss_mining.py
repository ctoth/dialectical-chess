from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st


from dialectical_chess.loss_mining import (  # noqa: E402
    LossTurningPoint,
    mine_loss_turning_points,
    reviewed_epd_lines,
)
from dialectical_chess.matches import build_fastchess_command  # noqa: E402


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
    assert "file=scratch\\losses.pgn" in command
    assert "notation=uci" in command
    assert "append=false" in command
    assert "args=run dialectical-chess-probe --uci --dialectic-depth 2 --search-depth 1 --search-backend alphabeta --selector-mode optimizer --reply-max-replies 64 --reply-max-defense-nodes 1000 --reply-min-defense-material 500 --no-smt-fork" in command


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
            suggested_avoid_uci=[],
        )
    ]


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
            suggested_avoid_uci=[],
        )
        for index, move in enumerate(prefix)
    ]

    lines = reviewed_epd_lines(points)

    assert len(lines) == len(prefix)
    for move, line in zip(prefix, lines, strict=True):
        assert f" am {move};" in line
        assert '\\"quoted\\"' in line
