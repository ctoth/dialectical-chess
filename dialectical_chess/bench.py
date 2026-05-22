"""Benchmark CLI dispatcher and compatibility exports."""

from __future__ import annotations

from typing import Any

from dialectical_chess import bench_epd as _bench_epd
from dialectical_chess import scoring as _scoring
from dialectical_chess.bench_epd import *  # noqa: F403
from dialectical_chess.bench_lichess import *  # noqa: F403
from dialectical_chess.bench_matrix import *  # noqa: F403
from dialectical_chess.epd import *  # noqa: F403
from dialectical_chess.scoring import *  # noqa: F403


DialecticalChessEngine = _scoring.DialecticalChessEngine
_ORIGINAL_SCORE_BOARD = _scoring.score_board
_ORIGINAL_RUN_EPD = _bench_epd.run_epd


def score_board(*args: Any, **kwargs: Any) -> dict[str, Any]:
    _scoring.DialecticalChessEngine = DialecticalChessEngine
    return _ORIGINAL_SCORE_BOARD(*args, **kwargs)


def run_epd(*args: Any, **kwargs: Any) -> dict[str, Any]:
    previous = _bench_epd.score_board
    _bench_epd.score_board = score_board
    try:
        return _ORIGINAL_RUN_EPD(*args, **kwargs)
    finally:
        _bench_epd.score_board = previous


def main() -> int:
    previous = _bench_epd.score_board
    _bench_epd.score_board = score_board
    try:
        return _bench_epd.main()
    finally:
        _bench_epd.score_board = previous
