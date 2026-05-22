"""Benchmark CLI dispatcher and compatibility exports."""

from __future__ import annotations

from typing import Any

from dialectical_chess import bench_epd as _impl
from dialectical_chess.bench_epd import *  # noqa: F403

DialecticalChessEngine = _impl.DialecticalChessEngine
_ORIGINAL_SCORE_BOARD = _impl.score_board
_ORIGINAL_RUN_EPD = _impl.run_epd


def score_board(*args: Any, **kwargs: Any) -> dict[str, Any]:
    _impl.DialecticalChessEngine = DialecticalChessEngine
    return _ORIGINAL_SCORE_BOARD(*args, **kwargs)


def run_epd(*args: Any, **kwargs: Any) -> dict[str, Any]:
    previous = _impl.score_board
    _impl.score_board = score_board
    try:
        return _ORIGINAL_RUN_EPD(*args, **kwargs)
    finally:
        _impl.score_board = previous


def main() -> int:
    previous = _impl.score_board
    _impl.score_board = score_board
    try:
        return _impl.main()
    finally:
        _impl.score_board = previous
