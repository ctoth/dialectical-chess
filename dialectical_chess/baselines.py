"""Baseline engine definitions for dialectical chess match runners."""

from __future__ import annotations

import shutil
from argparse import Namespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fastchess_baseline(baseline: str, uv_executable: str, args: Namespace) -> tuple[str, list[str]]:
    if baseline == "nosmt":
        return (
            "DialecticalNoSMT",
            [
                f"cmd={uv_executable}",
                "args=run dialectical-chess-probe --uci --no-smt-mate",
                "proto=uci",
                f"dir={PROJECT_ROOT}",
            ],
        )
    if baseline == "random":
        return (
            "DialecticalRandom",
            [
                f"cmd={uv_executable}",
                "args=run dialectical-chess-random-uci",
                "proto=uci",
                f"dir={PROJECT_ROOT}",
            ],
        )
    if baseline == "stockfish":
        stockfish = args.stockfish_path or shutil.which("stockfish") or shutil.which("stockfish.exe")
        if stockfish is None:
            raise RuntimeError("stockfish baseline requested but no stockfish executable was found")
        return (
            f"StockfishElo{args.stockfish_elo}",
            [
                f"cmd={stockfish}",
                "proto=uci",
                "option.Threads=1",
                "option.Hash=16",
                "option.UCI_LimitStrength=true",
                f"option.UCI_Elo={args.stockfish_elo}",
            ],
        )
    raise ValueError(f"unknown match baseline: {baseline}")
