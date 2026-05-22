"""EPD parser for benchmark suites."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import chess


BUILT_IN_EPD = '7k/6pp/8/8/8/8/6PP/R5K1 w - - bm Ra8#; id "mate-in-one-smoke";'
BM_RE = re.compile(r"\bbm\s+([^;]+);")
AM_RE = re.compile(r"\bam\s+([^;]+);")
ID_RE = re.compile(r"\bid\s+\"([^\"]+)\";")


def read_epd_lines(path: Path | None) -> list[str]:
    if path is None:
        return [BUILT_IN_EPD]
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_epd_case(line: str, *, index: int) -> dict[str, Any]:
    fields = line.split(maxsplit=4)
    if len(fields) < 5:
        raise ValueError(f"invalid EPD line {index}: {line}")
    fen = " ".join(fields[:4] + ["0", "1"])
    board = chess.Board(fen)
    operations = fields[4]
    bm_match = BM_RE.search(operations)
    am_match = AM_RE.search(operations)
    expected = set()
    avoid = set()
    if bm_match is not None:
        expected = {
            parse_expected_move(board, token).uci()
            for token in bm_match.group(1).split()
        }
    if am_match is not None:
        avoid = {
            parse_expected_move(board, token).uci()
            for token in am_match.group(1).split()
        }
    if not expected and not avoid:
        raise ValueError(f"EPD line {index} has no bm or am operation")
    id_match = ID_RE.search(operations)
    return {
        "id": id_match.group(1) if id_match else f"position-{index}",
        "board": board,
        "expected_uci": expected,
        "avoid_uci": avoid,
    }


def parse_expected_move(board: chess.Board, token: str) -> chess.Move:
    try:
        move = chess.Move.from_uci(token)
    except ValueError:
        move = chess.Move.null()
    if move in board.legal_moves:
        return move
    return board.parse_san(token)
