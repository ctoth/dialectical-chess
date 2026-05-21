from __future__ import annotations

from io import StringIO

import pytest

from dialectical_chess.board import OwnedBoard, OwnedMove
from dialectical_chess.uci import parse_uci_position_state, run_uci


@pytest.mark.parametrize(
    "text,match",
    [
        ("e2e", "invalid UCI move"),
        ("e2e4qq", "invalid UCI move"),
        ("e2e4k", "invalid promotion piece"),
        ("i2e4", "invalid square"),
        ("e9e4", "invalid square"),
    ],
)
def test_owned_move_from_uci_rejects_malformed_moves(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        OwnedMove.from_uci(text)


@pytest.mark.parametrize(
    "fen,match",
    [
        ("8/8/8/8/8/8/8/8 w - - 0", "six fields"),
        ("8/8/8/8/8/8/8/8 x - - 0 1", "side-to-move"),
        ("8/8/8/8/8/8/8/8 w KK - 0 1", "castling"),
        ("8/8/8/8/8/8/8/8 w A - 0 1", "castling"),
        ("8/8/8/8/8/8/8/8 w - e4 0 1", "en-passant"),
        ("8/8/8/8/8/8/8/8 w - - x 1", "clocks"),
        ("8/8/8/8/8/8/8/8 w - - -1 1", "clocks"),
        ("8/8/8/8/8/8/8/8 w - - 0 0", "clocks"),
        ("8/8/8/8/8/8/8 w - - 0 1", "eight ranks"),
        ("8/8/8/8/8/8/8/9 w - - 0 1", "eight files"),
        ("8/8/8/8/8/8/8/09 w - - 0 1", "rank digit"),
        ("8/8/8/8/8/8/8/1X6 w - - 0 1", "unknown FEN piece"),
        ("8/8/8/8/8/8/8/R7R w - - 0 1", "too many files"),
        ("8/8/8/8/8/8/8/7 w - - 0 1", "eight files"),
        ("8/8/8/8/8/8/8/8 w - - 0 1", "exactly one king"),
    ],
)
def test_owned_board_from_fen_rejects_documented_errors(fen: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        OwnedBoard.from_fen(fen)


@pytest.mark.parametrize(
    "command,match",
    [
        ("fen 8/8/8/8/8/8/8/8 w - - 0 1", "fen "),
        ("position", "position"),
        ("position garbage", "startpos or fen"),
        ("position fen 8/8/8/8/8/8/8/8 w - - 0", "six FEN fields"),
        ("position startpos garbage", "unexpected token"),
        ("position startpos moves e2e5", "illegal move"),
    ],
)
def test_parse_uci_position_state_rejects_malformed_position_commands(command: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_uci_position_state(command)


def test_uci_reports_invalid_position_and_continues_loop() -> None:
    output = StringIO()
    commands = StringIO(
        "position fen 8/8/8/8/8/8/8/8 w - - 0\n"
        "isready\n"
        "quit\n"
    )

    assert run_uci(commands, output) == 0
    text = output.getvalue()
    assert "info string invalid position" in text
    assert "readyok" in text
