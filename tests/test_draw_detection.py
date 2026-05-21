from __future__ import annotations

from dialectical_chess.board import OwnedBoard
from dialectical_chess.probe import probe_moves
from dialectical_chess.search import (
    SearchSettings,
    owned_is_draw,
    position_repetition_key,
    root_search_result,
    terminal_or_leaf_result,
)
from dialectical_chess.uci import parse_uci_position_state


def test_known_threefold_sequence_is_detected_from_uci_history() -> None:
    state = parse_uci_position_state(
        "position startpos moves "
        "g1f3 g8f6 f3g1 f6g8 "
        "g1f3 g8f6 f3g1 f6g8"
    )

    assert state.position_history.count(position_repetition_key(state.board)) == 3
    assert owned_is_draw(state.board, position_history=state.position_history)


def test_halfmove_clock_at_one_hundred_is_draw() -> None:
    board = OwnedBoard.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 100 75")

    assert board.is_fifty_move_draw()
    assert owned_is_draw(board)


def test_checkmate_at_one_hundred_halfmoves_is_not_scored_as_draw() -> None:
    board = OwnedBoard.from_fen("7k/5KQ1/8/8/8/8/8/8 b - - 100 75")

    terminal = terminal_or_leaf_result(board, depth=3)

    assert terminal.result is not None
    assert terminal.result.score == -100_003


def test_mate_in_one_on_one_hundredth_halfmove_scores_as_mate() -> None:
    board = OwnedBoard.from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 99 75")
    probes = {
        probe.uci: probe
        for probe in probe_moves(
            board,
            smt_mate=False,
            smt_fork=False,
            reply_mate_scan=False,
        )
    }

    assert probes["g6g7"].score >= 1_000_000
    assert "terminal:checkmate" in probes["g6g7"].reasons
    assert "strategy:fifty_move_draw:g6g7" not in probes["g6g7"].objections


def test_repetition_draw_move_is_not_scored_as_a_win() -> None:
    state = parse_uci_position_state(
        "position startpos moves "
        "g1f3 g8f6 f3g1 f6g8 "
        "g1f3 g8f6 f3g1 f6g8"
    )
    board = state.board
    move = next(move for move in board.legal_moves() if move.uci() == "g1f3")

    search_result = root_search_result(
        board,
        move,
        settings=SearchSettings(depth=1),
        position_history=state.position_history,
    )
    probes = {
        probe.uci: probe
        for probe in probe_moves(
            board,
            smt_mate=False,
            smt_fork=False,
            reply_mate_scan=False,
            position_history=state.position_history,
        )
    }

    assert search_result is not None
    assert search_result.score == 0
    assert probes["g1f3"].score == 0
    assert "strategy:threefold_repetition:g1f3" in probes["g1f3"].objections
