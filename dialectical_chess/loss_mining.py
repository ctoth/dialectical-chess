"""Loss turning-point mining for dialectical chess match PGNs."""

from __future__ import annotations

import io
from dataclasses import dataclass

import chess
import chess.pgn


@dataclass(frozen=True)
class LossTurningPoint:
    game_index: int
    ply: int
    fen_before: str
    played_move: str
    side_to_move: str
    result: str
    reason: str
    suggested_avoid_uci: list[str]


def mine_loss_turning_points(
    pgn_text: str,
    *,
    engine_name: str,
    mate_depth: int = 1,
) -> list[LossTurningPoint]:
    if mate_depth < 1:
        raise ValueError("mate_depth must be at least 1")
    points: list[LossTurningPoint] = []
    stream = io.StringIO(pgn_text)
    game_index = 0
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            return points
        game_index += 1
        engine_color = engine_color_for_loss(game, engine_name)
        if engine_color is None:
            continue
        board = game.board()
        for ply, move in enumerate(game.mainline_moves(), start=1):
            fen_before = board.fen()
            mover = board.turn
            move_uci = move.uci()
            board.push(move)
            if mover == engine_color and has_forced_mate(board, mate_depth=mate_depth):
                points.append(
                    LossTurningPoint(
                        game_index=game_index,
                        ply=ply,
                        fen_before=fen_before,
                        played_move=move_uci,
                        side_to_move="w" if engine_color == chess.WHITE else "b",
                        result=game.headers.get("Result", "*"),
                        reason=f"allows_mate_in_{mate_depth}",
                        suggested_avoid_uci=safe_legal_moves(
                            fen_before,
                            mate_depth=mate_depth,
                        ),
                    )
                )
                break


def engine_color_for_loss(game: chess.pgn.Game, engine_name: str) -> bool | None:
    result = game.headers.get("Result", "*")
    if game.headers.get("White") == engine_name and result == "0-1":
        return chess.WHITE
    if game.headers.get("Black") == engine_name and result == "1-0":
        return chess.BLACK
    return None


def has_immediate_mate(board: chess.Board) -> bool:
    return has_forced_mate(board, mate_depth=1)


def has_forced_mate(board: chess.Board, *, mate_depth: int) -> bool:
    """Return whether the side to move can force mate within mate_depth moves."""
    if mate_depth < 1:
        raise ValueError("mate_depth must be at least 1")
    if board.is_checkmate():
        return True

    for move in board.legal_moves:
        attacker_child = board.copy(stack=False)
        attacker_child.push(move)
        if attacker_child.is_checkmate():
            return True
        if mate_depth == 1:
            continue
        defender_replies = list(attacker_child.legal_moves)
        if not defender_replies:
            continue
        if all(
            has_forced_mate(defender_child(attacker_child, reply), mate_depth=mate_depth - 1)
            for reply in defender_replies
        ):
            return True
    return False


def defender_child(board: chess.Board, reply: chess.Move) -> chess.Board:
    child = board.copy(stack=False)
    child.push(reply)
    return child


def safe_legal_moves(fen_before: str, *, mate_depth: int) -> list[str]:
    board = chess.Board(fen_before)
    safe_moves = []
    for move in board.legal_moves:
        child = board.copy(stack=False)
        child.push(move)
        if not has_forced_mate(child, mate_depth=mate_depth):
            safe_moves.append(move.uci())
    return safe_moves


def reviewed_epd_lines(points: list[LossTurningPoint]) -> list[str]:
    lines = []
    for point in points:
        epd_position = " ".join(point.fen_before.split()[:4])
        reason = escape_epd_string(point.reason)
        fields = [f"am {point.played_move}"]
        if point.suggested_avoid_uci:
            fields.append(f"bm {' '.join(point.suggested_avoid_uci)}")
        fields.append(f'id "loss-{point.game_index}-ply-{point.ply}: {reason}"')
        lines.append(f"{epd_position} " + "; ".join(fields) + ";")
    return lines


def escape_epd_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')
