"""Loss turning-point mining for dialectical chess match PGNs.

Chess cartridge layer over ``dialectical_games.loss_mining``. Keeps the chess-
specific ``has_forced_mate`` prover and the EPD review tooling; delegates the
turning-point algorithm to the core.

The chess-local ``LossTurningPoint`` / ``mine_loss_turning_points`` /
``engine_color_for_loss`` are gone; callers use the core
``dialectical_games.loss_mining.LossTurningPoint`` and the
``mine_loss_turning_points`` thin wrapper below, which adapts a chess PGN
stream into a sequence of core-shaped :class:`GameResult` and invokes
``dialectical_games.loss_mining.mine_turning_point`` / ``mine_losses``.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any, Hashable

import chess
import chess.pgn

from dialectical_games.loss_mining import (
    LossTurningPoint,
    mine_losses,
    mine_turning_point,
)

from dialectical_chess.board import OwnedBoard

FORCED_MATE_CACHE: dict[tuple[Hashable, int], bool] = {}

# Chess outcome strings the cartridge passes to the core.
WHITE_WIN_OUTCOME = "1-0"
BLACK_WIN_OUTCOME = "0-1"
DRAW_OUTCOME = "1/2-1/2"
UNKNOWN_OUTCOME = "*"


@dataclass(frozen=True)
class ChessGameResult:
    """``GameResult`` Protocol adapter — one PGN game as
    ``(outcome, moves, positions)``.
    """

    outcome: str
    moves: tuple[Any, ...]
    positions: tuple[Any, ...]


def _owned_board_from_pychess(board: chess.Board) -> OwnedBoard:
    """Convert a ``chess.Board`` to a chess-cartridge ``OwnedBoard``."""
    return OwnedBoard.from_fen(board.fen(), legal_game=False)


def _owned_move_from_pychess(board: chess.Board, move: chess.Move) -> Any:
    """Convert a ``chess.Move`` (uci) to a chess-cartridge ``OwnedMove``."""
    from dialectical_chess.board import OwnedMove

    return OwnedMove.from_uci(move.uci())


def chess_game_result_from_pgn(game: chess.pgn.Game) -> ChessGameResult:
    """Build a Protocol-shaped :class:`ChessGameResult` from a PGN game."""
    outcome = game.headers.get("Result", UNKNOWN_OUTCOME)
    moves: list[Any] = []
    positions: list[Any] = []
    board = game.board()
    positions.append(_owned_board_from_pychess(board))
    for move in game.mainline_moves():
        moves.append(_owned_move_from_pychess(board, move))
        board.push(move)
        positions.append(_owned_board_from_pychess(board))
    return ChessGameResult(
        outcome=outcome,
        moves=tuple(moves),
        positions=tuple(positions),
    )


def engine_color_for_loss(game: chess.pgn.Game, engine_name: str) -> str | None:
    """Return the engine's side ("w" / "b") iff the engine **lost** this game.

    Mirrors the prior chess-local helper but returns the canonical side letter
    that ``OwnedBoard.turn`` uses, so the cartridge can pass it as
    ``engine_side`` to the core's ``mine_turning_point``.
    """
    result = game.headers.get("Result", UNKNOWN_OUTCOME)
    if game.headers.get("White") == engine_name and result == BLACK_WIN_OUTCOME:
        return "w"
    if game.headers.get("Black") == engine_name and result == WHITE_WIN_OUTCOME:
        return "b"
    return None


def mine_loss_turning_points(
    pgn_text: str,
    *,
    engine_name: str,
    mate_depth: int = 1,
) -> list[LossTurningPoint]:
    """Walk every game in ``pgn_text`` the engine lost; return the core
    :class:`LossTurningPoint`s using ``ChessForcedLossResolver`` at
    ``mate_depth``."""
    if mate_depth < 1:
        raise ValueError("mate_depth must be at least 1")
    # Lazy import to avoid an import cycle (forced_loss imports
    # loss_mining.has_forced_mate).
    from dialectical_chess.forced_loss import ChessForcedLossResolver

    resolver = ChessForcedLossResolver(mate_depth=mate_depth)
    points: list[LossTurningPoint] = []
    stream = io.StringIO(pgn_text)
    game_index = 0
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            return points
        game_index += 1
        engine_side = engine_color_for_loss(game, engine_name)
        if engine_side is None:
            continue
        engine_outcome = (
            WHITE_WIN_OUTCOME if engine_side == "w" else BLACK_WIN_OUTCOME
        )
        result = chess_game_result_from_pgn(game)
        point = mine_turning_point(
            result,
            resolver,
            engine_side=engine_side,
            engine_outcome=engine_outcome,
            draw_outcome=DRAW_OUTCOME,
            game_index=game_index,
        )
        if point is not None:
            points.append(point)


def has_forced_mate(
    board: Any,
    *,
    mate_depth: int,
    deadline: float | None = None,
) -> bool:
    """Return whether the side to move can force mate within mate_depth moves.

    When ``deadline`` (a ``time.monotonic()`` value) is supplied and elapses
    mid-search, the search returns its best answer so far: a forced mate is
    reported only when actually proven, so on expiry an unproven branch is
    treated as "no forced mate" -- the safe, non-hanging answer (M3).
    """
    if mate_depth < 1:
        raise ValueError("mate_depth must be at least 1")
    if not isinstance(board, chess.Board):
        board = chess.Board(board.fen())
    return _has_forced_mate_board(
        board, mate_depth, cache=FORCED_MATE_CACHE, deadline=deadline
    )


def _has_forced_mate_board(
    board: chess.Board,
    mate_depth: int,
    *,
    cache: dict[tuple[Hashable, int], bool],
    deadline: float | None = None,
) -> bool:
    cache_key = (position_key(board), mate_depth)
    if cache_key in cache:
        return cache[cache_key]
    if board.is_checkmate():
        cache[cache_key] = True
        return True

    for move in board.legal_moves:
        if deadline is not None and time.monotonic() >= deadline:
            # Budget spent inside this single call: return best-so-far. No
            # proven mate was found, so the safe answer is False; do not cache
            # an unfinished search.
            return False
        attacker_child = board.copy(stack=False)
        attacker_child.push(move)
        if attacker_child.is_checkmate():
            cache[cache_key] = True
            return True
        if mate_depth == 1:
            continue
        defender_replies = list(attacker_child.legal_moves)
        if not defender_replies:
            continue
        if all(
            _has_forced_mate_board(
                defender_child(attacker_child, reply),
                mate_depth - 1,
                cache=cache,
                deadline=deadline,
            )
            for reply in defender_replies
        ):
            cache[cache_key] = True
            return True
    cache[cache_key] = False
    return False


def position_key(board: chess.Board) -> Hashable:
    """Cache key for legal move generation: board, turn, castling, en passant."""
    transposition_key = getattr(board, "_transposition_key", None)
    if transposition_key is not None:
        return transposition_key()
    return " ".join(board.fen().split()[:4])


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
    """Render a list of core :class:`LossTurningPoint`s as EPD review lines."""
    lines = []
    for point in points:
        epd_position = " ".join(point.fen_before.split()[:4])
        reason = escape_epd_string(_reason_for_point(point))
        fields = [f"am {point.played_move}"]
        if point.safe_alternatives:
            fields.append(f"bm {' '.join(point.safe_alternatives)}")
        fields.append(f'id "loss-{point.game_index}-ply-{point.ply}: {reason}"')
        lines.append(f"{epd_position} " + "; ".join(fields) + ";")
    return lines


def _reason_for_point(point: LossTurningPoint) -> str:
    """Render a chess-flavoured ``reason`` string from a core turning point.

    The chess-local turning point carried a ``reason`` field
    (``"allows_mate_in_{n}"``); the core's :class:`LossTurningPoint` records
    ``shot_wins_game`` and ``was_avoidable`` instead. For chess (mate-only
    resolver) "shot_wins_game=True" always means "allows mate"; the exact
    mate distance is the resolver's ``mate_depth`` configuration, which is
    not on the core point. Render the chess-stable label without the depth.
    """
    if not point.shot_wins_game:
        return "loses_material"
    return "allows_forced_mate"


def escape_epd_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


__all__ = [
    "BLACK_WIN_OUTCOME",
    "ChessGameResult",
    "DRAW_OUTCOME",
    "FORCED_MATE_CACHE",
    "LossTurningPoint",
    "UNKNOWN_OUTCOME",
    "WHITE_WIN_OUTCOME",
    "chess_game_result_from_pgn",
    "defender_child",
    "engine_color_for_loss",
    "escape_epd_string",
    "has_forced_mate",
    "mine_loss_turning_points",
    "mine_losses",
    "mine_turning_point",
    "position_key",
    "reviewed_epd_lines",
    "safe_legal_moves",
]
