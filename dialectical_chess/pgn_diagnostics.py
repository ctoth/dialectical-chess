"""PGN position diagnostics for match evidence."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import chess
import chess.pgn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=Path, required=True)
    parser.add_argument("--engine-name", default="Dialectical")
    parser.add_argument("--game-index", type=int)
    parser.add_argument("--move-uci")
    parser.add_argument("--engine-only", action="store_true")
    args = parser.parse_args()

    payload = pgn_positions(
        args.pgn.read_text(encoding="utf-8"),
        engine_name=args.engine_name,
        game_index=args.game_index,
        move_uci=args.move_uci,
        engine_only=args.engine_only,
    )
    print(json.dumps(payload, indent=2))
    return 0


def pgn_positions(
    pgn_text: str,
    *,
    engine_name: str = "Dialectical",
    game_index: int | None = None,
    move_uci: str | None = None,
    engine_only: bool = False,
) -> dict[str, Any]:
    positions: list[dict[str, Any]] = []
    stream = io.StringIO(pgn_text)
    current_game = 0
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        current_game += 1
        if game_index is not None and current_game != game_index:
            continue
        engine_color = engine_color_for_game(game, engine_name)
        board = game.board()
        node = game
        for ply, move in enumerate(game.mainline_moves(), start=1):
            next_node = node.variation(0)
            fen_before = board.fen()
            mover = board.turn
            move_text = move.uci()
            board.push(move)
            if move_uci is not None and move_text != move_uci:
                node = next_node
                continue
            if engine_only and mover != engine_color:
                node = next_node
                continue
            positions.append(
                {
                    "game_index": current_game,
                    "ply": ply,
                    "mover": "w" if mover == chess.WHITE else "b",
                    "engine_to_move": mover == engine_color,
                    "fen_before": fen_before,
                    "move_uci": move_text,
                    "comment": next_node.comment,
                }
            )
            node = next_node
    return {"positions": positions}


def engine_color_for_game(game: chess.pgn.Game, engine_name: str) -> bool | None:
    if game.headers.get("White") == engine_name:
        return chess.WHITE
    if game.headers.get("Black") == engine_name:
        return chess.BLACK
    return None


if __name__ == "__main__":
    raise SystemExit(main())
