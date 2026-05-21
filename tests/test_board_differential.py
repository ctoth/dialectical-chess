from __future__ import annotations

import random

import chess
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dialectical_chess.board import (
    CURATED_FENS,
    PERFT_FIXTURES,
    OwnedBoard,
    compare_to_oracle,
    oracle_legal_moves,
    owned_perft,
)


def deterministic_random_fens(*, count: int = 200, seed: int = 20260520) -> tuple[str, ...]:
    rng = random.Random(seed)
    fens: list[str] = []
    board = chess.Board()
    while len(fens) < count:
        if board.is_game_over(claim_draw=True) or board.fullmove_number > 80:
            board = chess.Board()
            continue
        legal = sorted(board.legal_moves, key=lambda move: move.uci())
        board.push(rng.choice(legal))
        fens.append(board.fen())
    return tuple(fens)


RANDOM_FENS = deterministic_random_fens()


@pytest.mark.differential
@pytest.mark.parametrize("name,fen", tuple(CURATED_FENS.items()) + tuple((f"random-{index}", fen) for index, fen in enumerate(RANDOM_FENS)))
def test_owned_legal_moves_match_oracle(name: str, fen: str) -> None:
    comparison = compare_to_oracle(fen)

    assert comparison["match"], name
    assert comparison["missing"] == []
    assert comparison["extra"] == []


@pytest.mark.differential
@pytest.mark.parametrize(
    "name,fen,depth,expected",
    [
        (name, fen, depth, expected)
        for name, (fen, depths) in PERFT_FIXTURES.items()
        for depth, expected in depths.items()
    ],
)
def test_owned_perft_matches_fixtures(name: str, fen: str, depth: int, expected: int) -> None:
    assert owned_perft(OwnedBoard.from_fen(fen), depth) == expected, name


@pytest.mark.differential
@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(st.sampled_from(RANDOM_FENS))
def test_owned_legal_move_set_matches_oracle_on_random_legal_positions(fen: str) -> None:
    owned = {move.uci() for move in OwnedBoard.from_fen(fen).legal_moves()}

    assert owned == oracle_legal_moves(fen)
