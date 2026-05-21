"""Static-prior contract tests for the Phase-2 opinion-valued decider."""

from __future__ import annotations

from dataclasses import replace

import pytest

from dialectical_chess.arguments import MoveProbe
from dialectical_chess.probe import owned_board_from_fen, probe_moves
from dialectical_chess.static_prior import TAU_CLAMP, squash, static_prior


QUEEN_GRAB_INTO_MATE_FEN = "6nr/n4pp1/k6p/8/3p4/1P6/1PPP1PPP/r1B3K1 w - - 0 22"


def make_probe(
    uci: str,
    *,
    score: int = 0,
    reasons: tuple[str, ...] = (),
    objections: tuple[str, ...] = (),
    reply_attacks: tuple[str, ...] = (),
    post_fen: str | None = None,
) -> MoveProbe:
    return MoveProbe(
        uci=uci,
        san=uci,
        score=score,
        is_checkmate=False,
        gives_check=False,
        is_capture=False,
        captured_value=0,
        promotion_value=0,
        reasons=reasons,
        objections=objections,
        reply_attacks=reply_attacks,
        post_fen=post_fen,
    )


@pytest.mark.unit
def test_squash_monotone_and_centered() -> None:
    assert squash(0.0) == pytest.approx(0.5, abs=1e-12)
    assert squash(100.0) == pytest.approx(0.622459, abs=1e-6)
    assert squash(300.0) == pytest.approx(0.817574, abs=1e-6)
    assert squash(400.0) == pytest.approx(0.880797, abs=1e-6)
    values = [squash(value) for value in (-800.0, -200.0, 0.0, 200.0, 800.0)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


@pytest.mark.unit
def test_squash_open_interval_clamp() -> None:
    lo, hi = TAU_CLAMP
    assert (lo, hi) == (0.01, 0.99)
    assert squash(-1e6) == lo
    assert squash(1e6) == hi
    for prior in (-1e6, -1e3, -1.0, 0.0, 1.0, 1e3, 1e6):
        assert 0.0 < squash(prior) < 1.0


@pytest.mark.property
def test_static_prior_ignores_probe_score() -> None:
    board = owned_board_from_fen(QUEEN_GRAB_INTO_MATE_FEN)
    base = probe_moves(board, search_depth=0, smt_fork=False)[0]
    assert base.post_fen is not None
    mutated = replace(base, score=base.score + 50_000)
    assert static_prior(mutated) == pytest.approx(static_prior(base), abs=1e-9)


@pytest.mark.property
def test_static_prior_ignores_evidence_labels() -> None:
    board = owned_board_from_fen(QUEEN_GRAB_INTO_MATE_FEN)
    base = probe_moves(board, search_depth=0, smt_fork=False)[0]
    mutated = replace(
        base,
        reasons=base.reasons + ("material:capture:900", "development:zz9z9z:center_pawn"),
        objections=base.objections + ("safety:queen_blunder:zz9z9z:580",),
        reply_attacks=base.reply_attacks + ("reply_mate:zz9z9z",),
    )
    assert static_prior(mutated) == pytest.approx(static_prior(base), abs=1e-9)


@pytest.mark.property
def test_material_label_does_not_change_prior_for_same_post_board() -> None:
    post_fen = "4k3/8/8/8/8/8/8/4KQ2 b - - 0 1"
    with_capture_label = make_probe(
        "e1f1",
        reasons=("material:capture:900",),
        post_fen=post_fen,
    )
    without_capture_label = make_probe("e1f1", post_fen=post_fen)
    assert static_prior(with_capture_label) == pytest.approx(
        static_prior(without_capture_label),
        abs=1e-9,
    )


@pytest.mark.unit
def test_static_prior_has_honest_zero_without_board_snapshot() -> None:
    assert static_prior(make_probe("a1a2", score=999_999)) == 0.0
