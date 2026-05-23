"""Property + unit tests for :class:`ChessGradedPolicy` (Core Phase 3 chunk G.1).

Chunk G.1 made ``witness_opinion`` non-vacuous: the chess HEURISTIC vocabulary
now reaches the graded layer through ``dialectical_games.evidence``'s extended
taxonomy, and the chess policy maps ``(label, magnitude)`` to an
:class:`Opinion` via the saturated belief shape from chunk-G.1 plan §5.

Invariants pinned here:

* ``witness_belief`` is monotonic in magnitude (per prefix).
* Belief is bounded: ``0 <= belief <= _WITNESS_BELIEF_MAX``.
* Disbelief is non-negative: ``0 <= disbelief``.
* Uncertainty is constant at ``_WITNESS_UNCERTAINTY`` (no calibration in G.1).
* No-magnitude labels return ``_WITNESS_BELIEF_BASE``.
* The opinion's (b, d, u) sum to 1 (Jøsang opinion-sum constraint).
* Saturation differs by prefix: centipawn-scale labels saturate at 500,
  count-scale labels saturate at 4.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dialectical_chess.graded_policy import (
    ChessGradedPolicy,
    EDGE_TRUST_BASE_RATE,
    make_graded_policy,
    witness_belief,
)
from dialectical_chess.graded_policy import (
    _CENTIPAWN_MAGNITUDE_PREFIXES,
    _WITNESS_BELIEF_BASE,
    _WITNESS_BELIEF_MAX,
    _WITNESS_BASE_RATE,
    _WITNESS_MAGNITUDE_SAT_CENTIPAWN,
    _WITNESS_MAGNITUDE_SAT_COUNT,
    _WITNESS_UNCERTAINTY,
)


_COUNT_LABEL_EXAMPLES: tuple[str, ...] = (
    "pro:center_control",
    "pro:mobility",
    "obj:opening:premature_minor_check",
    "obj:opening:premature_rook",
    "obj:opening:premature_queen",
)


_CENTIPAWN_LABEL_EXAMPLES: tuple[str, ...] = tuple(_CENTIPAWN_MAGNITUDE_PREFIXES)


# ---------------------------------------------------------------------------
# unit — fixed constants are exactly the chunk-G.1 plan §5 values.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tuning_constants_match_plan_verbatim() -> None:
    """The six chunk-G.1 §5 tuning constants are pinned VERBATIM."""
    assert _WITNESS_UNCERTAINTY == 0.30
    assert _WITNESS_BELIEF_BASE == 0.55
    assert _WITNESS_BELIEF_MAX == 0.70
    assert _WITNESS_MAGNITUDE_SAT_CENTIPAWN == 500
    assert _WITNESS_MAGNITUDE_SAT_COUNT == 4
    assert _WITNESS_BASE_RATE == 0.5


@pytest.mark.unit
def test_no_magnitude_returns_base_belief() -> None:
    """A no-magnitude label (binary fire) returns ``_WITNESS_BELIEF_BASE``."""
    for label in ("pro:development:center_pawn", "obj:opening:king_walk"):
        assert witness_belief(None, label) == _WITNESS_BELIEF_BASE


# ---------------------------------------------------------------------------
# property — magnitude monotonicity (per prefix).
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    label=st.sampled_from(_COUNT_LABEL_EXAMPLES),
    pair=st.tuples(
        st.integers(min_value=0, max_value=10),
        st.integers(min_value=0, max_value=10),
    ),
)
def test_witness_belief_monotonic_in_magnitude_count_scale(
    label: str, pair: tuple[int, int]
) -> None:
    """``witness_belief(a, label) <= witness_belief(b, label)`` whenever
    ``a <= b`` for count-scale prefixes. Saturation cap ensures the
    inequality holds even past 4."""
    a, b = sorted(pair)
    label_a = f"{label}:{max(1, a)}"
    label_b = f"{label}:{max(1, b)}"
    belief_a = witness_belief(max(1, a), label_a)
    belief_b = witness_belief(max(1, b), label_b)
    assert belief_a <= belief_b + 1e-12


@pytest.mark.property
@given(
    label=st.sampled_from(_CENTIPAWN_LABEL_EXAMPLES),
    pair=st.tuples(
        st.integers(min_value=1, max_value=2000),
        st.integers(min_value=1, max_value=2000),
    ),
)
def test_witness_belief_monotonic_in_magnitude_centipawn_scale(
    label: str, pair: tuple[int, int]
) -> None:
    """Centipawn-scale prefixes saturate at 500; monotonicity holds across
    the full integer range."""
    a, b = sorted(pair)
    label_a = f"{label}:{a}"
    label_b = f"{label}:{b}"
    belief_a = witness_belief(a, label_a)
    belief_b = witness_belief(b, label_b)
    assert belief_a <= belief_b + 1e-12


@pytest.mark.unit
def test_witness_belief_specific_monotonicity_examples() -> None:
    """Spot-check ``witness_belief(2) <= witness_belief(5)`` per plan §5."""
    assert witness_belief(2, "pro:mobility:2") <= witness_belief(5, "pro:mobility:5")
    assert witness_belief(100, "pro:piece_safety:defended:100") <= witness_belief(
        500, "pro:piece_safety:defended:500"
    )


# ---------------------------------------------------------------------------
# property — bounds.
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    label=st.sampled_from(_COUNT_LABEL_EXAMPLES + _CENTIPAWN_LABEL_EXAMPLES),
    magnitude=st.integers(min_value=1, max_value=5000),
)
def test_witness_belief_bounded_above_by_max(label: str, magnitude: int) -> None:
    """For every valid magnitude, ``belief <= _WITNESS_BELIEF_MAX``."""
    full_label = f"{label}:{magnitude}"
    belief = witness_belief(magnitude, full_label)
    assert 0.0 <= belief <= _WITNESS_BELIEF_MAX + 1e-12


@pytest.mark.property
@given(
    label=st.sampled_from(_COUNT_LABEL_EXAMPLES + _CENTIPAWN_LABEL_EXAMPLES),
    magnitude=st.integers(min_value=1, max_value=5000),
)
def test_witness_belief_at_least_base(label: str, magnitude: int) -> None:
    """For every valid magnitude (positive), ``belief >= _WITNESS_BELIEF_BASE``."""
    full_label = f"{label}:{magnitude}"
    belief = witness_belief(magnitude, full_label)
    assert belief >= _WITNESS_BELIEF_BASE - 1e-12


# ---------------------------------------------------------------------------
# property — opinion sums to 1 (Jøsang sum constraint), u == 0.30,
# disbelief non-negative.
# ---------------------------------------------------------------------------


def _make_probe() -> Any:
    """Build a minimal probe object with the only field
    ``witness_opinion`` reads — ``child_eval`` — set to 0. The chess
    graded policy's witness_opinion ignores probe content; this is a
    Protocol-shape placeholder.
    """
    # Use a tiny duck-typed object so we don't need to import / construct
    # a full MoveProbe. The chess witness_opinion does not access probe
    # fields directly today.
    class _StubProbe:
        child_eval = 0
    return _StubProbe()


@pytest.mark.property
@given(
    label=st.sampled_from(_COUNT_LABEL_EXAMPLES + _CENTIPAWN_LABEL_EXAMPLES),
    magnitude=st.integers(min_value=1, max_value=5000),
)
def test_witness_opinion_sum_equals_one(label: str, magnitude: int) -> None:
    """A Jøsang opinion satisfies ``b + d + u == 1`` exactly (up to fp eps).

    The chess policy returns ``Opinion(belief, disbelief, u, a)`` with
    ``disbelief = max(0, 1 - belief - u)``; the sum is therefore
    ``belief + (1 - belief - u) + u = 1`` (since belief is bounded so
    ``1 - belief - u >= 0`` and the max is a no-op).
    """
    policy = make_graded_policy()
    full_label = f"{label}:{magnitude}"
    op = policy.witness_opinion(
        probe=_make_probe(), label=full_label, magnitude=magnitude
    )
    s = op.b + op.d + op.u
    assert math.isclose(s, 1.0, abs_tol=1e-9)


@pytest.mark.property
@given(
    label=st.sampled_from(_COUNT_LABEL_EXAMPLES + _CENTIPAWN_LABEL_EXAMPLES),
    magnitude=st.integers(min_value=1, max_value=5000),
)
def test_witness_opinion_u_equals_tuning_constant(label: str, magnitude: int) -> None:
    """The chunk-G.1 policy keeps ``u == _WITNESS_UNCERTAINTY`` (no
    conditioned-``u`` calibration in this cycle)."""
    policy = make_graded_policy()
    full_label = f"{label}:{magnitude}"
    op = policy.witness_opinion(
        probe=_make_probe(), label=full_label, magnitude=magnitude
    )
    assert op.u == _WITNESS_UNCERTAINTY


@pytest.mark.property
@given(
    label=st.sampled_from(_COUNT_LABEL_EXAMPLES + _CENTIPAWN_LABEL_EXAMPLES),
    magnitude=st.integers(min_value=1, max_value=5000),
)
def test_witness_opinion_disbelief_non_negative(
    label: str, magnitude: int
) -> None:
    """Disbelief is residual ``1 - belief - u`` clamped to non-negative;
    the residual is always non-negative anyway for the chunk-G.1 belief
    band (max belief 0.70 + u 0.30 = 1.00), so the clamp is a no-op."""
    policy = make_graded_policy()
    full_label = f"{label}:{magnitude}"
    op = policy.witness_opinion(
        probe=_make_probe(), label=full_label, magnitude=magnitude
    )
    assert op.d >= 0.0


@pytest.mark.property
@given(magnitude=st.integers(min_value=1, max_value=5000))
def test_witness_opinion_base_rate(magnitude: int) -> None:
    """The opinion's base rate is ``_WITNESS_BASE_RATE`` (0.5)."""
    policy = make_graded_policy()
    op = policy.witness_opinion(
        probe=_make_probe(),
        label=f"pro:mobility:{magnitude}",
        magnitude=magnitude,
    )
    assert op.a == _WITNESS_BASE_RATE


# ---------------------------------------------------------------------------
# unit — saturation differs by prefix (centipawn vs count).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_centipawn_scale_saturates_at_500() -> None:
    """A centipawn-scale label's belief at 500 equals belief at 5000
    (saturation cap)."""
    label_500 = "pro:piece_safety:defended:500"
    label_5000 = "pro:piece_safety:defended:5000"
    assert math.isclose(
        witness_belief(500, label_500),
        witness_belief(5000, label_5000),
        abs_tol=1e-12,
    )


@pytest.mark.unit
def test_count_scale_saturates_at_4() -> None:
    """A count-scale label's belief at 4 equals belief at 40 (saturation cap)."""
    label_4 = "pro:center_control:4"
    label_40 = "pro:center_control:40"
    assert math.isclose(
        witness_belief(4, label_4),
        witness_belief(40, label_40),
        abs_tol=1e-12,
    )


@pytest.mark.unit
def test_belief_at_saturation_equals_max() -> None:
    """At saturation the belief equals ``_WITNESS_BELIEF_MAX`` exactly."""
    assert math.isclose(
        witness_belief(4, "pro:center_control:4"),
        _WITNESS_BELIEF_MAX,
        abs_tol=1e-12,
    )
    assert math.isclose(
        witness_belief(500, "pro:piece_safety:defended:500"),
        _WITNESS_BELIEF_MAX,
        abs_tol=1e-12,
    )


# ---------------------------------------------------------------------------
# unit — edge_trust is unchanged in G.1.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_edge_trust_is_dogmatic_true_at_base_rate() -> None:
    """``edge_trust`` is ``Opinion.dogmatic_true(EDGE_TRUST_BASE_RATE)``,
    unchanged in chunk G.1."""
    policy = ChessGradedPolicy()
    op = policy.edge_trust
    assert op.b == 1.0
    assert op.d == 0.0
    assert op.u == 0.0
    assert op.a == EDGE_TRUST_BASE_RATE
