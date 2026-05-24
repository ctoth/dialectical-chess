"""Property + unit tests for the chunk H' :class:`ChessGradedPolicy`.

Chunk H' (Core Phase 3, 2026-05-24) dissolved the chunk G.1 tuned constants
in favour of principled derivations:

* MATERIAL witness  -> ``Opinion(b, 0, 1 - b, MAX_ENT_PRIOR)`` where ``b``
  is the Hazen rank-fraction of the magnitude in the per-position
  per-label-prefix CDF;
* COUNT witness     -> ``Opinion.from_evidence(n, 0, MAX_ENT_PRIOR)``;
* BOOLEAN witness   -> ``Opinion.from_evidence(1, 0, MAX_ENT_PRIOR)``;
* SEARCH witness    -> vacuous (no core translation yet -- chunk-H' plan
                       section 6-G defers);
* move base rate    -> per-position Hazen rank-fraction over sibling
                       ``child_eval`` (ASCENDING -- chess child_eval is
                       mover-relative, larger is better for the mover).

The ONE literal that survives in the witness / policy path is
:data:`MAX_ENT_PRIOR` (= 0.5, the max-entropy binary prior).

Invariants pinned here:

* opinion components in ``[0, 1]`` and ``b + d + u == 1`` exactly;
* COUNT: ``u`` is monotone-DECREASING in magnitude, ``b`` is
  monotone-INCREASING;
* MATERIAL: ``b`` is monotone-INCREASING in rank (a larger magnitude in
  the same per-prefix corpus gets at least as high a belief);
* base rate is exactly :data:`MAX_ENT_PRIOR` for every witness;
* :meth:`with_probes` returns a policy whose :meth:`move_base_rate` is in
  ``(0, 1)`` for every survivor probe (the Hazen open-interval guarantee).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dialectical_games.arguments import MoveProbe
from dialectical_chess.graded_policy import (
    ChessGradedPolicy,
    MAX_ENT_PRIOR,
    make_graded_policy,
)


_COUNT_LABEL_EXAMPLES: tuple[str, ...] = (
    "pro:center_control",
    "pro:mobility",
    "obj:opening:premature_minor_check",
    "obj:opening:premature_rook",
    "obj:opening:premature_queen",
)


_MATERIAL_LABEL_EXAMPLES: tuple[str, ...] = (
    "pro:piece_safety:defended",
    "pro:tactical:threat",
    "pro:smt:fork",
    "obj:smt:fork:moved_piece_en_pris",
)


_BOOLEAN_LABEL_EXAMPLES: tuple[str, ...] = (
    "pro:development:center_pawn",
    "obj:opening:king_walk",
    "pro:king_safety:castle",
)


# --- The single literal -----------------------------------------------------


@pytest.mark.unit
def test_max_ent_prior_is_one_half() -> None:
    """The ONLY literal in the chunk H' witness / policy path is 0.5."""
    assert MAX_ENT_PRIOR == 0.5


# --- BOOLEAN witness derivation --------------------------------------------


@pytest.mark.unit
def test_boolean_witness_is_single_observation_max_ent() -> None:
    """A BOOLEAN witness (magnitude=None) is ``from_evidence(1, 0, 0.5)``."""
    policy = make_graded_policy()
    op = policy.witness_opinion(
        probe=MoveProbe(move_id="m1"),
        label="pro:development:center_pawn",
        magnitude=None,
    )
    # Beta-binomial with r=1, s=0, W=2: b = 1/3; u = 2/3.
    assert math.isclose(op.b, 1.0 / 3.0, abs_tol=1e-9)
    assert op.d == 0.0
    assert math.isclose(op.u, 2.0 / 3.0, abs_tol=1e-9)
    assert op.a == MAX_ENT_PRIOR


# --- COUNT witness derivation ----------------------------------------------


@pytest.mark.property
@given(
    label=st.sampled_from(_COUNT_LABEL_EXAMPLES),
    magnitude=st.integers(min_value=1, max_value=10_000),
)
def test_count_witness_components_sum_to_one(label: str, magnitude: int) -> None:
    """The Jøsang sum constraint ``b + d + u == 1`` holds for any magnitude."""
    policy = make_graded_policy()
    op = policy.witness_opinion(
        probe=MoveProbe(move_id="m1"),
        label=f"{label}:{magnitude}",
        magnitude=magnitude,
    )
    assert math.isclose(op.b + op.d + op.u, 1.0, abs_tol=1e-9)


@pytest.mark.property
@given(
    label=st.sampled_from(_COUNT_LABEL_EXAMPLES),
    magnitude=st.integers(min_value=1, max_value=10_000),
)
def test_count_witness_components_in_unit_interval(
    label: str, magnitude: int
) -> None:
    """Every opinion component lives in ``[0, 1]``."""
    policy = make_graded_policy()
    op = policy.witness_opinion(
        probe=MoveProbe(move_id="m1"),
        label=f"{label}:{magnitude}",
        magnitude=magnitude,
    )
    for component in (op.b, op.d, op.u):
        assert 0.0 <= component <= 1.0


@pytest.mark.property
@given(
    label=st.sampled_from(_COUNT_LABEL_EXAMPLES),
    pair=st.tuples(
        st.integers(min_value=1, max_value=10_000),
        st.integers(min_value=1, max_value=10_000),
    ),
)
def test_count_witness_uncertainty_monotone_decreasing(
    label: str, pair: tuple[int, int]
) -> None:
    """``u`` is monotone-DECREASING in magnitude."""
    policy = make_graded_policy()
    a, b = sorted(pair)
    op_a = policy.witness_opinion(
        probe=MoveProbe(move_id="m1"),
        label=f"{label}:{a}",
        magnitude=a,
    )
    op_b = policy.witness_opinion(
        probe=MoveProbe(move_id="m1"),
        label=f"{label}:{b}",
        magnitude=b,
    )
    assert op_a.u >= op_b.u - 1e-12


@pytest.mark.property
@given(
    label=st.sampled_from(_COUNT_LABEL_EXAMPLES),
    pair=st.tuples(
        st.integers(min_value=1, max_value=10_000),
        st.integers(min_value=1, max_value=10_000),
    ),
)
def test_count_witness_belief_monotone_increasing(
    label: str, pair: tuple[int, int]
) -> None:
    """``b`` is monotone-INCREASING in magnitude."""
    policy = make_graded_policy()
    a, b = sorted(pair)
    op_a = policy.witness_opinion(
        probe=MoveProbe(move_id="m1"),
        label=f"{label}:{a}",
        magnitude=a,
    )
    op_b = policy.witness_opinion(
        probe=MoveProbe(move_id="m1"),
        label=f"{label}:{b}",
        magnitude=b,
    )
    assert op_a.b <= op_b.b + 1e-12


# --- MATERIAL witness derivation (chess-only, requires bound policy) -------


@pytest.mark.unit
def test_material_witness_single_observation_falls_back_to_boolean() -> None:
    """A MATERIAL prefix with N<=1 in the corpus falls back to BOOLEAN.

    Single-observation case (chunk H' plan section 2): the rank space is
    "largest of one observation" -- no comparison. Falls back to
    ``Opinion.from_evidence(1, 0, MAX_ENT_PRIOR)`` -- the BOOLEAN shape.
    """
    probes = [
        MoveProbe(move_id="m1", reasons=("pro:piece_safety:defended:100",)),
    ]
    policy = make_graded_policy().with_probes(probes)
    op = policy.witness_opinion(
        probe=probes[0],
        label="pro:piece_safety:defended:100",
        magnitude=100,
    )
    assert math.isclose(op.b, 1.0 / 3.0, abs_tol=1e-9)
    assert math.isclose(op.u, 2.0 / 3.0, abs_tol=1e-9)


@pytest.mark.unit
def test_material_witness_rank_fraction_in_open_interval() -> None:
    """A MATERIAL witness with N>=2 in the corpus uses the Hazen rank-fraction."""
    probes = [
        MoveProbe(move_id="m1", reasons=("pro:piece_safety:defended:100",)),
        MoveProbe(move_id="m2", reasons=("pro:piece_safety:defended:300",)),
        MoveProbe(move_id="m3", reasons=("pro:piece_safety:defended:500",)),
    ]
    policy = make_graded_policy().with_probes(probes)
    # Rank fractions for [100, 300, 500] under Hazen: 1/4, 2/4, 3/4.
    op_100 = policy.witness_opinion(
        probe=probes[0],
        label="pro:piece_safety:defended:100",
        magnitude=100,
    )
    op_300 = policy.witness_opinion(
        probe=probes[1],
        label="pro:piece_safety:defended:300",
        magnitude=300,
    )
    op_500 = policy.witness_opinion(
        probe=probes[2],
        label="pro:piece_safety:defended:500",
        magnitude=500,
    )
    assert math.isclose(op_100.b, 1.0 / 4.0, abs_tol=1e-9)
    assert math.isclose(op_300.b, 2.0 / 4.0, abs_tol=1e-9)
    assert math.isclose(op_500.b, 3.0 / 4.0, abs_tol=1e-9)
    # All three opinions have d=0 and u=1-b.
    for op, b in ((op_100, 1.0 / 4.0), (op_300, 2.0 / 4.0), (op_500, 3.0 / 4.0)):
        assert op.d == 0.0
        assert math.isclose(op.u, 1.0 - b, abs_tol=1e-9)
        assert op.a == MAX_ENT_PRIOR


@pytest.mark.unit
def test_material_witness_per_prefix_isolation() -> None:
    """Per-prefix MATERIAL CDFs are isolated across prefixes."""
    # Two different MATERIAL prefixes, each with two observations.
    probes = [
        MoveProbe(move_id="m1", reasons=("pro:piece_safety:defended:100",)),
        MoveProbe(move_id="m2", reasons=("pro:piece_safety:defended:300",)),
        MoveProbe(move_id="m3", reasons=("pro:tactical:threat:200",)),
        MoveProbe(move_id="m4", reasons=("pro:tactical:threat:400",)),
    ]
    policy = make_graded_policy().with_probes(probes)
    op_def_100 = policy.witness_opinion(
        probe=probes[0],
        label="pro:piece_safety:defended:100",
        magnitude=100,
    )
    op_thr_200 = policy.witness_opinion(
        probe=probes[2],
        label="pro:tactical:threat:200",
        magnitude=200,
    )
    # Each prefix has its own corpus of size 2; both at rank 1: b = 1/3.
    assert math.isclose(op_def_100.b, 1.0 / 3.0, abs_tol=1e-9)
    assert math.isclose(op_thr_200.b, 1.0 / 3.0, abs_tol=1e-9)


@pytest.mark.property
@given(
    label=st.sampled_from(_MATERIAL_LABEL_EXAMPLES),
    magnitudes=st.lists(
        st.integers(min_value=1, max_value=2_000),
        min_size=2,
        max_size=8,
        unique=True,
    ),
)
def test_material_witness_belief_monotone_in_magnitude(
    label: str, magnitudes: list[int]
) -> None:
    """A larger magnitude in the same per-prefix corpus gets at least as
    high a belief (Hazen rank-fraction is monotone-increasing in rank)."""
    probes = [
        MoveProbe(move_id=f"m{i}", reasons=(f"{label}:{m}",))
        for i, m in enumerate(magnitudes)
    ]
    policy = make_graded_policy().with_probes(probes)
    sorted_magnitudes = sorted(magnitudes)
    prev_b: float | None = None
    for m, p in zip(sorted_magnitudes, sorted(probes, key=lambda p: int(p.reasons[0].split(":")[-1]))):
        op = policy.witness_opinion(
            probe=p, label=f"{label}:{m}", magnitude=m
        )
        if prev_b is not None:
            assert op.b >= prev_b - 1e-12
        prev_b = op.b


# --- Combined invariants (BOOLEAN + COUNT + MATERIAL) ----------------------


@pytest.mark.property
@given(
    label=st.sampled_from(_BOOLEAN_LABEL_EXAMPLES),
)
def test_witness_opinion_base_rate_for_boolean(label: str) -> None:
    """The opinion's base rate is :data:`MAX_ENT_PRIOR` for every BOOLEAN."""
    policy = make_graded_policy()
    op = policy.witness_opinion(
        probe=MoveProbe(move_id="m1"),
        label=label,
        magnitude=None,
    )
    assert op.a == MAX_ENT_PRIOR


@pytest.mark.property
@given(magnitude=st.integers(min_value=1, max_value=10_000))
def test_witness_opinion_base_rate_for_count(magnitude: int) -> None:
    """The opinion's base rate is :data:`MAX_ENT_PRIOR` for every COUNT."""
    policy = make_graded_policy()
    op = policy.witness_opinion(
        probe=MoveProbe(move_id="m1"),
        label=f"pro:mobility:{magnitude}",
        magnitude=magnitude,
    )
    assert op.a == MAX_ENT_PRIOR


# --- move_base_rate via with_probes ----------------------------------------


@pytest.mark.unit
def test_unbound_policy_returns_max_ent_prior() -> None:
    """An unbound policy returns the neutral max-entropy prior."""
    policy = make_graded_policy()
    rate = policy.move_base_rate(MoveProbe(move_id="m1", child_eval=0))
    assert rate == MAX_ENT_PRIOR


@pytest.mark.unit
def test_with_probes_ascending_orientation() -> None:
    """Larger ``child_eval`` (better for mover) -> higher base rate."""
    probes = [
        MoveProbe(move_id="worst", child_eval=-200),
        MoveProbe(move_id="middle", child_eval=0),
        MoveProbe(move_id="best", child_eval=200),
    ]
    bound = make_graded_policy().with_probes(probes)
    rate_worst = bound.move_base_rate(probes[0])
    rate_middle = bound.move_base_rate(probes[1])
    rate_best = bound.move_base_rate(probes[2])
    assert rate_worst < rate_middle < rate_best


@pytest.mark.unit
def test_with_probes_single_move_returns_half() -> None:
    """A forced-move position (N=1) gets the neutral rank-fraction 1/2."""
    probes = [MoveProbe(move_id="forced", child_eval=100)]
    bound = make_graded_policy().with_probes(probes)
    assert bound.move_base_rate(probes[0]) == pytest.approx(0.5)


@pytest.mark.property
@given(
    evals=st.lists(
        st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=10
    )
)
def test_with_probes_rank_fractions_open_interval(evals: list[int]) -> None:
    """Every survivor's rank-fraction is strictly in ``(0, 1)``."""
    probes = [
        MoveProbe(move_id=f"m{i}", child_eval=e) for i, e in enumerate(evals)
    ]
    bound = make_graded_policy().with_probes(probes)
    for p in probes:
        rate = bound.move_base_rate(p)
        assert 0.0 < rate < 1.0


# --- edge_trust -------------------------------------------------------------


@pytest.mark.unit
def test_edge_trust_is_dogmatic_true_at_max_ent_prior() -> None:
    """``edge_trust`` is ``Opinion.dogmatic_true(MAX_ENT_PRIOR)``."""
    policy = make_graded_policy()
    op = policy.edge_trust
    assert op.b == 1.0
    assert op.d == 0.0
    assert op.u == 0.0
    assert op.a == MAX_ENT_PRIOR


# --- with_probes Protocol compliance ---------------------------------------


@pytest.mark.unit
def test_with_probes_returns_same_class() -> None:
    """:meth:`with_probes` returns a :class:`ChessGradedPolicy`."""
    probes = [MoveProbe(move_id="m1", child_eval=0)]
    bound = make_graded_policy().with_probes(probes)
    assert isinstance(bound, ChessGradedPolicy)


@pytest.mark.unit
def test_with_probes_does_not_mutate_self() -> None:
    """:meth:`with_probes` returns a NEW policy (immutability preserved)."""
    base = make_graded_policy()
    probes = [MoveProbe(move_id="m1", child_eval=0)]
    bound = base.with_probes(probes)
    # The base policy still has no cache -> unbound base rate is MAX_ENT_PRIOR.
    assert base.move_base_rate(probes[0]) == MAX_ENT_PRIOR
    # The bound policy has the per-position rank-fraction.
    assert bound.move_base_rate(probes[0]) == pytest.approx(0.5)


# --- SEARCH-class fall-through ---------------------------------------------


@pytest.mark.unit
def test_search_class_returns_vacuous_opinion() -> None:
    """A SEARCH-class label returns the vacuous opinion (chunk-H' plan §6-G).

    No core taxonomy entry translates ``search_support:{backend}:{score}``
    yet; the policy is honest about "we don't have a derivation for this
    witness class yet" by returning the vacuous opinion. Translation is
    deferred to a later cycle.
    """
    policy = make_graded_policy()
    op = policy.witness_opinion(
        probe=MoveProbe(move_id="m1"),
        label="search_support:stockfish:120",
        magnitude=120,
    )
    # Vacuous: b=0, d=0, u=1.
    assert op.b == 0.0
    assert op.d == 0.0
    assert op.u == 1.0
    assert op.a == MAX_ENT_PRIOR
