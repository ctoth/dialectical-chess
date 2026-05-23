"""Chess cartridge: :class:`ChessGradedPolicy`.

Implements the core ``dialectical_games.arguments.GradedPolicy`` Protocol.
A per-build policy bound to the root board (so position-level features can
be cached once).

* ``move_base_rate(probe)`` — squashes ``probe.child_eval`` (cartridge-pre-
  computed centipawn-scale int) into ``(0, 1)``. The cartridge sets
  ``child_eval = static_prior(probe) * 1000`` at probe time, so the policy
  re-squashes to ``(0.01, 0.99)``.
* ``witness_opinion`` — chunk-G.1 (core Phase 3) made this non-vacuous.
  Per-prefix saturation: count-scale (1-4) for positional counts,
  centipawn-scale (500 cp) for material magnitudes. Belief band
  ``[0.55, 0.70]`` linearly interpolated from magnitude; ``u = 0.30``
  baseline; ``a = 0.5`` (no a-priori asymmetry). See
  ``dialectical-chess/reports/core-phase3-chunkg-plan.md`` §5 for the
  design and the §7-A risk on dual-scale magnitudes.
* ``edge_trust`` — chess uses the pre-Phase-3 ``EDGE_TRUST_BASE_RATE = 0.5``
  as a dogmatic-true opinion (the chess opinion-graph builder's prior
  edge trust).

Calibration (conditioned-``u``, per-label belief overrides) is **deferred**
to a follow-up cycle ("chunk H"): the chess flip dataset does not yet
exist; the starting values here mirror checkers' pre-calibration band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from doxa import Opinion

from dialectical_games.arguments import GradedPolicy, MoveProbe

EDGE_TRUST_BASE_RATE: float = 0.5
OPINION_LEAF_BASE_RATE: float = 0.5

_TAU_SCALE: float = 400.0  # mirrors dialectical_chess.static_prior.TAU_SCALE
_TAU_CLAMP_LO: float = 0.01
_TAU_CLAMP_HI: float = 0.99


# --- chunk G.1: witness_opinion tuning ------------------------------------
#
# Tuning constants — VERBATIM from ``reports/core-phase3-chunkg-plan.md`` §5.
# These mirror checkers' pre-calibration band (``dialectical-checkers``
# ``graded_tuning.py:117-125``). Conditioned-``u`` calibration is a separate
# follow-up cycle ("chunk H"); not done here.

_WITNESS_UNCERTAINTY: float = 0.30
_WITNESS_BELIEF_BASE: float = 0.55
_WITNESS_BELIEF_MAX: float = 0.70
_WITNESS_MAGNITUDE_SAT_CENTIPAWN: int = 500
_WITNESS_MAGNITUDE_SAT_COUNT: int = 4
_WITNESS_BASE_RATE: float = 0.5


# Magnitude prefixes whose ``:{n}`` value is centipawn-scale (chess
# material). Everything else is count-scale (positional counts). The chess
# core-label translator emits these prefixes via ``core_labels.py``; the
# saturation lookup here keys off the prefix (the label minus its final
# ``:{n}`` segment).
_CENTIPAWN_MAGNITUDE_PREFIXES: frozenset[str] = frozenset({
    "pro:material",                      # FACT — never reaches HEURISTIC policy
    "pro:piece_safety:defended",
    "pro:tactical:threat",
    "pro:smt:fork",
    "obj:smt:fork:moved_piece_en_pris",
})


def _squash_centipawn(centipawn: int) -> float:
    """Squash a centipawn-scale evaluation to ``(0.01, 0.99)``."""
    raw = 0.5 + 0.5 * math.tanh(centipawn / (_TAU_SCALE * 1000.0))
    return max(_TAU_CLAMP_LO, min(_TAU_CLAMP_HI, raw))


def _saturation_for_prefix(label_prefix: str) -> int:
    """Return the magnitude saturation for ``label_prefix`` (chunk G.1 §5).

    Centipawn-scale prefixes saturate at 500; count-scale prefixes saturate
    at 4. The label-prefix is everything up to the final ``:{n}`` segment.
    """
    if label_prefix in _CENTIPAWN_MAGNITUDE_PREFIXES:
        return _WITNESS_MAGNITUDE_SAT_CENTIPAWN
    return _WITNESS_MAGNITUDE_SAT_COUNT


def witness_belief(magnitude: int | None, label: str) -> float:
    """Belief from witness magnitude (chunk G.1 §5).

    No-magnitude labels return ``_WITNESS_BELIEF_BASE``. A magnitude-carrying
    label interpolates linearly from base to max as magnitude rises to the
    per-prefix saturation; magnitudes at or above saturation cap at max.
    Negative magnitudes are clamped to zero (treated as no-evidence).
    """
    if magnitude is None:
        return _WITNESS_BELIEF_BASE
    prefix = label.rpartition(":")[0]
    sat = _saturation_for_prefix(prefix)
    capped = max(0, min(magnitude, sat))
    span = _WITNESS_BELIEF_MAX - _WITNESS_BELIEF_BASE
    return _WITNESS_BELIEF_BASE + span * (capped / sat)


@dataclass(frozen=True)
class ChessGradedPolicy:
    """``GradedPolicy`` impl for chess. The bound root board (if any) is
    kept on the policy for future position-level caching; the v1 policy
    does not yet use it but the Protocol's per-build construction
    convention reserves the slot."""

    board: Any = None

    def move_base_rate(self, probe: MoveProbe) -> float:
        """Return ``a`` in ``(0, 1)`` derived from the move's child_eval."""
        return _squash_centipawn(probe.child_eval)

    def witness_opinion(
        self,
        *,
        probe: MoveProbe,
        label: str,
        magnitude: int | None,
    ) -> Opinion:
        """Build a witness opinion from ``(label, magnitude)`` (chunk G.1 §5).

        Belief: ``witness_belief(magnitude, label)`` — base 0.55, max 0.70,
        saturated per-prefix. Uncertainty: ``_WITNESS_UNCERTAINTY`` (0.30,
        baseline; calibration deferred to chunk H). Disbelief: residual
        ``1 - belief - u`` clamped to non-negative. Base rate:
        ``_WITNESS_BASE_RATE`` (0.5).
        """
        belief = witness_belief(magnitude, label)
        u = _WITNESS_UNCERTAINTY
        disbelief = max(0.0, 1.0 - belief - u)
        return Opinion(belief, disbelief, u, _WITNESS_BASE_RATE)

    @property
    def edge_trust(self) -> Opinion:
        """The (witness -> move) edge trust opinion."""
        return Opinion.dogmatic_true(EDGE_TRUST_BASE_RATE)


def make_graded_policy(board: Any = None) -> ChessGradedPolicy:
    """Construct a per-build chess graded policy bound to ``board``."""
    return ChessGradedPolicy(board=board)


__all__ = [
    "ChessGradedPolicy",
    "EDGE_TRUST_BASE_RATE",
    "OPINION_LEAF_BASE_RATE",
    "make_graded_policy",
    "witness_belief",
]
