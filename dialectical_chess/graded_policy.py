"""Chess cartridge: :class:`ChessGradedPolicy`.

Implements the core ``dialectical_games.arguments.GradedPolicy`` Protocol.
A per-build policy bound to the root board (so position-level features can
be cached once).

* ``move_base_rate(probe)`` — squashes ``probe.child_eval`` (cartridge-pre-
  computed centipawn-scale int) into ``(0, 1)``. The cartridge sets
  ``child_eval = static_prior(probe) * 1000`` at probe time, so the policy
  re-squashes to ``(0.01, 0.99)``.
* ``witness_opinion`` — per the Phase-3 foreman directive 3, chess HEURISTIC
  vocabulary does NOT enter the core graded layer this cycle: the chess
  cartridge emits NO HEURISTIC labels on the core MoveProbe's tuples. The
  Protocol method is implemented as a vacuous-default for symmetry but is
  not invoked in practice.
* ``edge_trust`` — chess uses the pre-Phase-3 ``EDGE_TRUST_BASE_RATE = 0.5``
  as a dogmatic-true opinion (the chess opinion-graph builder's prior
  edge trust).
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


def _squash_centipawn(centipawn: int) -> float:
    """Squash a centipawn-scale evaluation to ``(0.01, 0.99)``."""
    raw = 0.5 + 0.5 * math.tanh(centipawn / (_TAU_SCALE * 1000.0))
    return max(_TAU_CLAMP_LO, min(_TAU_CLAMP_HI, raw))


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
        """Vacuous default — chess emits no HEURISTIC labels into the core
        graded layer this cycle (foreman directive 3). Never invoked in
        practice; implemented for Protocol symmetry."""
        return Opinion.vacuous(OPINION_LEAF_BASE_RATE)

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
]
