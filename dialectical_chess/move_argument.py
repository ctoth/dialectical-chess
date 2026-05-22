"""Generic per-move argument inputs — the game-agnostic core data model.

This module is the cartridge seam's *generic* side. The generic argumentation
machinery (:mod:`~dialectical_chess.opinion_graph`,
:mod:`~dialectical_chess.decide`) consumes exactly the three types declared
here — :class:`Role`, :class:`Evidence`, :class:`MoveArgument` — plus the
generic :class:`~dialectical_chess.scheme.Tier`. It never sees a chess board,
a chess ``MoveProbe``, a chess objection kind, or any chess policy module.

Every chess-specific input is computed on the chess-cartridge side
(:mod:`~dialectical_chess.argumentation_cartridge`) and handed to the generic
layer as one of these typed values:

* a move's base rate is a plain ``float`` (:attr:`MoveArgument.prior`) — the
  cartridge has already run its static board evaluation and squashed it;
* every support / objection is a typed :class:`Evidence` carrying only generic
  discriminants — its :class:`Role`, its :class:`~dialectical_chess.scheme.Tier`,
  an aggregate ``strength``, an optional ``magnitude``, and whether it
  ``refutes`` (a FACT objection that hard-defeats the move in the crisp layer);
* the worst proven material/terminal loss a move walks into is a generic
  ``int`` magnitude (:attr:`MoveArgument.fact_objection_magnitude`) — the
  cartridge has already classified it as FACT-tier;
* the slowest-loss distance for the empty-survivor fallback is a generic
  ``int`` (:attr:`MoveArgument.empty_survivor_loss_distance`).

Because the generic layer reads only this module and ``Tier``, ``opinion_graph``
and ``decide`` are extractable as-is into a game-agnostic ``dialectical-games``
core: a second game (checkers, Othello) supplies its own cartridge that
produces :class:`MoveArgument` values, and the core is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dialectical_chess.scheme import Tier


class Role(Enum):
    """Whether a piece of evidence argues *for* or *against* a move.

    The generic argumentation layer keys an evidence item's graph channel off
    this role: a :attr:`SUPPORT` item feeds the move's support leaf, an
    :attr:`OBJECTION` item feeds its objection leaf (and, when it ``refutes``
    and is FACT-tier, the crisp filter framework).
    """

    SUPPORT = "support"
    OBJECTION = "objection"


@dataclass(frozen=True)
class Evidence:
    """One typed piece of generic argumentation evidence for a move.

    The generic graph builder and decider read only these fields — never the
    game-specific record behind :attr:`source`:

    * :attr:`label` — a stable identifier, used to name the evidence's graph
      arguments and to key the explainability trace.
    * :attr:`role` — :class:`Role.SUPPORT` or :class:`Role.OBJECTION`.
    * :attr:`tier` — :class:`~dialectical_chess.scheme.Tier.FACT` for a proven
      loss / proven support, :class:`~dialectical_chess.scheme.Tier.HEURISTIC`
      for a positional judgement. The crisp filter selects FACT objections;
      the generic decider orders FACT terms strictly before graded terms.
    * :attr:`strength` — the aggregate evidence strength feeding the opinion
      graph's graded layer (a leaf is built only when this is positive). For an
      objection this is the *residual* strength: any cartridge suppression has
      already been applied.
    * :attr:`magnitude` — an optional proven scalar (e.g. a material loss in
      centipawns, a forced-mate distance). ``None`` when the evidence carries
      no proven magnitude.
    * :attr:`refutes` — ``True`` for a FACT objection that *hard-defeats* the
      move in the crisp Dung filter (a forced-mate / proven-refutation
      objection). ``False`` for a graded objection or a FACT objection that
      only contributes to the decider's FACT term (e.g. a material loss).
    * :attr:`source` — the opaque game-specific evidence record this generic
      item was lifted from. Carried purely for the explainability trace; the
      generic layer never inspects it.
    """

    label: str
    role: Role
    tier: Tier
    strength: int = 0
    magnitude: int | None = None
    refutes: bool = False
    source: Any = None


@dataclass(frozen=True)
class MoveArgument:
    """The generic per-move argument the core argumentation layer consumes.

    One :class:`MoveArgument` carries everything the generic graph builder and
    decider need for a single legal move, all already computed by the chess
    cartridge:

    * :attr:`move_id` — the move's stable identifier (chess: its UCI string).
    * :attr:`prior` — the move node's base rate, in ``(0, 1)``: the cartridge's
      squashed static board evaluation. Used directly as the move node's
      vacuous-opinion base rate; no game policy is folded in here.
    * :attr:`supports` / :attr:`objections` — the typed :class:`Evidence`
      arguing for and against the move. The decider's top-priority FACT term
      is the worst ``magnitude`` over the FACT-tier (:class:`Tier.FACT`)
      objections — read generically off this tuple, never from a precomputed
      scalar (see :func:`~dialectical_chess.decide.worst_fact_objection_magnitude`).
    * :attr:`empty_survivor_loss_distance` — the proven loss *distance* for the
      empty-survivor fallback (chess: a forced-mate distance). A larger value
      is a slower — better — loss when every move is hard-refuted.
    """

    move_id: str
    prior: float
    supports: tuple[Evidence, ...] = field(default_factory=tuple)
    objections: tuple[Evidence, ...] = field(default_factory=tuple)
    empty_survivor_loss_distance: int = 0

    @property
    def crisp_refutations(self) -> tuple[Evidence, ...]:
        """The FACT-tier objections that hard-defeat this move in the filter.

        An objection enters the crisp Dung filter framework iff it is
        :class:`~dialectical_chess.scheme.Tier.FACT` *and* marked as
        :attr:`Evidence.refutes`. This is the generic, ``Tier``-keyed crisp
        gate — it never asks what *kind* of objection it is.
        """
        return tuple(
            objection
            for objection in self.objections
            if objection.tier is Tier.FACT and objection.refutes
        )
