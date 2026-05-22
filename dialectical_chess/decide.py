"""Opinion-valued move decider — the generic lexicographic FACT-then-graded key.

The decider is a single lexicographic key (design D2, modelled on
dialectical-checkers' decider): the FACT-tier term — the worst proven loss a
move walks into — is ordered strictly before the graded term — the move's
opinion-valued ``expectation()``. A FACT decision always dominates a graded
one.

This module is the generic, game-agnostic half of the cartridge seam. It
consumes generic :class:`~dialectical_chess.move_argument.MoveArgument` values
and reads only game-agnostic discriminants — its FACT term is the worst
``magnitude`` over the move's :class:`~dialectical_chess.scheme.Tier.FACT`
objection evidence, read directly off the generic
:class:`~dialectical_chess.move_argument.Evidence` tuple; its fallback term is
the move's empty-survivor loss distance; its graded term is the resolved
opinion ``expectation()``. It imports nothing chess-specific — no ``chess``
board, no chess ``MoveProbe``, no chess policy module — and is extractable
as-is into a game-agnostic ``dialectical-games`` core. Every chess-specific
input (the proven material loss, the forced-mate distance) is computed
cartridge-side and handed in on the
:class:`~dialectical_chess.move_argument.MoveArgument`, or — for the
potentially expensive empty-survivor loss-distance proof — supplied as a
generic callback the decider invokes lazily, only on the empty-survivor
fallback path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from doxa import Opinion
from doxa.argumentation import evaluate

from dialectical_chess.move_argument import MoveArgument, Role
from dialectical_chess.opinion_graph import (
    MoveArgumentationArtifacts,
    build_argumentation_artifacts,
)
from dialectical_chess.scheme import Tier
from dialectical_chess.skeptical_filter import skeptical_survivors

# The empty-survivor loss-distance source. The generic decider needs a move's
# proven-loss *distance* only on the empty-survivor fallback path (every move
# hard-refuted). Computing it can be expensive (a game's forced-loss proof),
# so the cartridge supplies it as a callback the decider invokes *lazily* —
# only for the moves it actually ranks in that fallback, never eagerly for
# every move. ``None`` means "no cartridge source"; the decider then reads the
# move argument's own ``empty_survivor_loss_distance`` field (default 0).
EmptySurvivorLoss = Callable[[MoveArgument], int]


@dataclass(frozen=True)
class ArgumentationDecision:
    """The opinion-valued decision over one legal-move argument set."""

    selected: MoveArgument
    empty_survivors: bool
    move_opinion: dict[str, Opinion]


def decide(
    move_arguments: list[MoveArgument],
    *,
    empty_survivor_loss: EmptySurvivorLoss | None = None,
) -> ArgumentationDecision:
    """Return the argumentation decision for the input generic move arguments.

    Builds the opinion graph and the Dung filter, takes the grounded crisp
    survivors (or — when every move is hard-refuted — falls back to all
    moves), and picks the survivor maximising the lexicographic selection key.

    ``empty_survivor_loss`` is the cartridge's proven-loss-distance source for
    the empty-survivor fallback. It is invoked **lazily** — only when there
    are no crisp survivors, and then only once per move actually ranked in
    that fallback. On the normal (crisp-survivor) path it is never called, so
    a game's forced-loss proof never runs when a clean move exists. When it is
    ``None`` the decider falls back to each move argument's own
    ``empty_survivor_loss_distance`` field.
    """
    if not move_arguments:
        raise ValueError("position has no legal moves")
    artifacts: MoveArgumentationArtifacts = build_argumentation_artifacts(
        move_arguments
    )
    opinions = evaluate(artifacts.graph.graph)
    survivors = skeptical_survivors(artifacts)
    empty_survivors = not survivors
    pool = survivors if survivors else {arg.move_id for arg in move_arguments}
    if empty_survivors:
        selected = max(
            (arg for arg in move_arguments if arg.move_id in pool),
            key=lambda arg: empty_survivors_selection_key(
                arg, artifacts, opinions, empty_survivor_loss
            ),
        )
    else:
        selected = max(
            (arg for arg in move_arguments if arg.move_id in pool),
            key=lambda arg: expectation_selection_key(arg, artifacts, opinions),
        )
    return ArgumentationDecision(
        selected=selected,
        empty_survivors=empty_survivors,
        move_opinion={
            move_id: opinions[argument]
            for move_id, argument in artifacts.move_arg.items()
        },
    )


def worst_fact_objection_magnitude(argument: MoveArgument) -> int:
    """The worst proven loss ``argument`` walks into — the decider's FACT term.

    Computed generically, by reading the FACT-tier objection evidence the
    cartridge attached to the move argument: the term is the largest
    :attr:`~dialectical_chess.move_argument.Evidence.magnitude` over every
    objection whose :class:`~dialectical_chess.scheme.Tier` is ``Tier.FACT``.
    This keys strictly on ``Tier`` — never on what kind of objection it is —
    and is therefore the game-agnostic FACT term: any game's cartridge that
    tags its proven losses ``Tier.FACT`` with a magnitude feeds it unchanged.
    A move with no FACT-tier objection scores 0.
    """
    return max(
        (
            objection.magnitude
            for objection in argument.objections
            if objection.role is Role.OBJECTION
            and objection.tier is Tier.FACT
            and objection.magnitude is not None
        ),
        default=0,
    )


def expectation_selection_key(
    argument: MoveArgument,
    artifacts: MoveArgumentationArtifacts,
    opinions: dict[str, Opinion],
) -> tuple[int, float, str]:
    """The lexicographic selection key for a crisp survivor (design D2).

    The key is consumed by ``max`` — larger is better. Its terms, in order:

    1. the FACT term — the negated worst proven loss
       (:func:`worst_fact_objection_magnitude`). A move with no proven loss
       scores 0 here and outranks every move that walks into one; among moves
       that do, the smaller loss outranks the larger. This term is the
       FACT-tier prefix of the key: it dominates the graded term completely
       (design D2 — fact-as-highest-value). It is computed by reading the
       generic FACT-tier (``Tier.FACT``) objection evidence's magnitude.
    2. the graded term — the move's opinion-valued ``expectation()`` over the
       crisp survivors.
    3. the deterministic tiebreak — the move id (the lexicographically
       largest id wins an exact tie).
    """
    expectation = opinions[artifacts.move_arg[argument.move_id]].expectation()
    return (
        -worst_fact_objection_magnitude(argument),
        expectation,
        argument.move_id,
    )


def empty_survivors_selection_key(
    argument: MoveArgument,
    artifacts: MoveArgumentationArtifacts,
    opinions: dict[str, Opinion],
    empty_survivor_loss: EmptySurvivorLoss | None = None,
) -> tuple[int, float, str]:
    """The selection key for the empty-survivor fallback (design v2 §5d).

    When every legal move is hard-refuted there is no clean choice; the
    decider picks the least-bad move — the slowest proven loss (a larger
    distance is a slower loss), then the highest graded ``expectation()``,
    then the largest move id.

    The slowest-loss distance comes from the ``empty_survivor_loss`` callback
    when one is supplied — invoked here, lazily, only because this key
    function runs only on the empty-survivor path. When no callback is
    supplied it falls back to the move argument's own
    :attr:`~dialectical_chess.move_argument.MoveArgument.empty_survivor_loss_distance`.
    """
    loss_distance = (
        empty_survivor_loss(argument)
        if empty_survivor_loss is not None
        else argument.empty_survivor_loss_distance
    )
    return (
        loss_distance,
        opinions[artifacts.move_arg[argument.move_id]].expectation(),
        argument.move_id,
    )
