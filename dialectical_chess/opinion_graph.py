"""Opinion-valued argumentation artifacts — the generic graph builder.

This is the generic, game-agnostic half of the cartridge seam. It builds a
``doxa.BipolarOpinionGraph`` and a Dung filter framework from a list of
generic :class:`~dialectical_chess.move_argument.MoveArgument` values, reading
only game-agnostic discriminants:

* a piece of evidence's :class:`~dialectical_chess.move_argument.Role`
  (support / objection) and its aggregate ``strength``;
* a piece of evidence's :class:`~dialectical_chess.scheme.Tier` — the crisp
  filter framework is built from objections whose ``tier`` is ``Tier.FACT``;
* a move's precomputed ``prior`` base rate.

It imports nothing chess-specific — no ``chess`` board, no chess ``MoveProbe``,
no chess objection kind, no chess policy module. Every chess-specific input is
computed cartridge-side (see :mod:`~dialectical_chess.argumentation_cartridge`)
and handed in as one of those generic typed values. This module is therefore
extractable as-is into a game-agnostic ``dialectical-games`` core.
"""

from __future__ import annotations

from dataclasses import dataclass

from argumentation.dung import ArgumentationFramework
from doxa import Opinion
from doxa.argumentation import BipolarOpinionGraph

from dialectical_chess.move_argument import Evidence, MoveArgument, Role
from dialectical_chess.scheme import Tier
from dialectical_chess.tuning import (
    OPINION_EVIDENCE_UNITS_PER_STRENGTH,
    OPINION_LEAF_BASE_RATE,
)

EV: float = OPINION_EVIDENCE_UNITS_PER_STRENGTH
A_ROLE: float = OPINION_LEAF_BASE_RATE
EDGE_TRUST_BASE_RATE: float = 0.5


@dataclass(frozen=True)
class BipolarMoveGraph:
    """The opinion graph for one position plus the move argument index."""

    graph: BipolarOpinionGraph
    move_arg: dict[str, str]


@dataclass(frozen=True)
class MoveArgumentationArtifacts:
    """The single artifact consumed by the filter and decider."""

    graph: BipolarMoveGraph
    filter_af: ArgumentationFramework
    evidence_trace: dict[str, list[Evidence]]

    @property
    def move_arg(self) -> dict[str, str]:
        """Return the single move-argument index owned by the graph artifact."""
        return self.graph.move_arg


def leaf_intrinsic(strength: int) -> Opinion:
    """Encode a positive aggregate evidence strength as a leaf opinion."""
    if strength <= 0:
        raise ValueError("leaf intrinsic strength must be positive")
    return Opinion.from_evidence(strength * EV, 0.0, A_ROLE)


def build_argumentation_artifacts(
    move_arguments: list[MoveArgument],
) -> MoveArgumentationArtifacts:
    """Build the opinion graph, move index, filter AF, and evidence trace.

    Consumes a list of generic :class:`MoveArgument` values. For each move it
    builds, in the bipolar opinion graph: a vacuous move node at the move's
    precomputed ``prior``, an aggregate support leaf (when the move carries
    positive-strength support), and an aggregate objection leaf (when the move
    carries positive-strength objections). In the separate Dung filter
    framework it adds one defeater argument per ``Tier.FACT`` refuting
    objection — the crisp hard gate keys on :class:`~dialectical_chess.scheme.Tier`.
    """
    arguments: set[str] = set()
    intrinsic: dict[str, Opinion] = {}
    supports: set[tuple[str, str]] = set()
    attacks: set[tuple[str, str]] = set()
    edge_opinions: dict[tuple[str, str], Opinion] = {}
    move_arg: dict[str, str] = {}
    filter_arguments: set[str] = set()
    filter_defeats: set[tuple[str, str]] = set()
    evidence_trace: dict[str, list[Evidence]] = {}
    edge_trust = Opinion.dogmatic_true(EDGE_TRUST_BASE_RATE)

    for argument in move_arguments:
        move = move_argument_id(argument.move_id)
        move_arg[argument.move_id] = move
        arguments.add(move)
        filter_arguments.add(move)

        support_items = [
            evidence
            for evidence in argument.supports
            if evidence.role is Role.SUPPORT and evidence.strength > 0
        ]
        support_strength = sum(evidence.strength for evidence in support_items)
        if support_strength > 0:
            support = support_argument_id(argument.move_id)
            arguments.add(support)
            intrinsic[support] = leaf_intrinsic(support_strength)
            edge = (support, move)
            supports.add(edge)
            edge_opinions[edge] = edge_trust
            evidence_trace[support] = list(support_items)

        objection_items = [
            evidence
            for evidence in argument.objections
            if evidence.role is Role.OBJECTION and evidence.strength > 0
        ]
        objection_strength = sum(evidence.strength for evidence in objection_items)
        # The move-node base rate is the precomputed disjoint prior only — no
        # game policy is folded in here. The cartridge has already classified
        # any proven material loss as a FACT-tier decision term carried on the
        # MoveArgument, not as a base-rate nudge.
        intrinsic[move] = Opinion.vacuous(argument.prior)
        if objection_strength > 0:
            objection_arg = objection_argument_id(argument.move_id)
            arguments.add(objection_arg)
            intrinsic[objection_arg] = leaf_intrinsic(objection_strength)
            edge = (objection_arg, move)
            attacks.add(edge)
            edge_opinions[edge] = edge_trust
            evidence_trace[objection_arg] = objection_items

        for evidence in argument.crisp_refutations:
            refute = refutation_argument_id(argument.move_id, evidence)
            filter_arguments.add(refute)
            filter_defeats.add((refute, move))
            evidence_trace[refute] = [evidence]

    graph = BipolarOpinionGraph(
        arguments=frozenset(arguments),
        intrinsic=intrinsic,
        supports=frozenset(supports),
        attacks=frozenset(attacks),
        edge_opinions=edge_opinions,
    )
    filter_af = ArgumentationFramework(
        arguments=frozenset(filter_arguments),
        defeats=frozenset(filter_defeats),
    )
    bmg = BipolarMoveGraph(graph=graph, move_arg=move_arg)
    return MoveArgumentationArtifacts(
        graph=bmg,
        filter_af=filter_af,
        evidence_trace=evidence_trace,
    )


def move_argument_id(move_id: str) -> str:
    return f"move:{move_id}"


def support_argument_id(move_id: str) -> str:
    return f"support:{move_id}"


def objection_argument_id(move_id: str) -> str:
    return f"objection:{move_id}"


def refutation_argument_id(move_id: str, evidence: Evidence) -> str:
    return f"refute:{move_id}:{evidence.label}"


# A FACT objection that hard-defeats a move in the crisp filter — the generic
# Tier-keyed crisp gate. Kept as a named predicate so the decider and tests
# can ask "is this a refuting FACT objection" without re-spelling the rule.
def is_crisp_refutation(evidence: Evidence) -> bool:
    """Return whether ``evidence`` hard-defeats its move in the crisp filter."""
    return (
        evidence.role is Role.OBJECTION
        and evidence.tier is Tier.FACT
        and evidence.refutes
    )
