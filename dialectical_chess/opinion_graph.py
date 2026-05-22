"""Opinion-valued argumentation artifacts — the generic graph builder.

This is the generic, game-agnostic half of the cartridge seam. It builds a
``doxa.BipolarOpinionGraph`` and a Dung filter framework from typed evidence,
reading only the generic discriminants — a piece of evidence's role
(support / objection / defeater / reply) and its strength. It never names a
chess objection kind: the chess-specific suppression policy lives behind the
``suppression`` hook (design D3).
"""

from __future__ import annotations

from dataclasses import dataclass

from argumentation.dung import ArgumentationFramework
from doxa import Opinion
from doxa.argumentation import BipolarOpinionGraph

from dialectical_chess.arguments import MoveProbe
from dialectical_chess.evidence import (
    ArgumentEvidence,
    DefeaterEvidence,
    ObjectionEvidence,
    ReplyEvidence,
    SupportEvidence,
    is_forced_mate_refutation,
)
from dialectical_chess.suppression import suppressing_defeaters
from dialectical_chess.static_prior import squash, static_prior
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
    evidence_trace: dict[str, list[ArgumentEvidence]]

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
    probes: list[MoveProbe],
) -> MoveArgumentationArtifacts:
    """Build the opinion graph, move index, filter AF, and evidence trace."""
    arguments: set[str] = set()
    intrinsic: dict[str, Opinion] = {}
    supports: set[tuple[str, str]] = set()
    attacks: set[tuple[str, str]] = set()
    edge_opinions: dict[tuple[str, str], Opinion] = {}
    move_arg: dict[str, str] = {}
    filter_arguments: set[str] = set()
    filter_defeats: set[tuple[str, str]] = set()
    evidence_trace: dict[str, list[ArgumentEvidence]] = {}
    edge_trust = Opinion.dogmatic_true(EDGE_TRUST_BASE_RATE)

    for probe in probes:
        move = move_argument_id(probe.uci)
        move_arg[probe.uci] = move
        arguments.add(move)
        filter_arguments.add(move)

        support_items = [
            evidence
            for evidence in probe.reason_evidence
            if isinstance(evidence, SupportEvidence | DefeaterEvidence)
            and evidence.supports_argument
            and evidence.support_strength > 0
        ]
        support_strength = sum(evidence.support_strength for evidence in support_items)
        if support_strength > 0:
            support = support_argument_id(probe.uci)
            arguments.add(support)
            intrinsic[support] = leaf_intrinsic(support_strength)
            edge = (support, move)
            supports.add(edge)
            edge_opinions[edge] = edge_trust
            evidence_trace[support] = list(support_items)

        objection_items = [
            *probe.objection_evidence,
            *probe.reply_attack_evidence,
        ]
        effective_objections: list[ArgumentEvidence] = []
        objection_strength = 0
        for objection in objection_items:
            residual = effective_objection_strength(probe, objection)
            if residual <= 0:
                continue
            objection_strength += residual
            effective_objections.append(objection)
        # The move-node base rate is the disjoint static prior only — no
        # chess-specific penalty is folded in. Chess's material-safety loss is
        # now an honest FACT-tier decision term (``suppression.fact_material_
        # loss``), consulted by the decider, not smuggled into this base rate.
        intrinsic[move] = Opinion.vacuous(squash(static_prior(probe)))
        if objection_strength > 0:
            objection_arg = objection_argument_id(probe.uci)
            arguments.add(objection_arg)
            intrinsic[objection_arg] = leaf_intrinsic(objection_strength)
            edge = (objection_arg, move)
            attacks.add(edge)
            edge_opinions[edge] = edge_trust
            evidence_trace[objection_arg] = effective_objections

        for evidence in objection_items:
            if not is_forced_mate_refutation(evidence):
                continue
            refute = refutation_argument_id(probe.uci, evidence)
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


def effective_objection_strength(
    probe: MoveProbe,
    objection: ArgumentEvidence,
) -> int:
    """Return an objection's strength after chess-policy suppression.

    The generic residual-suppression rule (design v2 §1e): an objection's
    strength is cancelled, at aggregation time, by the suppression strength of
    every defeater the chess :mod:`~dialectical_chess.suppression` policy
    returns for it. This function reads only generic strengths — the chess
    suppression knowledge lives entirely behind the ``suppressing_defeaters``
    hook (design D3).
    """
    strength = 0
    if isinstance(objection, ObjectionEvidence):
        strength = objection.objection_strength
    elif isinstance(objection, ReplyEvidence):
        strength = objection.reply_attack_strength
    if strength <= 0:
        return 0
    suppression = sum(
        defeater_strength_value(defeater)
        for defeater in suppressing_defeaters(probe, objection)
    )
    return max(0, strength - suppression)


def defeater_strength_value(evidence: ArgumentEvidence) -> int:
    """Return the suppression strength a defeater / defended reply carries."""
    if isinstance(evidence, DefeaterEvidence):
        return evidence.defeater_strength
    if isinstance(evidence, ReplyEvidence):
        return evidence.defense_strength
    return 0


def move_argument_id(uci: str) -> str:
    return f"move:{uci}"


def support_argument_id(uci: str) -> str:
    return f"support:{uci}"


def objection_argument_id(uci: str) -> str:
    return f"objection:{uci}"


def refutation_argument_id(uci: str, evidence: ArgumentEvidence) -> str:
    return f"refute:{uci}:{evidence.label}"
