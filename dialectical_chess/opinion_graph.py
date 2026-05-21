"""Opinion-valued Phase-2 argumentation artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from argumentation.dung import ArgumentationFramework
from doxa import Opinion
from doxa.argumentation import BipolarOpinionGraph

from dialectical_chess.arguments import MoveProbe
from dialectical_chess.evidence import (
    COMPENSATING_TACTICAL_THREAT_THRESHOLD,
    ArgumentEvidence,
    DefeaterKind,
    ObjectionKind,
    is_forced_mate_refutation,
    to_argument_evidence,
)
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
    move_arg: dict[str, str]
    filter_af: ArgumentationFramework
    evidence_trace: dict[str, list[ArgumentEvidence]]


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
        intrinsic[move] = Opinion.vacuous(squash(static_prior(probe)))
        filter_arguments.add(move)

        support_items = [
            evidence
            for evidence in probe.reason_evidence
            if evidence.supports_argument and evidence.support_strength > 0
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
        move_arg=move_arg,
        filter_af=filter_af,
        evidence_trace=evidence_trace,
    )


def effective_objection_strength(
    probe: MoveProbe,
    objection: ArgumentEvidence,
) -> int:
    strength = objection.objection_strength + objection.reply_attack_strength
    if strength <= 0:
        return 0
    suppression = sum(
        defeater.defeater_strength + defeater.defense_strength
        for defeater in objection_defeater_evidence(probe, objection)
    )
    return max(0, strength - suppression)


def objection_defeater_evidence(
    probe: MoveProbe,
    objection: ArgumentEvidence,
) -> tuple[ArgumentEvidence, ...]:
    defeaters: list[ArgumentEvidence] = []
    if (
        objection.objection_kind == ObjectionKind.QUEEN_BLUNDER
        and has_compensating_forcing_pressure(probe)
    ):
        defeaters.append(defeater_evidence(DefeaterKind.COMPENSATING_FORCING_PRESSURE))
    if (
        objection.objection_kind == ObjectionKind.MOVED_PIECE_EN_PRIS
        and objection.moved_piece_en_pris_value is not None
        and objection.moved_piece_en_pris_value >= 300
    ):
        if has_compensating_tactical_pressure(probe):
            defeaters.append(defeater_evidence(DefeaterKind.COMPENSATING_TACTICAL_PRESSURE))
        if has_forcing_material_gain(probe):
            defeaters.append(defeater_evidence(DefeaterKind.FORCING_MATERIAL_GAIN))
    if (
        objection.objection_kind == ObjectionKind.OPENING_PREMATURE_MINOR_CHECK
        and has_typed_reason_defeater(probe, DefeaterKind.SEARCH_SUPPORT)
    ):
        defeaters.append(defeater_evidence(DefeaterKind.SEARCH_SUPPORT))
    if (
        objection.objection_kind
        in {
            ObjectionKind.FLANK_PAWN_WEAKENING,
            ObjectionKind.FLANK_PAWN_LUNGE,
        }
        and has_typed_reason_defeater(probe, DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE)
    ):
        defeaters.append(defeater_evidence(DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE))
    if objection.defense_strength > 0:
        defeaters.append(objection)
    return tuple(defeaters)


def defeater_evidence(defeater_kind: DefeaterKind) -> ArgumentEvidence:
    return to_argument_evidence(f"defeater:{defeater_kind.value}")


def has_compensating_tactical_pressure(probe: MoveProbe) -> bool:
    return any(
        evidence.tactical_threat_value >= COMPENSATING_TACTICAL_THREAT_THRESHOLD
        for evidence in probe.reason_evidence
    )


def has_compensating_forcing_pressure(probe: MoveProbe) -> bool:
    return has_compensating_tactical_pressure(probe) and (
        probe.gives_check or material_or_promotion_gain(probe) > 0
    )


def has_forcing_material_gain(probe: MoveProbe) -> bool:
    return probe.gives_check and material_or_promotion_gain(probe) > 0


def has_typed_reason_defeater(probe: MoveProbe, defeater_kind: DefeaterKind) -> bool:
    return any(evidence.defeater_kind == defeater_kind for evidence in probe.reason_evidence)


def material_or_promotion_gain(probe: MoveProbe) -> int:
    return probe.captured_value + probe.promotion_value


def move_argument_id(uci: str) -> str:
    return f"move:{uci}"


def support_argument_id(uci: str) -> str:
    return f"support:{uci}"


def objection_argument_id(uci: str) -> str:
    return f"objection:{uci}"


def refutation_argument_id(uci: str, evidence: ArgumentEvidence) -> str:
    return f"refute:{uci}:{evidence.label}"
