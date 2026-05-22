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
    DefeaterEvidence,
    ObjectionKind,
    EvidenceWorld,
    ObjectionEvidence,
    ReplyEvidence,
    SupportEvidence,
    SupportKind,
    defeater_evidence as make_defeater_evidence,
    has_search_refutation_at_most,
    is_forced_mate_refutation,
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
        material_safety_prior_penalty = 0
        for objection in objection_items:
            residual = effective_objection_strength(probe, objection)
            if residual <= 0:
                continue
            objection_strength += residual
            material_safety_prior_penalty += material_safety_prior_penalty_for(
                probe, objection
            )
            effective_objections.append(objection)
        intrinsic[move] = Opinion.vacuous(
            squash(static_prior(probe) - material_safety_prior_penalty)
        )
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
    strength = 0
    if isinstance(objection, ObjectionEvidence):
        strength = objection.objection_strength
    elif isinstance(objection, ReplyEvidence):
        strength = objection.reply_attack_strength
    if strength <= 0:
        return 0
    suppression = sum(
        defeater_strength_value(defeater)
        for defeater in objection_defeater_evidence(probe, objection)
    )
    return max(0, strength - suppression)


def defeater_strength_value(evidence: ArgumentEvidence) -> int:
    if isinstance(evidence, DefeaterEvidence):
        return evidence.defeater_strength
    if isinstance(evidence, ReplyEvidence):
        return evidence.defense_strength
    return 0


def objection_defeater_evidence(
    probe: MoveProbe,
    objection: ArgumentEvidence,
) -> tuple[ArgumentEvidence, ...]:
    defeaters: list[ArgumentEvidence] = []
    if isinstance(objection, ReplyEvidence):
        if objection.defense_strength > 0:
            defeaters.append(objection)
        return tuple(defeaters)
    if not isinstance(objection, ObjectionEvidence):
        return tuple(defeaters)
    if (
        objection.objection_kind == ObjectionKind.QUEEN_BLUNDER
        and has_compensating_forcing_pressure(probe)
    ):
        defeaters.append(synthetic_defeater_evidence(DefeaterKind.COMPENSATING_FORCING_PRESSURE))
    if (
        objection.objection_kind == ObjectionKind.MOVED_PIECE_EN_PRIS
        and objection.moved_piece_en_pris_value is not None
        and objection.moved_piece_en_pris_value >= 300
    ):
        if has_compensating_tactical_pressure(probe):
            defeaters.append(synthetic_defeater_evidence(DefeaterKind.COMPENSATING_TACTICAL_PRESSURE))
        if has_forcing_material_gain(probe):
            defeaters.append(synthetic_defeater_evidence(DefeaterKind.FORCING_MATERIAL_GAIN))
    if (
        objection.objection_kind == ObjectionKind.OPENING_PREMATURE_MINOR_CHECK
        and has_typed_reason_defeater(probe, DefeaterKind.SEARCH_SUPPORT)
    ):
        defeaters.append(synthetic_defeater_evidence(DefeaterKind.SEARCH_SUPPORT))
    if (
        objection.objection_kind
        in {
            ObjectionKind.FLANK_PAWN_WEAKENING,
            ObjectionKind.FLANK_PAWN_LUNGE,
        }
        and has_typed_reason_defeater(probe, DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE)
    ):
        defeaters.append(synthetic_defeater_evidence(DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE))
    return tuple(defeaters)


def synthetic_defeater_evidence(defeater_kind: DefeaterKind) -> ArgumentEvidence:
    return make_defeater_evidence(
        f"defeater:{defeater_kind.value}",
        world=EvidenceWorld.PROCEDURAL,
        defeater_kind=defeater_kind,
        defeater_strength=defeater_strength_for(defeater_kind),
    )


def defeater_strength_for(defeater_kind: DefeaterKind) -> int:
    match defeater_kind:
        case DefeaterKind.SEARCH_SUPPORT:
            return 97
        case (
            DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE
            | DefeaterKind.COMPENSATING_FORCING_PRESSURE
            | DefeaterKind.FORCING_MATERIAL_GAIN
        ):
            return 33
        case DefeaterKind.COMPENSATING_TACTICAL_PRESSURE:
            return 17


def has_compensating_tactical_pressure(probe: MoveProbe) -> bool:
    return any(
        isinstance(evidence, SupportEvidence | DefeaterEvidence)
        and evidence.tactical_threat_value >= COMPENSATING_TACTICAL_THREAT_THRESHOLD
        for evidence in probe.reason_evidence
    )


def has_compensating_forcing_pressure(probe: MoveProbe) -> bool:
    return has_compensating_tactical_pressure(probe) and (
        probe.gives_check or material_or_promotion_gain(probe) > 0
    )


def has_forcing_material_gain(probe: MoveProbe) -> bool:
    return probe.gives_check and material_or_promotion_gain(probe) > 0


def has_typed_reason_defeater(probe: MoveProbe, defeater_kind: DefeaterKind) -> bool:
    return any(
        isinstance(evidence, DefeaterEvidence) and evidence.defeater_kind == defeater_kind
        for evidence in probe.reason_evidence
    )


def material_or_promotion_gain(probe: MoveProbe) -> int:
    return probe.captured_value + probe.promotion_value


def material_safety_prior_penalty_for(
    probe: MoveProbe,
    objection: ArgumentEvidence,
) -> int:
    if not isinstance(objection, ObjectionEvidence):
        return 0
    if objection.objection_kind == ObjectionKind.QUEEN_FLANK_INVASION:
        if has_development_reason(probe) and not has_search_refutation_at_most(
            probe.objection_evidence, -300
        ):
            return 0
        return 300
    if (
        objection.objection_kind == ObjectionKind.MOVED_PIECE_EN_PRIS
        and objection.moved_piece_en_pris_value is not None
        and has_search_refutation_at_most(probe.objection_evidence, -400)
    ):
        return 4 * objection.moved_piece_en_pris_value
    if (
        objection.objection_kind == ObjectionKind.IGNORED_HANGING_PIECE
        and has_search_refutation_at_most(probe.objection_evidence, -400)
    ):
        return 300
    return 0


def has_development_reason(probe: MoveProbe) -> bool:
    return any(
        isinstance(evidence, SupportEvidence | DefeaterEvidence)
        and evidence.support_kind == SupportKind.DEVELOPMENT
        for evidence in probe.reason_evidence
    )


def move_argument_id(uci: str) -> str:
    return f"move:{uci}"


def support_argument_id(uci: str) -> str:
    return f"support:{uci}"


def objection_argument_id(uci: str) -> str:
    return f"objection:{uci}"


def refutation_argument_id(uci: str, evidence: ArgumentEvidence) -> str:
    return f"refute:{uci}:{evidence.label}"
