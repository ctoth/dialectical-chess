"""Argument graph construction and move selection for dialectical chess."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from argumentation.dung import ArgumentationFramework, grounded_extension
from argumentation.ranking import categoriser_scores
from argumentation.vaf import ValueBasedArgumentationFramework

from dialectical_chess.evidence import (
    ArgumentEvidence,
    DefeaterKind,
    ObjectionKind,
    LARGE_SEARCH_REFUTATION_THRESHOLD,
    is_defensible_reply_attack,
    is_forced_mate_refutation as evidence_is_forced_mate_refutation,
    is_undefended_reply_capture,
    to_argument_evidence,
)

POSITIONAL_SCORE_BONUS = 25
COMPENSATING_TACTICAL_THREAT_THRESHOLD = 700


@dataclass(frozen=True)
class MoveProbe:
    uci: str
    san: str
    score: int
    is_checkmate: bool
    gives_check: bool
    is_capture: bool
    captured_value: int
    promotion_value: int
    reasons: tuple[str, ...]
    objections: tuple[str, ...]
    reply_attacks: tuple[str, ...] = ()
    search_score: int | None = None
    search_line: tuple[str, ...] = ()
    smt_witnesses: tuple[str, ...] = ()
    optimizer_trace: dict[str, Any] = field(default_factory=dict)
    reason_evidence: tuple[ArgumentEvidence, ...] = field(init=False, repr=False, compare=False)
    objection_evidence: tuple[ArgumentEvidence, ...] = field(init=False, repr=False, compare=False)
    reply_attack_evidence: tuple[ArgumentEvidence, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reason_evidence",
            tuple(to_argument_evidence(reason) for reason in self.reasons),
        )
        object.__setattr__(
            self,
            "objection_evidence",
            tuple(to_argument_evidence(objection) for objection in self.objections),
        )
        object.__setattr__(
            self,
            "reply_attack_evidence",
            tuple(to_argument_evidence(reply_attack) for reply_attack in self.reply_attacks),
        )


@dataclass(frozen=True)
class RootArgumentGraph:
    arguments: frozenset[str]
    defeats: frozenset[tuple[str, str]]
    move_arguments: dict[str, str]
    grounded_extension: frozenset[str]
    ranking: dict[str, Any]
    evidence: dict[str, ArgumentEvidence] = field(default_factory=dict)


def choose_move(
    probes: list[MoveProbe],
    graph: RootArgumentGraph | None = None,
) -> MoveProbe:
    if not probes:
        raise ValueError("position has no legal moves")
    graph = graph or build_root_argument_graph(probes)
    return sorted(
        probes,
        key=lambda probe: categoriser_decision_key(probe, graph),
    )[0]


def categoriser_decision_key(
    probe: MoveProbe,
    graph: RootArgumentGraph,
) -> tuple[Any, ...]:
    move_arg = graph.move_arguments[probe.uci]
    ranking_scores = graph.ranking["scores"]
    move_rank = float(ranking_scores.get(move_arg, 0.0))
    return (-move_rank, -probe.score, probe.uci)


def accepted_tactical_support_count(probe: MoveProbe, graph: RootArgumentGraph) -> int:
    return _accepted_reason_count(probe, graph, lambda evidence: evidence.counts_as_tactical)


def accepted_positional_support_count(probe: MoveProbe, graph: RootArgumentGraph) -> int:
    return _accepted_reason_count(probe, graph, lambda evidence: evidence.counts_as_positional)


def effective_positional_support_count(
    probe: MoveProbe,
    graph: RootArgumentGraph,
    mode: str | None = None,
) -> int:
    if mode is None:
        mode = positional_support_mode(graph)
    if mode != "quiet":
        return 0
    return accepted_positional_support_count(probe, graph)


def positional_support_mode(graph: RootArgumentGraph, *, include_positional: bool = True) -> str:
    if not include_positional:
        return "disabled"
    if any(
        argument.startswith("reason:")
        and graph.evidence[argument].counts_as_tactical
        and argument in graph.grounded_extension
        for argument in graph.arguments
    ):
        return "tactical_gated"
    return "quiet"


def effective_score(probe: MoveProbe, mode: str) -> int:
    if mode == "quiet":
        return probe.score
    return probe.score - POSITIONAL_SCORE_BONUS * soft_positional_reason_count(probe)


def positional_reason_count(probe: MoveProbe) -> int:
    return sum(1 for evidence in probe.reason_evidence if evidence.counts_as_positional)


def soft_positional_reason_count(probe: MoveProbe) -> int:
    return sum(
        1
        for evidence in probe.reason_evidence
        if evidence.counts_as_positional
        and not is_concrete_non_queen_piece_safety(evidence)
    )


def is_concrete_non_queen_piece_safety(evidence: ArgumentEvidence) -> bool:
    return evidence.defended_piece_value is not None and evidence.defended_piece_value < 900


def material_or_promotion_gain(probe: MoveProbe) -> int:
    return probe.captured_value + probe.promotion_value


def severe_objection_count(probe: MoveProbe) -> int:
    return sum(evidence.objection_strength for evidence in probe.objection_evidence)


def has_forced_mate_refutation(probe: MoveProbe) -> bool:
    return any(evidence_is_forced_mate_refutation(evidence) for evidence in probe.objection_evidence)


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
    if objection.objection_kind == ObjectionKind.OPENING_PREMATURE_MINOR_CHECK and has_typed_reason_defeater(
        probe,
        DefeaterKind.SEARCH_SUPPORT,
    ):
        defeaters.append(defeater_evidence(DefeaterKind.SEARCH_SUPPORT))
    if objection.objection_kind in {
        ObjectionKind.FLANK_PAWN_WEAKENING,
        ObjectionKind.FLANK_PAWN_LUNGE,
    } and has_typed_reason_defeater(probe, DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE):
        defeaters.append(defeater_evidence(DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE))
    return tuple(defeaters)


def defeater_evidence(defeater_kind: DefeaterKind) -> ArgumentEvidence:
    return to_argument_evidence(f"defeater:{defeater_kind.value}")


def _accepted_reason_count(
    probe: MoveProbe,
    graph: RootArgumentGraph,
    predicate,
) -> int:
    return sum(
        1
        for reason in probe.reason_evidence
        if predicate(graph.evidence[evidence_argument_id("reason", probe.uci, reason)])
        and evidence_argument_id("reason", probe.uci, reason) in graph.grounded_extension
    )


def evidence_argument_id(role: str, uci: str, evidence: ArgumentEvidence) -> str:
    return f"{role}:{uci}:{evidence.label}"


def add_typed_attack(
    arguments: set[str],
    defeats: set[tuple[str, str]],
    evidence_by_argument: dict[str, ArgumentEvidence],
    *,
    attacker: str,
    target: str,
    evidence: ArgumentEvidence,
    strength: int,
) -> tuple[str, ...]:
    if strength <= 0:
        return ()
    arguments.add(attacker)
    evidence_by_argument[attacker] = evidence
    defeats.add((attacker, target))
    attackers = [attacker]
    for index in range(1, strength):
        weighted_attacker = f"{attacker}:strength:{index}"
        arguments.add(weighted_attacker)
        evidence_by_argument[weighted_attacker] = evidence
        defeats.add((weighted_attacker, target))
        attackers.append(weighted_attacker)
    return tuple(attackers)


def accepted_defense_count(probe: MoveProbe, graph: RootArgumentGraph) -> int:
    return sum(
        1
        for reply_attack in probe.reply_attacks
        if f"defense:{probe.uci}:{reply_attack}" in graph.grounded_extension
    )


def unresolved_attack_count(probe: MoveProbe, graph: RootArgumentGraph) -> int:
    return sum(
        1
        for reply_attack in probe.reply_attacks
        if f"reply_attack:{probe.uci}:{reply_attack}" in graph.grounded_extension
    )


def build_root_argument_graph(probes: list[MoveProbe]) -> RootArgumentGraph:
    arguments: set[str] = set()
    defeats: set[tuple[str, str]] = set()
    evidence: dict[str, ArgumentEvidence] = {}
    move_args = {probe.uci: f"move:{probe.uci}" for probe in probes}

    for probe in probes:
        move_arg = move_args[probe.uci]
        doubt_arg = f"doubt:{probe.uci}"
        arguments.add(move_arg)
        arguments.add(doubt_arg)
        defeats.add((doubt_arg, move_arg))
        support_args: list[str] = []
        objection_defeater_args: list[str] = []
        for reason in probe.reason_evidence:
            reason_arg = evidence_argument_id("reason", probe.uci, reason)
            arguments.add(reason_arg)
            evidence[reason_arg] = reason
            if reason.supports_argument:
                support_args.extend(
                    add_typed_attack(
                        arguments,
                        defeats,
                        evidence,
                        attacker=reason_arg,
                        target=doubt_arg,
                        evidence=reason,
                        strength=reason.support_strength,
                    )
                )
            if reason.label == "terminal:checkmate":
                for other in probes:
                    if other.uci != probe.uci:
                        defeats.add((reason_arg, move_args[other.uci]))
        for objection in probe.objection_evidence:
            objection_arg = evidence_argument_id("objection", probe.uci, objection)
            objection_args = list(
                add_typed_attack(
                    arguments,
                    defeats,
                    evidence,
                    attacker=objection_arg,
                    target=move_arg,
                    evidence=objection,
                    strength=objection.objection_strength,
                )
            )
            if not objection_args:
                arguments.add(objection_arg)
                evidence[objection_arg] = objection
            for defeater in objection_defeater_evidence(probe, objection):
                defeater_arg = evidence_argument_id("defeater", probe.uci, defeater)
                for target_arg in objection_args:
                    objection_defeater_args.extend(
                        add_typed_attack(
                            arguments,
                            defeats,
                            evidence,
                            attacker=defeater_arg,
                            target=target_arg,
                            evidence=defeater,
                            strength=defeater.defeater_strength,
                        )
                    )
        for reply_evidence in probe.reply_attack_evidence:
            reply_arg = evidence_argument_id("reply_attack", probe.uci, reply_evidence)
            reply_attackers = add_typed_attack(
                arguments,
                defeats,
                evidence,
                attacker=reply_arg,
                target=move_arg,
                evidence=reply_evidence,
                strength=reply_evidence.reply_attack_strength,
            )
            if is_undefended_reply_capture(reply_evidence.label):
                for support_arg in support_args:
                    defeats.add((reply_arg, support_arg))
                for defeater_arg in objection_defeater_args:
                    defeats.add((reply_arg, defeater_arg))
            if is_defensible_reply_attack(reply_evidence.label):
                defense_arg = evidence_argument_id("defense", probe.uci, reply_evidence)
                for target in reply_attackers:
                    add_typed_attack(
                        arguments,
                        defeats,
                        evidence,
                        attacker=defense_arg,
                        target=target,
                        evidence=reply_evidence,
                        strength=reply_evidence.defense_strength,
                    )

    frozen_arguments = frozenset(arguments)
    frozen_defeats = frozenset(defeats)
    grounded_extension = local_grounded_extension(frozen_arguments, frozen_defeats)
    ranking = local_argumentation_ranking(frozen_arguments, frozen_defeats, evidence)
    return RootArgumentGraph(
        arguments=frozen_arguments,
        defeats=frozen_defeats,
        move_arguments=move_args,
        grounded_extension=grounded_extension,
        ranking=ranking,
        evidence=evidence,
    )


def build_argument_payload(
    probes: list[MoveProbe],
    graph: RootArgumentGraph | None = None,
) -> dict[str, Any]:
    graph = graph or build_root_argument_graph(probes)
    return {
        "arguments": sorted(graph.arguments),
        "defeats": sorted([list(pair) for pair in graph.defeats]),
        "move_scores": [probe_payload(probe) for probe in probes],
        "move_arguments": dict(sorted(graph.move_arguments.items())),
        "grounded_extension": sorted(graph.grounded_extension),
        "argumentation_ranking": graph.ranking,
    }


def probe_payload(probe: MoveProbe) -> dict[str, Any]:
    payload = asdict(probe)
    payload.pop("reason_evidence", None)
    payload.pop("objection_evidence", None)
    payload.pop("reply_attack_evidence", None)
    return payload


ARGUMENT_VALUES = frozenset(
    {
        "terminal",
        "reply_refutation",
        "material_safety",
        "search",
        "tactical",
        "positional",
        "procedural",
    }
)
AUDIENCE = (
    "terminal",
    "reply_refutation",
    "material_safety",
    "search",
    "tactical",
    "positional",
    "procedural",
)


def local_argumentation_ranking(
    arguments: frozenset[str],
    defeats: frozenset[tuple[str, str]],
    evidence: dict[str, ArgumentEvidence] | None = None,
) -> dict[str, Any]:
    framework = value_induced_framework(arguments, defeats, evidence or {})
    result = categoriser_scores(framework)
    return {
        "scores": dict(sorted(result.scores.items())),
        "ranking": [sorted(tier) for tier in result.ranking],
        "converged": result.converged,
        "iterations": result.iterations,
        "semantics": result.semantics,
        "attack_semantics": "value_based_defeater_undercut",
    }


def value_induced_framework(
    arguments: frozenset[str],
    defeats: frozenset[tuple[str, str]],
    evidence: dict[str, ArgumentEvidence],
) -> ArgumentationFramework:
    framework = ValueBasedArgumentationFramework(
        arguments=arguments,
        attacks=defeats,
        values=ARGUMENT_VALUES,
        valuation={argument: argument_value(argument, evidence) for argument in arguments},
        audience=AUDIENCE,
    )
    return ArgumentationFramework(
        arguments=arguments,
        defeats=preference_undercut_attacks(defeats, framework, evidence),
    )


def preference_undercut_attacks(
    defeats: frozenset[tuple[str, str]],
    framework: ValueBasedArgumentationFramework,
    evidence: dict[str, ArgumentEvidence],
) -> frozenset[tuple[str, str]]:
    attackers_by_target: dict[str, list[str]] = {}
    for attacker, target in defeats:
        attackers_by_target.setdefault(target, []).append(attacker)
    active: set[tuple[str, str]] = set()
    for attacker, target in defeats:
        if not is_defeater_argument(attacker, evidence):
            active.add((attacker, target))
            continue
        attacker_value = framework.valuation[attacker]
        undercut = any(
            framework.value_preferred(framework.valuation[undercutter], attacker_value)
            for undercutter in attackers_by_target.get(attacker, [])
        )
        if not undercut:
            active.add((attacker, target))
    return frozenset(active)


def is_defeater_argument(argument: str, evidence: dict[str, ArgumentEvidence]) -> bool:
    argument_evidence = evidence.get(argument)
    return argument_evidence is not None and argument_evidence.defeater_kind is not None


def argument_value(argument: str, evidence: dict[str, ArgumentEvidence]) -> str:
    if argument.startswith("move:") or argument.startswith("doubt:"):
        return "procedural"
    argument_evidence = evidence.get(argument)
    if argument_evidence is not None:
        return argument_evidence.argument_value
    return "procedural"


def local_grounded_extension(
    arguments: frozenset[str],
    defeats: frozenset[tuple[str, str]],
) -> frozenset[str]:
    framework = ArgumentationFramework(arguments=arguments, defeats=defeats)
    return grounded_extension(framework)
