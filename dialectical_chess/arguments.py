"""Argument graph construction and move selection for dialectical chess."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from argumentation.dung import ArgumentationFramework, grounded_extension
from argumentation.ranking import categoriser_scores

from dialectical_chess.evidence import (
    ArgumentEvidence,
    EvidenceWorld,
    search_refutation_score,
    to_argument_evidence,
)

SELECTOR_MODES = frozenset({"argument", "score", "grounded", "support", "categoriser", "optimizer"})
POSITIONAL_SCORE_BONUS = 25
LARGE_SEARCH_REFUTATION_THRESHOLD = -500
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
    *,
    selector_mode: str = "argument",
) -> MoveProbe:
    if not probes:
        raise SystemExit("position has no legal moves")
    if selector_mode not in SELECTOR_MODES:
        raise ValueError(f"unknown selector_mode: {selector_mode}")
    graph = graph or build_root_argument_graph(probes)
    if selector_mode == "score":
        return sorted(probes, key=score_selection_key)[0]
    if selector_mode == "grounded":
        return sorted(grounded_candidates(probes, graph), key=score_selection_key)[0]
    if selector_mode == "support":
        return sorted(probes, key=lambda probe: support_selection_key(probe, graph))[0]
    if selector_mode == "categoriser":
        return sorted(probes, key=lambda probe: categoriser_decision_key(probe, graph))[0]
    if selector_mode == "optimizer":
        from dialectical_chess.optimizer import choose_optimized_move

        return choose_optimized_move(probes, graph)
    return sorted(
        probes,
        key=lambda probe: categoriser_decision_key(probe, graph),
    )[0]


def grounded_candidates(probes: list[MoveProbe], graph: RootArgumentGraph) -> list[MoveProbe]:
    accepted = [
        probe
        for probe in probes
        if graph.move_arguments[probe.uci] in graph.grounded_extension
    ]
    return accepted if accepted else probes


def argument_selection_candidates(probes: list[MoveProbe], graph: RootArgumentGraph) -> list[MoveProbe]:
    candidates = grounded_candidates(probes, graph)
    if any(has_forced_mate_refutation(probe) for probe in candidates):
        return probes
    return candidates


def score_selection_key(probe: MoveProbe) -> tuple[int, str]:
    return (-probe.score, probe.uci)


def categoriser_decision_key(
    probe: MoveProbe,
    graph: RootArgumentGraph,
) -> tuple[Any, ...]:
    move_arg = graph.move_arguments[probe.uci]
    ranking_scores = graph.ranking["scores"]
    move_rank = float(ranking_scores.get(move_arg, 0.0))
    return (-move_rank, -probe.score, probe.uci)


def selection_key(
    probe: MoveProbe,
    graph: RootArgumentGraph,
) -> tuple[Any, ...]:
    move_arg = graph.move_arguments[probe.uci]
    ranking_scores = graph.ranking["scores"]
    move_rank = float(ranking_scores.get(move_arg, 0.0))
    mode = positional_support_mode(graph)
    accepted_tactical = accepted_tactical_support_count(probe, graph)
    accepted_positional = effective_positional_support_count(probe, graph, mode)
    accepted_defenses = sum(
        1
        for reply_attack in probe.reply_attacks
        if f"defense:{probe.uci}:{reply_attack}" in graph.grounded_extension
    )
    unresolved_attacks = sum(
        1
        for reply_attack in probe.reply_attacks
        if f"reply_attack:{probe.uci}:{reply_attack}" in graph.grounded_extension
    )
    if mode == "quiet":
        forced_mate_refuted = has_forced_mate_refutation(probe)
        return (
            severe_objection_count(probe),
            1 if forced_mate_refuted else 0,
            -effective_score(probe, mode) if forced_mate_refuted else 0,
            -move_rank,
            -accepted_tactical,
            unresolved_attacks,
            -accepted_positional,
            -accepted_defenses,
            -effective_score(probe, mode),
            probe.uci,
        )
    forced_mate_refuted = has_forced_mate_refutation(probe)
    return (
        severe_objection_count(probe),
        1 if forced_mate_refuted else 0,
        -effective_score(probe, mode) if forced_mate_refuted else 0,
        -accepted_tactical,
        unresolved_attacks,
        -effective_score(probe, mode),
        -material_or_promotion_gain(probe),
        -accepted_defenses,
        -move_rank,
        -accepted_positional,
        probe.uci,
    )


def support_selection_key(
    probe: MoveProbe,
    graph: RootArgumentGraph,
) -> tuple[Any, ...]:
    mode = positional_support_mode(graph)
    accepted_tactical = accepted_tactical_support_count(probe, graph)
    accepted_positional = effective_positional_support_count(probe, graph, mode)
    accepted_defenses = accepted_defense_count(probe, graph)
    unresolved_attacks = unresolved_attack_count(probe, graph)
    if mode == "quiet":
        return (
            severe_objection_count(probe),
            -accepted_tactical,
            unresolved_attacks,
            -accepted_positional,
            -accepted_defenses,
            -effective_score(probe, mode),
            probe.uci,
        )
    return (
        severe_objection_count(probe),
        -accepted_tactical,
        unresolved_attacks,
        -effective_score(probe, mode),
        -material_or_promotion_gain(probe),
        -accepted_defenses,
        -accepted_positional,
        probe.uci,
    )


def categoriser_selection_key(
    probe: MoveProbe,
    graph: RootArgumentGraph,
) -> tuple[Any, ...]:
    move_arg = graph.move_arguments[probe.uci]
    ranking_scores = graph.ranking["scores"]
    move_rank = float(ranking_scores.get(move_arg, 0.0))
    mode = positional_support_mode(graph)
    if mode == "quiet":
        return (
            severe_objection_count(probe),
            -move_rank,
            -accepted_tactical_support_count(probe, graph),
            unresolved_attack_count(probe, graph),
            -effective_positional_support_count(probe, graph, mode),
            -effective_score(probe, mode),
            probe.uci,
        )
    return (
        severe_objection_count(probe),
        -accepted_tactical_support_count(probe, graph),
        unresolved_attack_count(probe, graph),
        -effective_score(probe, mode),
        -material_or_promotion_gain(probe),
        -move_rank,
        -effective_positional_support_count(probe, graph, mode),
        probe.uci,
    )


def accepted_support_count(probe: MoveProbe, graph: RootArgumentGraph) -> int:
    return sum(
        1
        for reason in probe.reasons
        if f"reason:{probe.uci}:{reason}" in graph.grounded_extension
    )


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
    return sum(1 for reason in probe.reasons if to_argument_evidence(reason).counts_as_positional)


def soft_positional_reason_count(probe: MoveProbe) -> int:
    return sum(
        1
        for reason in probe.reasons
        if to_argument_evidence(reason).counts_as_positional
        and not is_concrete_non_queen_piece_safety(reason)
    )


def is_concrete_non_queen_piece_safety(reason: str) -> bool:
    prefix = "piece_safety:defended:"
    if not reason.startswith(prefix):
        return False
    parts = reason.split(":")
    if len(parts) != 4:
        return False
    try:
        moved_value = int(parts[3])
    except ValueError:
        return False
    return moved_value < 900


def material_or_promotion_gain(probe: MoveProbe) -> int:
    return probe.captured_value + probe.promotion_value


def severe_objection_count(probe: MoveProbe) -> int:
    return sum(severe_objection_weight(objection) for objection in probe.objections)


def has_forced_mate_refutation(probe: MoveProbe) -> bool:
    return any(
        is_forced_mate_refutation(objection)
        or objection.startswith("tactical:allows_reply_mate_in_one:")
        or objection.startswith("tactical:allows_reply_forced_mate_in_")
        for objection in probe.objections
    )


def severe_objection_weight(objection: str) -> int:
    if is_forced_mate_refutation(objection):
        return 6
    if is_large_search_refutation(objection):
        return 1
    if objection.startswith("smt:fork:high_value_piece:"):
        return 3
    if objection.startswith("tactical:allows_reply_mate_in_one:") or objection.startswith(
        "tactical:allows_reply_forced_mate_in_2:"
    ):
        return 6
    if objection.startswith("tactical:allows_reply_forced_mate_in_"):
        return 3
    if objection.startswith("safety:queen_blunder:"):
        return 2
    if objection.startswith("safety:ignored_hanging_piece:"):
        return 1
    if is_moved_minor_or_major_en_pris(objection):
        return 1
    if objection.startswith("king_safety:queen_flank_invasion:"):
        return 2
    if objection.startswith("king_safety:unanswered_advanced_flank_pawn:"):
        return 4
    if objection.startswith("strategy:unsupported_major_drift:"):
        return 1
    if (
        objection.startswith("opening:king_walk:")
        or objection.startswith("opening:king_center_flight:")
        or objection.startswith("opening:premature_queen:")
        or objection.startswith("opening:premature_rook:")
        or objection.startswith("opening:minor_retreat:")
        or objection.startswith("king_safety:flank_pawn_weakening:")
        or objection.startswith("king_safety:castled_flank_pawn_weakening:")
        or objection.startswith("king_safety:flank_pawn_lunge:")
    ):
        return 1
    if objection.startswith("opening:premature_minor_check:"):
        return 1
    return 0


def is_forced_mate_refutation(objection: str) -> bool:
    score = search_refutation_score(objection)
    return score is not None and score <= -100_000


def is_large_search_refutation(objection: str) -> bool:
    score = search_refutation_score(objection)
    return score is not None and score <= LARGE_SEARCH_REFUTATION_THRESHOLD


def is_moved_minor_or_major_en_pris(objection: str) -> bool:
    prefix = "safety:moved_piece_en_pris:"
    if not objection.startswith(prefix):
        return False
    try:
        value = int(objection.removeprefix(prefix))
    except ValueError:
        return False
    return value >= 300


def has_compensating_tactical_pressure(probe: MoveProbe) -> bool:
    return any(
        to_argument_evidence(reason).tactical_threat_value >= COMPENSATING_TACTICAL_THREAT_THRESHOLD
        for reason in probe.reasons
    )


def has_compensating_forcing_pressure(probe: MoveProbe) -> bool:
    return has_compensating_tactical_pressure(probe) and (
        probe.gives_check or material_or_promotion_gain(probe) > 0
    )


def has_forcing_material_gain(probe: MoveProbe) -> bool:
    return probe.gives_check and material_or_promotion_gain(probe) > 0


def has_search_support(probe: MoveProbe) -> bool:
    return any(reason.startswith("search_support:") for reason in probe.reasons)


def has_advanced_flank_pawn_response(probe: MoveProbe) -> bool:
    return any(
        reason.startswith("king_safety:advanced_flank_pawn_response:")
        for reason in probe.reasons
    )


def objection_defeaters(probe: MoveProbe, objection: str) -> tuple[str, ...]:
    defeaters = []
    if objection.startswith("safety:queen_blunder:") and has_compensating_forcing_pressure(probe):
        defeaters.append("compensating_forcing_pressure")
    if is_moved_minor_or_major_en_pris(objection):
        if has_compensating_tactical_pressure(probe):
            defeaters.append("compensating_tactical_pressure")
        if has_forcing_material_gain(probe):
            defeaters.append("forcing_material_gain")
    if objection.startswith("opening:premature_minor_check:") and has_search_support(probe):
        defeaters.append("search_support")
    if objection.startswith("king_safety:flank_pawn_weakening:") and has_advanced_flank_pawn_response(probe):
        defeaters.append("advanced_flank_pawn_response")
    if objection.startswith("king_safety:flank_pawn_lunge:") and has_advanced_flank_pawn_response(probe):
        defeaters.append("advanced_flank_pawn_response")
    return tuple(defeaters)


def _accepted_reason_count(
    probe: MoveProbe,
    graph: RootArgumentGraph,
    predicate,
) -> int:
    return sum(
        1
        for reason in probe.reasons
        if predicate(graph.evidence[f"reason:{probe.uci}:{reason}"])
        and f"reason:{probe.uci}:{reason}" in graph.grounded_extension
    )


def extra_support_copies(evidence: ArgumentEvidence) -> int:
    if evidence.label.startswith("material:promotion:"):
        return 16
    if evidence.label.startswith("material:capture:"):
        return material_support_copies(evidence.label)
    if evidence.label.startswith("king_safety:advanced_flank_pawn_response:"):
        return 12
    if evidence.label.startswith("piece_safety:defended:"):
        return defended_piece_support_copies(evidence.label)
    if evidence.label == "tactical:check":
        return 6
    if evidence.tactical_threat_value >= COMPENSATING_TACTICAL_THREAT_THRESHOLD:
        return 5
    if evidence.world in {EvidenceWorld.TERMINAL, EvidenceWorld.PROCEDURAL}:
        return 8
    if evidence.world in {EvidenceWorld.SMT, EvidenceWorld.SEARCH}:
        return 3
    if evidence.world == EvidenceWorld.TACTICAL:
        return 2
    return 0


def material_support_copies(label: str) -> int:
    parts = label.split(":")
    if len(parts) != 3:
        return 3
    try:
        value = int(parts[2])
    except ValueError:
        return 3
    if value >= 500:
        return 8
    if value >= 300:
        return 5
    if value > 0:
        return 2
    return 0


def defended_piece_support_copies(label: str) -> int:
    parts = label.split(":")
    if len(parts) != 4:
        return 0
    try:
        value = int(parts[3])
    except ValueError:
        return 0
    if value >= 900:
        return 3
    if value >= 500:
        return 2
    return 0


def extra_defeater_copies(defeater: str) -> int:
    if defeater == "search_support":
        return 96
    if defeater == "advanced_flank_pawn_response":
        return 32
    if defeater in {"compensating_forcing_pressure", "forcing_material_gain"}:
        return 32
    if defeater == "compensating_tactical_pressure":
        return 16
    return 1


def extra_objection_copies(objection: str) -> int:
    return max(severe_objection_weight(objection) - 1, 0)


def extra_defense_copies(reply_attack: str) -> int:
    if ":defended:" in reply_attack:
        return 12
    return 0


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
        for reason in probe.reasons:
            reason_arg = f"reason:{probe.uci}:{reason}"
            arguments.add(reason_arg)
            reason_evidence = to_argument_evidence(reason)
            evidence[reason_arg] = reason_evidence
            if reason_evidence.supports_argument:
                defeats.add((reason_arg, doubt_arg))
                for index in range(extra_support_copies(reason_evidence)):
                    support_arg = f"support:{probe.uci}:{reason}:{index}"
                    arguments.add(support_arg)
                    defeats.add((support_arg, doubt_arg))
            if reason == "terminal:checkmate":
                for other in probes:
                    if other.uci != probe.uci:
                        defeats.add((reason_arg, move_args[other.uci]))
        for objection in probe.objections:
            objection_arg = f"{objection}:{probe.uci}"
            objection_args = [objection_arg]
            arguments.add(objection_arg)
            if severe_objection_weight(objection) > 0:
                defeats.add((objection_arg, move_arg))
                for index in range(extra_objection_copies(objection)):
                    weighted_objection_arg = f"objection:{probe.uci}:{objection}:{index}"
                    objection_args.append(weighted_objection_arg)
                    arguments.add(weighted_objection_arg)
                    defeats.add((weighted_objection_arg, move_arg))
            for defeater in objection_defeaters(probe, objection):
                defeater_arg = f"defeater:{probe.uci}:{defeater}"
                arguments.add(defeater_arg)
                for target_arg in objection_args:
                    defeats.add((defeater_arg, target_arg))
                for index in range(extra_defeater_copies(defeater)):
                    support_arg = f"defeater:{probe.uci}:{defeater}:{index}"
                    arguments.add(support_arg)
                    for target_arg in objection_args:
                        defeats.add((support_arg, target_arg))
        for reply_attack in probe.reply_attacks:
            reply_arg = f"reply_attack:{probe.uci}:{reply_attack}"
            arguments.add(reply_arg)
            defeats.add((reply_arg, move_arg))
            if ":defended:" in reply_attack:
                defense_arg = f"defense:{probe.uci}:{reply_attack}"
                arguments.add(defense_arg)
                defeats.add((defense_arg, reply_arg))
                for index in range(extra_defense_copies(reply_attack)):
                    defense_support_arg = f"defense:{probe.uci}:{reply_attack}:{index}"
                    arguments.add(defense_support_arg)
                    defeats.add((defense_support_arg, reply_arg))

    frozen_arguments = frozenset(arguments)
    frozen_defeats = frozenset(defeats)
    grounded_extension = local_grounded_extension(frozen_arguments, frozen_defeats)
    ranking = local_argumentation_ranking(frozen_arguments, frozen_defeats)
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
        "move_scores": [asdict(probe) for probe in probes],
        "move_arguments": dict(sorted(graph.move_arguments.items())),
        "grounded_extension": sorted(graph.grounded_extension),
        "argumentation_ranking": graph.ranking,
    }


def local_argumentation_ranking(
    arguments: frozenset[str],
    defeats: frozenset[tuple[str, str]],
) -> dict[str, Any]:
    framework = ArgumentationFramework(arguments=arguments, defeats=defeats)
    result = categoriser_scores(framework)
    return {
        "scores": dict(sorted(result.scores.items())),
        "ranking": [sorted(tier) for tier in result.ranking],
        "converged": result.converged,
        "iterations": result.iterations,
        "semantics": result.semantics,
    }


def local_grounded_extension(
    arguments: frozenset[str],
    defeats: frozenset[tuple[str, str]],
) -> frozenset[str]:
    framework = ArgumentationFramework(arguments=arguments, defeats=defeats)
    return grounded_extension(framework)
