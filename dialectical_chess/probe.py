"""Move probing for dialectical chess."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import chess

from dialectical_chess.arguments import MoveProbe
from dialectical_chess.board import OwnedBoard, file_of, opposite, piece_color, rank_of, square_index, square_name
from dialectical_chess.evidence import (
    ArgumentEvidence,
    DefeaterKind,
    EvidenceWorld,
    ObjectionKind,
    SupportKind,
    base_objection_strength,
    defeater_evidence,
    defeater_strength,
    material_cost_objection_strength,
    objection_evidence,
    reply_evidence,
    support_evidence,
)
from dialectical_chess.loss_mining import has_forced_mate
from dialectical_chess.search import (
    OWNED_PIECE_VALUE,
    ReplyAnalysisCache,
    ReplyAnalysisSettings,
    SearchSettings,
    bounded_reply_attack_evidence,
    bounded_reply_attacks,
    owned_capture_value,
    owned_is_capture,
    owned_is_checkmate,
    owned_is_draw,
    owned_is_stalemate,
    owned_is_threefold_repetition,
    append_position_history,
    root_search_result,
)
from dialectical_chess.smt import (
    SmtSettings,
    moved_piece_attacks_square,
    smt_fork_witnesses,
    smt_mate_in_one_moves,
)


@dataclass(frozen=True)
class ProbeSettings:
    dialectic_depth: int = 1
    search: SearchSettings = field(default_factory=SearchSettings)
    reply_analysis: ReplyAnalysisSettings = field(default_factory=ReplyAnalysisSettings)
    smt: SmtSettings = field(default_factory=SmtSettings)
    positional_reasons: bool = True
    reply_mate_scan: bool = True
    position_history: tuple[str, ...] = ()
    deadline: float | None = None


@dataclass(frozen=True)
class EvidenceLabels:
    labels: tuple[str, ...]
    evidence: tuple[ArgumentEvidence, ...] = ()
    score: int = 0


def support(
    label: str,
    *,
    world: EvidenceWorld,
    strength: int,
    counts_as_positional: bool = False,
    counts_as_tactical: bool = False,
    argument_value: str = "procedural",
    tactical_threat_value: int = 0,
    defended_piece_value: int | None = None,
    search_support_score: int | None = None,
    support_kind: SupportKind = SupportKind.GENERIC,
) -> ArgumentEvidence:
    return support_evidence(
        label,
        world=world,
        counts_as_positional=counts_as_positional,
        counts_as_tactical=counts_as_tactical,
        argument_value=argument_value,
        support_strength=strength,
        tactical_threat_value=tactical_threat_value,
        defended_piece_value=defended_piece_value,
        search_support_score=search_support_score,
        support_kind=support_kind,
    )


def display_evidence(label: str, *, world: EvidenceWorld = EvidenceWorld.PROCEDURAL) -> ArgumentEvidence:
    return support_evidence(label, world=world)


def objection(
    label: str,
    *,
    kind: ObjectionKind,
    strength: int | None = None,
    world: EvidenceWorld = EvidenceWorld.UNKNOWN,
    moved_piece_en_pris_value: int | None = None,
    search_refutation_score: int | None = None,
    forced_mate_distance: int | None = None,
    argument_value: str = "procedural",
) -> ArgumentEvidence:
    return objection_evidence(
        label,
        world=world,
        objection_kind=kind,
        objection_strength=base_objection_strength(kind) if strength is None else strength,
        moved_piece_en_pris_value=moved_piece_en_pris_value,
        search_refutation_score=search_refutation_score,
        forced_mate_distance=forced_mate_distance,
        argument_value=argument_value,
    )


def search_refutation_strength(score: int) -> int:
    if score <= -100_000:
        return 6
    if score <= -500:
        return 1
    return 0


def material_support_strength(value: int) -> int:
    if value >= 500:
        return 9
    if value >= 300:
        return 6
    if value > 0:
        return 3
    return 1


def defended_piece_support_strength(value: int) -> int:
    if value >= 900:
        return 4
    if value >= 500:
        return 3
    return 1


def probe_moves(
    board: Any,
    *,
    dialectic_depth: int = 1,
    search_depth: int = 0,
    search_backend: str = "negamax",
    smt_mate: bool = True,
    smt_fork: bool = True,
    positional_reasons: bool = True,
    reply_mate_scan: bool = True,
    reply_analysis: ReplyAnalysisSettings | None = None,
    position_history: tuple[str, ...] = (),
    deadline: float | None = None,
) -> list[MoveProbe]:
    settings = ProbeSettings(
        dialectic_depth=dialectic_depth,
        search=SearchSettings(depth=search_depth, backend=search_backend),
        reply_analysis=reply_analysis or ReplyAnalysisSettings(),
        smt=SmtSettings(mate_in_one=smt_mate, fork=smt_fork),
        positional_reasons=positional_reasons,
        reply_mate_scan=reply_mate_scan,
        position_history=position_history,
        deadline=deadline,
    )
    return probe_moves_with_settings(board, settings)


def probe_moves_with_settings(board: Any, settings: ProbeSettings) -> list[MoveProbe]:
    if settings.dialectic_depth < 0:
        raise ValueError("dialectic_depth must be non-negative")
    if settings.search.depth < 0:
        raise ValueError("search_depth must be non-negative")
    board = ensure_owned_board(board)
    legal_moves = sorted(board.legal_moves(), key=lambda move: move.uci())
    smt_mate_moves = (
        smt_mate_in_one_moves(board) if settings.smt.mate_in_one else frozenset()
    )
    smt_fork_witness_map = smt_fork_witnesses(board) if settings.smt.fork else {}
    smt_fork_move_set = frozenset(smt_fork_witness_map)
    reply_cache = ReplyAnalysisCache()
    probes = []
    for move in legal_moves:
        if probes and settings.deadline is not None and time.monotonic() >= settings.deadline:
            break
        san = move.uci()
        is_capture = owned_is_capture(board, move)
        captured_value = owned_capture_value(board, move)
        promotion_value = OWNED_PIECE_VALUE.get(move.promotion or "", 0)
        child = board.apply(move)
        child_position_history = append_position_history(settings.position_history, child)
        is_checkmate = owned_is_checkmate(child)
        is_stalemate = owned_is_stalemate(child)
        is_draw = (
            False
            if is_checkmate or is_stalemate
            else owned_is_draw(child, position_history=child_position_history)
        )
        gives_check = child.in_check(child.turn)

        reasons: list[str] = []
        objections: list[str] = []
        reason_evidence: list[ArgumentEvidence] = []
        objection_evidence_items: list[ArgumentEvidence] = []
        score = 0

        if is_checkmate:
            score += 1_000_000
            label = "terminal:checkmate"
            reasons.append(label)
            reason_evidence.append(
                support(
                    label,
                    world=EvidenceWorld.TERMINAL,
                    counts_as_tactical=True,
                    argument_value="terminal",
                    strength=9,
                )
            )
        elif is_draw:
            draw = draw_objections(move, child=child, position_history=child_position_history)
            objections.extend(draw.labels)
            objection_evidence_items.extend(draw.evidence)
        if not is_draw and gives_check:
            score += 1_000
            label = "tactical:check"
            reasons.append(label)
            reason_evidence.append(
                support(
                    label,
                    world=EvidenceWorld.TACTICAL,
                    counts_as_tactical=True,
                    argument_value="tactical",
                    strength=7,
                )
            )
        if not is_draw and is_capture:
            score += captured_value
            label = f"material:capture:{captured_value}"
            reasons.append(label)
            reason_evidence.append(
                support(
                    label,
                    world=EvidenceWorld.MATERIAL,
                    counts_as_tactical=True,
                    argument_value="tactical",
                    strength=material_support_strength(captured_value),
                )
            )
        if not is_draw and promotion_value:
            score += promotion_value
            label = f"material:promotion:{promotion_value}"
            reasons.append(label)
            reason_evidence.append(
                support(
                    label,
                    world=EvidenceWorld.MATERIAL,
                    counts_as_tactical=True,
                    argument_value="tactical",
                    strength=17,
                )
            )
        if not is_draw and not is_checkmate:
            safety_reasons, safety_reason_evidence, safety_objections, safety_objection_evidence, safety_score = moved_piece_safety_labels(
                board,
                move,
                child,
                captured_value=captured_value,
                gives_check=gives_check,
                promotion_value=promotion_value,
            )
            reasons.extend(safety_reasons)
            reason_evidence.extend(safety_reason_evidence)
            objections.extend(safety_objections)
            objection_evidence_items.extend(safety_objection_evidence)
            score += safety_score
            threat = moved_piece_threat_labels(board, move, child)
            reasons.extend(threat.labels)
            reason_evidence.extend(threat.evidence)
            score += threat.score
            opening = opening_development_objections(
                board,
                move,
                captured_value=captured_value,
                gives_check=gives_check,
            )
            objections.extend(opening.labels)
            objection_evidence_items.extend(opening.evidence)
            score += opening.score
            minor_retreat = opening_minor_retreat_objections(
                board,
                move,
                captured_value=captured_value,
                gives_check=gives_check,
            )
            objections.extend(minor_retreat.labels)
            objection_evidence_items.extend(minor_retreat.evidence)
            score += minor_retreat.score
            king = opening_king_safety_objections(
                board,
                move,
                captured_value=captured_value,
            )
            objections.extend(king.labels)
            objection_evidence_items.extend(king.evidence)
            score += king.score
            flank_pawn = flank_pawn_weakening_objections(board, move)
            objections.extend(flank_pawn.labels)
            objection_evidence_items.extend(flank_pawn.evidence)
            score += flank_pawn.score
            flank_response_reasons, flank_response_evidence, flank_unanswered_objections, flank_unanswered_evidence, flank_response_score = advanced_flank_pawn_response_labels(
                board,
                move,
                child,
            )
            reasons.extend(flank_response_reasons)
            reason_evidence.extend(flank_response_evidence)
            objections.extend(flank_unanswered_objections)
            objection_evidence_items.extend(flank_unanswered_evidence)
            score += flank_response_score
            ignored = ignored_hanging_piece_objections(board, move, child)
            objections.extend(ignored.labels)
            objection_evidence_items.extend(ignored.evidence)
            score += ignored.score
            flank = queen_flank_invasion_objections(board, move, child)
            objections.extend(flank.labels)
            objection_evidence_items.extend(flank.evidence)
            score += flank.score
            if settings.positional_reasons:
                escape = king_escape_square_reasons(board, move, child)
                reasons.extend(escape.labels)
                reason_evidence.extend(escape.evidence)
                score += escape.score
            if settings.reply_mate_scan and should_scan_reply_mate(
                settings.search.depth,
                board,
                move,
                captured_value=captured_value,
                reason_evidence=reason_evidence,
                objection_evidence=objection_evidence_items,
            ):
                reply_mate = reply_mate_in_one_objections(child, move)
                objections.extend(reply_mate.labels)
                objection_evidence_items.extend(reply_mate.evidence)
                score += reply_mate.score
        if not is_draw and settings.positional_reasons:
            positional = positional_reason_labels(board, move, child)
            if positional.labels:
                score += 25 * len(positional.labels)
                reasons.extend(positional.labels)
                reason_evidence.extend(positional.evidence)
        smt_witnesses: list[str] = []
        if not is_draw and move.uci() in smt_mate_moves:
            score += 1_000_000
            label = "procedural:mate_in_one"
            reasons.append(label)
            reason_evidence.append(
                support(
                    label,
                    world=EvidenceWorld.PROCEDURAL,
                    counts_as_tactical=True,
                    strength=9,
                )
            )
            smt_witnesses.append("procedural_mate_in_one")
        if not is_draw and move.uci() in smt_fork_move_set:
            fork_reasons, fork_reason_evidence, fork_objections, fork_objection_evidence, fork_score = fork_witness_labels(smt_fork_witness_map[move.uci()], gives_check)
            score += fork_score
            reasons.extend(fork_reasons)
            reason_evidence.extend(fork_reason_evidence)
            objections.extend(fork_objections)
            objection_evidence_items.extend(fork_objection_evidence)
            smt_witnesses.append("fork")
        search_result = None if is_draw else root_search_result(
            board,
            move,
            settings=settings.search,
            position_history=settings.position_history,
        )
        if search_result is not None:
            search_line_label = "search_line:" + "-".join(search_result.line)
            if search_result.score > 0:
                search_label = f"search:{settings.search.backend}:{search_result.score}"
                support_label = f"search_support:{settings.search.backend}:{search_result.score}"
                reasons.append(search_label)
                reasons.append(support_label)
                reasons.append(search_line_label)
                reason_evidence.append(
                    support(
                        search_label,
                        world=EvidenceWorld.SEARCH,
                        counts_as_tactical=True,
                        argument_value="search",
                        strength=4,
                    )
                )
                reason_evidence.append(
                    defeater_evidence(
                        support_label,
                        world=EvidenceWorld.SEARCH,
                        defeater_kind=DefeaterKind.SEARCH_SUPPORT,
                        defeater_strength=defeater_strength(DefeaterKind.SEARCH_SUPPORT),
                        counts_as_tactical=True,
                        argument_value="search",
                        support_strength=4,
                        search_support_score=search_result.score,
                    )
                )
                reason_evidence.append(display_evidence(search_line_label, world=EvidenceWorld.SEARCH))
            elif search_result.score < 0:
                search_label = f"search:{settings.search.backend}:{search_result.score}"
                refutes_label = f"search_refutes:{settings.search.backend}:{search_result.score}"
                objections.append(search_label)
                objections.append(refutes_label)
                objections.append(search_line_label)
                objection_evidence_items.append(display_evidence(search_label, world=EvidenceWorld.SEARCH))
                objection_evidence_items.append(
                    objection(
                        refutes_label,
                        kind=ObjectionKind.SEARCH_REFUTATION,
                        strength=search_refutation_strength(search_result.score),
                        world=EvidenceWorld.SEARCH,
                        search_refutation_score=search_result.score,
                        argument_value="search",
                    )
                )
                objection_evidence_items.append(display_evidence(search_line_label, world=EvidenceWorld.SEARCH))
            score += search_result.score
            if (
                settings.search.depth == 1
                and search_result.score <= -700
                and settings.reply_mate_scan
                and not has_reply_mate_in_one_objection(objection_evidence_items)
            ):
                reply_mate = reply_mate_in_one_objections(child, move)
                objections.extend(reply_mate.labels)
                objection_evidence_items.extend(reply_mate.evidence)
                score += reply_mate.score
        if not is_draw:
            drift = unsupported_major_drift_objections(
                board,
                move,
                captured_value=captured_value,
                gives_check=gives_check,
                reason_evidence=reason_evidence,
            )
            objections.extend(drift.labels)
            objection_evidence_items.extend(drift.evidence)
            score += drift.score
        reply_attack_evidence = () if is_draw else bounded_reply_attack_evidence(
            board,
            move,
            reply_depth=settings.dialectic_depth,
            settings=settings.reply_analysis,
            cache=reply_cache,
        )
        reply_attacks = tuple(evidence.label for evidence in reply_attack_evidence)
        if not reasons:
            label = "objection:no_immediate_tactical_warrant"
            objections.append(label)
            objection_evidence_items.append(
                objection(
                    label,
                    kind=ObjectionKind.NO_IMMEDIATE_TACTICAL_WARRANT,
                    strength=0,
                    world=EvidenceWorld.PROCEDURAL,
                )
            )

        probes.append(
            MoveProbe(
                uci=move.uci(),
                san=san,
                score=score,
                is_checkmate=is_checkmate,
                gives_check=gives_check,
                is_capture=is_capture,
                captured_value=captured_value,
                promotion_value=promotion_value,
                reasons=tuple(reasons),
                objections=tuple(objections),
                reply_attacks=reply_attacks,
                search_score=None if search_result is None else search_result.score,
                search_line=() if search_result is None else search_result.line,
                smt_witnesses=tuple(smt_witnesses),
                post_fen=child.fen(),
                reason_evidence=tuple(reason_evidence),
                objection_evidence=tuple(objection_evidence_items),
                reply_attack_evidence=reply_attack_evidence,
            )
        )
    if settings.reply_mate_scan:
        probes = scan_forced_reply_mates_for_candidate_moves(
            board,
            legal_moves,
            probes,
            dialectic_depth=settings.dialectic_depth,
            search_depth=settings.search.depth,
            deadline=settings.deadline,
        )
    return sorted(probes, key=lambda probe: (-probe.score, probe.uci))


def has_reply_mate_in_one_objection(objections: list[ArgumentEvidence]) -> bool:
    return any(
        objection.objection_kind == ObjectionKind.REPLY_MATE_IN_ONE
        for objection in objections
    )


def unsupported_major_drift_objections(
    board: OwnedBoard,
    move: Any,
    *,
    captured_value: int,
    gives_check: bool,
    reason_evidence: list[ArgumentEvidence],
) -> EvidenceLabels:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.lower() != "q":
        return EvidenceLabels(())
    if board.fullmove_number > 35 or captured_value > 0 or gives_check:
        return EvidenceLabels(())
    if any(reason.supports_argument and reason.counts_as_tactical for reason in reason_evidence):
        return EvidenceLabels(())
    label = f"strategy:unsupported_major_drift:{move.uci()}"
    return EvidenceLabels(
        (label,),
        (
            objection(
                label,
                kind=ObjectionKind.UNSUPPORTED_MAJOR_DRIFT,
                strength=1,
            ),
        ),
        -300,
    )


def draw_objections(
    move: Any,
    *,
    child: OwnedBoard,
    position_history: tuple[str, ...],
) -> EvidenceLabels:
    labels: list[str] = []
    evidence: list[ArgumentEvidence] = []
    move_text = move.uci()
    if owned_is_threefold_repetition(child, position_history=position_history):
        label = f"strategy:threefold_repetition:{move_text}"
        labels.append(label)
        evidence.append(objection(label, kind=ObjectionKind.THREEFOLD_REPETITION))
    if child.is_fifty_move_draw():
        label = f"strategy:fifty_move_draw:{move_text}"
        labels.append(label)
        evidence.append(objection(label, kind=ObjectionKind.FIFTY_MOVE_DRAW))
    return EvidenceLabels(tuple(labels), tuple(evidence))


def king_escape_square_reasons(
    board: OwnedBoard,
    move: Any,
    child: OwnedBoard,
) -> EvidenceLabels:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.lower() != "p":
        return EvidenceLabels(())
    color = piece_color(piece)
    king_square = board.king_square(color)
    if not king_adjacent(king_square, move.from_square):
        return EvidenceLabels(())
    if child.piece_at(move.from_square) is not None:
        return EvidenceLabels(())
    if child.is_square_attacked(move.from_square, opposite(color)):
        return EvidenceLabels(())
    label = f"king_safety:escape_square:{move.uci()}:{square_name(move.from_square)}"
    return EvidenceLabels(
        (label,),
        (
            support(
                label,
                world=EvidenceWorld.POSITIONAL,
                counts_as_positional=True,
                argument_value="positional",
                strength=1,
            ),
        ),
        300,
    )


def king_adjacent(left: int, right: int) -> bool:
    return max(
        abs(file_of(left) - file_of(right)),
        abs(rank_of(left) - rank_of(right)),
    ) == 1


def fork_witness_labels(
    witness: Any,
    gives_check: bool,
) -> tuple[tuple[str, ...], tuple[ArgumentEvidence, ...], tuple[str, ...], tuple[ArgumentEvidence, ...], int]:
    labels = [
        f"smt:fork:targets:{witness.target_count}:value:{witness.target_value}",
        f"smt:fork:piece:{witness.piece}",
        f"smt:fork:net:{witness.net_value}",
    ]
    if gives_check:
        labels.append("smt:fork:gives_check")
    reason_evidence = [display_evidence(label, world=EvidenceWorld.SMT) for label in labels]
    if witness.piece in {"q", "r"} and not gives_check:
        objection_label = f"smt:fork:high_value_piece:{witness.piece}"
        return (
            tuple(labels),
            tuple(reason_evidence),
            (objection_label,),
            (
                objection(
                    objection_label,
                    kind=ObjectionKind.SMT_FORK_HIGH_VALUE,
                    strength=3,
                    world=EvidenceWorld.SMT,
                    argument_value="tactical",
                ),
            ),
            0,
        )
    if witness.moved_piece_en_pris_value:
        objection_label = f"smt:fork:moved_piece_en_pris:{witness.moved_piece_en_pris_value}"
        objections = (objection_label,)
        objection_evidence = (display_evidence(objection_label, world=EvidenceWorld.SMT),)
        if witness.net_value <= 0 and not gives_check:
            return tuple(labels), tuple(reason_evidence), objections, objection_evidence, 0
    compatibility = f"smt:fork:{witness.target_count}:{witness.target_value}"
    return (
        (compatibility, *labels),
        (
            support(
                compatibility,
                world=EvidenceWorld.SMT,
                counts_as_tactical=True,
                argument_value="tactical",
                strength=4,
            ),
            *reason_evidence,
        ),
        (),
        (),
        max(0, witness.net_value),
    )


def moved_piece_safety_labels(
    board: OwnedBoard,
    move: Any,
    child: OwnedBoard,
    *,
    captured_value: int,
    gives_check: bool,
    promotion_value: int,
) -> tuple[tuple[str, ...], tuple[ArgumentEvidence, ...], tuple[str, ...], tuple[ArgumentEvidence, ...], int]:
    moved_piece = board.piece_at(move.from_square)
    if moved_piece is None:
        return (), (), (), (), 0
    moved_value = OWNED_PIECE_VALUE.get((move.promotion or moved_piece).lower(), 0)
    if moved_value <= 0:
        return (), (), (), (), 0
    defended = child.is_square_attacked(move.to_square, opposite(child.turn))
    en_pris = child.is_square_attacked(move.to_square, child.turn)
    reasons: list[str] = []
    objections: list[str] = []
    reason_evidence: list[ArgumentEvidence] = []
    objection_evidence_items: list[ArgumentEvidence] = []
    score = 0
    if defended:
        label = f"piece_safety:defended:{move.uci()}:{moved_value}"
        reasons.append(label)
        reason_evidence.append(
            support(
                label,
                world=EvidenceWorld.POSITIONAL,
                counts_as_positional=True,
                argument_value="positional",
                strength=defended_piece_support_strength(moved_value),
                defended_piece_value=moved_value,
            )
        )
        score += 15
    if en_pris:
        exchange_gain = captured_value + promotion_value - moved_value
        if gives_check and exchange_gain >= -100:
            label = f"tactical:checking_exchange_pressure:{move.uci()}:{exchange_gain}"
            reasons.append(label)
            reason_evidence.append(
                support(
                    label,
                    world=EvidenceWorld.TACTICAL,
                    counts_as_tactical=True,
                    argument_value="tactical",
                    strength=3,
                )
            )
        elif exchange_gain < 0:
            label = f"safety:moved_piece_en_pris:{moved_value}"
            objections.append(label)
            objection_evidence_items.append(
                objection(
                    label,
                    kind=ObjectionKind.MOVED_PIECE_EN_PRIS,
                    strength=material_cost_objection_strength(moved_value),
                    world=EvidenceWorld.MATERIAL,
                    moved_piece_en_pris_value=moved_value,
                    argument_value="material_safety",
                )
            )
            score += exchange_gain
            if moved_value >= OWNED_PIECE_VALUE["q"] and exchange_gain <= -300:
                label = f"safety:queen_blunder:{move.uci()}:{-exchange_gain}"
                objections.append(label)
                objection_evidence_items.append(
                    objection(
                        label,
                        kind=ObjectionKind.QUEEN_BLUNDER,
                        strength=2,
                        world=EvidenceWorld.MATERIAL,
                        argument_value="material_safety",
                    )
                )
                score -= moved_value
        else:
            label = f"material:exchange_nonnegative:{exchange_gain}"
            reasons.append(label)
            reason_evidence.append(display_evidence(label, world=EvidenceWorld.MATERIAL))
    return tuple(reasons), tuple(reason_evidence), tuple(objections), tuple(objection_evidence_items), score


def moved_piece_threat_labels(
    board: OwnedBoard,
    move: Any,
    child: OwnedBoard,
) -> EvidenceLabels:
    moved_piece = board.piece_at(move.from_square)
    if moved_piece is None:
        return EvidenceLabels(())
    targets = []
    for square, piece in enumerate(child.squares):
        if piece is None or piece_color(piece) != child.turn:
            continue
        if moved_piece_attacks_square(child, move.to_square, square, moved_piece):
            value = OWNED_PIECE_VALUE[piece.lower()]
            targets.append(value)
    target_value = sum(targets)
    if target_value < 500:
        return EvidenceLabels(())
    label = f"tactical:threat:targets:{len(targets)}:value:{target_value}"
    return EvidenceLabels(
        (label,),
        (
            support(
                label,
                world=EvidenceWorld.TACTICAL,
                counts_as_tactical=True,
                argument_value="tactical",
                strength=6 if target_value >= 700 else 3,
                tactical_threat_value=target_value,
            ),
        ),
        min(target_value, 700),
    )


def opening_development_objections(
    board: OwnedBoard,
    move: Any,
    *,
    captured_value: int,
    gives_check: bool,
) -> EvidenceLabels:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return EvidenceLabels(())
    color = piece_color(piece)
    kind = piece.lower()
    undeveloped_minors = undeveloped_minor_count(board, color)
    if (
        kind in {"n", "b"}
        and gives_check
        and captured_value == 0
        and board.fullmove_number <= 10
        and undeveloped_minors >= 2
    ):
        label = f"opening:premature_minor_check:{move.uci()}:undeveloped_minors:{undeveloped_minors}"
        return (
            EvidenceLabels(
                (label,),
                (
                    objection(
                        label,
                        kind=ObjectionKind.OPENING_PREMATURE_MINOR_CHECK,
                        strength=1,
                    ),
                ),
                -900,
            )
        )
    if kind not in {"q", "r"}:
        return EvidenceLabels(())
    if captured_value >= OWNED_PIECE_VALUE["n"]:
        return EvidenceLabels(())
    if kind == "r" and captured_value == 0 and board.fullmove_number <= 20:
        label = f"opening:premature_rook:{move.uci()}:undeveloped_minors:{undeveloped_minors}"
        return EvidenceLabels(
            (label,),
            (
                objection(
                    label,
                    kind=ObjectionKind.OPENING_PREMATURE_ROOK,
                    strength=1,
                ),
            ),
            -250,
        )
    if kind != "q" or board.fullmove_number > 10 or undeveloped_minors < 2:
        return EvidenceLabels(())
    label = f"opening:premature_queen:{move.uci()}:undeveloped_minors:{undeveloped_minors}"
    return EvidenceLabels(
        (label,),
        (
            objection(
                label,
                kind=ObjectionKind.OPENING_PREMATURE_QUEEN,
                strength=1,
            ),
        ),
        -1_200,
    )


def undeveloped_minor_count(board: OwnedBoard, color: str) -> int:
    home_squares = (
        ("b1", "g1", "c1", "f1")
        if color == "w"
        else ("b8", "g8", "c8", "f8")
    )
    expected = ("N", "N", "B", "B") if color == "w" else ("n", "n", "b", "b")
    return sum(
        1
        for square, piece in zip(home_squares, expected, strict=True)
        if board.piece_at(square) == piece
    )


def opening_minor_retreat_objections(
    board: OwnedBoard,
    move: Any,
    *,
    captured_value: int,
    gives_check: bool,
) -> EvidenceLabels:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.lower() not in {"n", "b"}:
        return EvidenceLabels(())
    if board.fullmove_number > 20 or captured_value > 0 or gives_check:
        return EvidenceLabels(())
    color = piece_color(piece)
    if is_minor_home_square(move.from_square, piece):
        return EvidenceLabels(())
    to_rank = rank_of(move.to_square)
    retreats_to_home_ranks = to_rank <= 1 if color == "w" else to_rank >= 6
    if not retreats_to_home_ranks:
        return EvidenceLabels(())
    label = f"opening:minor_retreat:{move.uci()}"
    return EvidenceLabels(
        (label,),
        (
            objection(
                label,
                kind=ObjectionKind.OPENING_MINOR_RETREAT,
                strength=1,
            ),
        ),
        -400,
    )


def is_minor_home_square(square: int, piece: str) -> bool:
    if piece == "N":
        return square in {square_index("b1"), square_index("g1")}
    if piece == "B":
        return square in {square_index("c1"), square_index("f1")}
    if piece == "n":
        return square in {square_index("b8"), square_index("g8")}
    if piece == "b":
        return square in {square_index("c8"), square_index("f8")}
    return False


def opening_king_safety_objections(
    board: OwnedBoard,
    move: Any,
    *,
    captured_value: int = 0,
) -> EvidenceLabels:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.lower() != "k":
        return EvidenceLabels(())
    if move.kind == "castle" or board.fullmove_number > 20:
        return EvidenceLabels(())
    color = piece_color(piece)
    if board.in_check(color):
        if captured_value == 0 and not king_stays_on_home_rank(color, move.to_square):
            label = f"opening:king_center_flight:{move.uci()}"
            return EvidenceLabels(
                (label,),
                (
                    objection(
                        label,
                        kind=ObjectionKind.OPENING_KING_CENTER_FLIGHT,
                        strength=1,
                    ),
                ),
                -400,
            )
        return EvidenceLabels(())
    label = f"opening:king_walk:{move.uci()}"
    return EvidenceLabels(
        (label,),
        (
            objection(
                label,
                kind=ObjectionKind.OPENING_KING_WALK,
                strength=1,
            ),
        ),
        -400,
    )


def king_stays_on_home_rank(color: str, square: int) -> bool:
    return rank_of(square) == (0 if color == "w" else 7)


def flank_pawn_weakening_objections(
    board: OwnedBoard,
    move: Any,
) -> EvidenceLabels:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.lower() != "p" or board.fullmove_number > 20:
        return EvidenceLabels(())
    color = piece_color(piece)
    king_square = board.king_square(color)
    from_file = file_of(move.from_square)
    labels: list[str] = []
    evidence: list[ArgumentEvidence] = []
    score = 0
    if king_square in {square_index("g1"), square_index("g8")} and from_file in {6, 7}:
        label = f"king_safety:castled_flank_pawn_weakening:{move.uci()}"
        labels.append(label)
        evidence.append(objection(label, kind=ObjectionKind.CASTLED_FLANK_PAWN_WEAKENING, strength=1))
        score -= 900
    elif king_square in {square_index("c1"), square_index("c8")} and from_file in {0, 1, 2}:
        label = f"king_safety:castled_flank_pawn_weakening:{move.uci()}"
        labels.append(label)
        evidence.append(objection(label, kind=ObjectionKind.CASTLED_FLANK_PAWN_WEAKENING, strength=1))
        score -= 900
    elif from_file in {6, 7}:
        label = f"king_safety:flank_pawn_weakening:{move.uci()}"
        labels.append(label)
        evidence.append(objection(label, kind=ObjectionKind.FLANK_PAWN_WEAKENING, strength=1))
        score -= 900
    if labels and abs(rank_of(move.to_square) - rank_of(move.from_square)) == 2:
        label = f"king_safety:flank_pawn_lunge:{move.uci()}"
        labels.append(label)
        evidence.append(objection(label, kind=ObjectionKind.FLANK_PAWN_LUNGE, strength=1))
        score -= 400
    return EvidenceLabels(tuple(labels), tuple(evidence), score)


def advanced_flank_pawn_response_labels(
    board: OwnedBoard,
    move: Any,
    child: OwnedBoard,
) -> tuple[tuple[str, ...], tuple[ArgumentEvidence, ...], tuple[str, ...], tuple[ArgumentEvidence, ...], int]:
    threats = advanced_flank_pawn_threats(board, board.turn)
    if not threats:
        return (), (), (), (), 0
    child_threats = advanced_flank_pawn_threats(child, board.turn)
    if len(child_threats) < len(threats):
        label = f"king_safety:advanced_flank_pawn_response:{move.uci()}"
        return (
            (label,),
            (
                defeater_evidence(
                    label,
                    world=EvidenceWorld.POSITIONAL,
                    defeater_kind=DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE,
                    defeater_strength=defeater_strength(DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE),
                    counts_as_positional=True,
                    argument_value="positional",
                    support_strength=13,
                ),
            ),
            (),
            (),
            1_200,
        )
    labels = tuple(
        f"king_safety:unanswered_advanced_flank_pawn:{move.uci()}:{square_name(pawn_square)}:{square_name(target_square)}"
        for pawn_square, target_square in threats
    )
    return (
        (),
        (),
        labels,
        tuple(
            objection(
                label,
                kind=ObjectionKind.UNANSWERED_ADVANCED_FLANK_PAWN,
                strength=4,
            )
            for label in labels
        ),
        -1_500 * len(labels),
    )


def advanced_flank_pawn_threats(
    board: OwnedBoard,
    color: str,
) -> tuple[tuple[int, int], ...]:
    opponent = opposite(color)
    threats: list[tuple[int, int]] = []
    for target_square in sorted(king_flank_pawn_squares(color)):
        target_piece = board.piece_at(target_square)
        if target_piece is None or piece_color(target_piece) != color:
            continue
        for pawn_square, pawn in enumerate(board.squares):
            if pawn is None or pawn.lower() != "p" or piece_color(pawn) != opponent:
                continue
            if moved_piece_attacks_square(board, pawn_square, target_square, pawn):
                threats.append((pawn_square, target_square))
    return tuple(sorted(set(threats)))


def ignored_hanging_piece_objections(
    board: OwnedBoard,
    move: Any,
    child: OwnedBoard,
) -> EvidenceLabels:
    color = board.turn
    opponent = opposite(color)
    labels: list[str] = []
    evidence: list[ArgumentEvidence] = []
    score = 0
    for square, piece in enumerate(board.squares):
        if piece is None or piece_color(piece) != color:
            continue
        value = OWNED_PIECE_VALUE[piece.lower()]
        if value < OWNED_PIECE_VALUE["n"]:
            continue
        if not lower_value_attacker_exists(board, square, opponent, value):
            continue
        if move.from_square == square:
            continue
        if child.piece_at(square) == piece and lower_value_attacker_exists(child, square, opponent, value):
            label = f"safety:ignored_hanging_piece:{move.uci()}:{square_name(square)}:{value}"
            labels.append(label)
            evidence.append(
                objection(
                    label,
                    kind=ObjectionKind.IGNORED_HANGING_PIECE,
                    strength=material_cost_objection_strength(value),
                    world=EvidenceWorld.MATERIAL,
                    argument_value="material_safety",
                )
            )
            score -= value
    return EvidenceLabels(tuple(labels), tuple(evidence), score)


def lower_value_attacker_exists(
    board: OwnedBoard,
    square: int,
    attacker_color: str,
    target_value: int,
) -> bool:
    for attacker_square, attacker in enumerate(board.squares):
        if attacker is None or piece_color(attacker) != attacker_color:
            continue
        if OWNED_PIECE_VALUE[attacker.lower()] >= target_value:
            continue
        if moved_piece_attacks_square(board, attacker_square, square, attacker):
            return True
    return False


def king_is_castled(board: OwnedBoard, color: str) -> bool:
    king_square = board.king_square(color)
    if color == "w":
        return king_square in {square_index("g1"), square_index("c1")}
    return king_square in {square_index("g8"), square_index("c8")}


def queen_flank_invasion_objections(
    board: OwnedBoard,
    move: Any,
    child: OwnedBoard,
) -> EvidenceLabels:
    color = piece_color(board.piece_at(move.from_square) or ("P" if board.turn == "w" else "p"))
    vulnerable = king_flank_pawn_squares(color)
    labels: list[str] = []
    evidence: list[ArgumentEvidence] = []
    opponent = child.turn
    for queen_square, queen in enumerate(child.squares):
        if queen is None or queen.lower() != "q" or piece_color(queen) != opponent:
            continue
        for target in vulnerable:
            captured = child.piece_at(target)
            if (
                captured is not None
                and captured.lower() == "p"
                and moved_piece_attacks_square(child, queen_square, target, queen)
            ):
                label = f"king_safety:queen_flank_invasion:{move.uci()}:{square_name(target)}"
                labels.append(label)
                evidence.append(
                    objection(
                        label,
                        kind=ObjectionKind.QUEEN_FLANK_INVASION,
                        strength=9,
                    )
                )
    if not labels:
        return EvidenceLabels(())
    by_label = {item.label: item for item in evidence}
    unique_labels = tuple(sorted(set(labels)))
    return EvidenceLabels(
        unique_labels,
        tuple(by_label[label] for label in unique_labels),
        -2_000,
    )


def king_flank_pawn_squares(color: str) -> frozenset[int]:
    if color == "w":
        return frozenset({square_index("g2"), square_index("h2")})
    return frozenset({square_index("g7"), square_index("h7")})


def reply_mate_in_one_objections(
    child: OwnedBoard,
    move: Any,
) -> EvidenceLabels:
    replies = []
    for reply in child.legal_moves():
        if owned_is_checkmate(child.apply(reply)):
            replies.append(reply.uci())
    if not replies:
        return EvidenceLabels(())
    labels = tuple(
        f"tactical:allows_reply_mate_in_one:{move.uci()}:{reply}"
        for reply in sorted(replies)
    )
    return EvidenceLabels(
        labels,
        tuple(
            objection(
                label,
                kind=ObjectionKind.REPLY_MATE_IN_ONE,
                strength=6,
                world=EvidenceWorld.TACTICAL,
                forced_mate_distance=1,
                argument_value="reply_refutation",
            )
            for label in labels
        ),
        -100_000,
    )


def reply_forced_mate_objections(
    child: OwnedBoard,
    move: Any,
    *,
    mate_depth: int,
    deadline: float | None = None,
) -> EvidenceLabels:
    if owned_is_checkmate(child):
        return EvidenceLabels(())
    if not has_forced_mate(
        chess.Board(child.fen()), mate_depth=mate_depth, deadline=deadline
    ):
        return EvidenceLabels(())
    label = f"tactical:allows_reply_forced_mate_in_{mate_depth}:{move.uci()}"
    return EvidenceLabels(
        (label,),
        (
            objection(
                label,
                kind=ObjectionKind.REPLY_FORCED_MATE,
                strength=6 if mate_depth == 2 else 3,
                world=EvidenceWorld.TACTICAL,
                forced_mate_distance=mate_depth,
                argument_value="reply_refutation",
            ),
        ),
        -100_000,
    )


def scan_forced_reply_mates_for_candidate_moves(
    board: OwnedBoard,
    legal_moves: list[Any],
    probes: list[MoveProbe],
    *,
    dialectic_depth: int,
    search_depth: int,
    deadline: float | None = None,
) -> list[MoveProbe]:
    if search_depth not in {0, 1, 2}:
        return probes
    if search_depth == 0:
        candidate_limit = 8
    elif search_depth == 1:
        candidate_limit = 6
    else:
        candidate_limit = 12
    move_by_uci = {move.uci(): move for move in legal_moves}
    legal_move_count = len(legal_moves)
    scan_depth_one_mate_three = (
        search_depth == 1
        and legal_move_count <= 2
        and board.in_check(board.turn)
    )
    scan_depth_zero_positive_mate_three = search_depth == 0
    updated: dict[str, MoveProbe] = {}
    scanned: set[str] = set()
    scan_budget = candidate_limit * 3 if search_depth == 1 else candidate_limit
    while len(scanned) < scan_budget:
        if deadline is not None and time.monotonic() >= deadline:
            break
        current_probes = [updated.get(probe.uci, probe) for probe in probes]
        made_progress = False
        remaining_budget = scan_budget - len(scanned)
        for probe in forced_reply_mate_scan_candidates(
            board,
            move_by_uci,
            current_probes,
            dialectic_depth=dialectic_depth,
            search_depth=search_depth,
            candidate_limit=min(candidate_limit, remaining_budget),
            legal_move_count=legal_move_count,
        ):
            if probe.uci in scanned:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                break
            scanned.add(probe.uci)
            made_progress = True
            move = move_by_uci[probe.uci]
            mate_depths = forced_reply_mate_depths(
                probe,
                board=board,
                move=move,
                search_depth=search_depth,
                scan_depth_one_mate_three=scan_depth_one_mate_three,
                scan_depth_zero_positive_mate_three=scan_depth_zero_positive_mate_three,
                legal_move_count=legal_move_count,
            )
            child = board.apply(move)
            forced_mate: EvidenceLabels = EvidenceLabels(())
            forced_mate_score = 0
            for mate_depth in mate_depths:
                forced_mate = reply_forced_mate_objections(
                    child,
                    move,
                    mate_depth=mate_depth,
                    deadline=deadline,
                )
                forced_mate_score = forced_mate.score
                if forced_mate.labels:
                    break
            if not forced_mate.labels:
                continue
            updated[probe.uci] = replace(
                probe,
                score=probe.score + forced_mate_score,
                objections=probe.objections + forced_mate.labels,
                objection_evidence=probe.objection_evidence + forced_mate.evidence,
            )
        if not made_progress:
            break
        if search_depth != 1:
            break
        if len(scanned) >= scan_budget:
            break
        if not updated:
            break
    return [updated.get(probe.uci, probe) for probe in probes]


def forced_reply_mate_scan_candidates(
    board: OwnedBoard,
    move_by_uci: dict[str, Any],
    probes: list[MoveProbe],
    *,
    dialectic_depth: int,
    search_depth: int,
    candidate_limit: int,
    legal_move_count: int,
) -> list[MoveProbe]:
    eligible = [
        probe
        for probe in probes
        if should_consider_forced_reply_mate_candidate(
            probe,
            dialectic_depth=dialectic_depth,
            search_depth=search_depth,
        )
        if should_scan_reply_forced_mate(
            search_depth,
            board,
            move_by_uci[probe.uci],
            reason_evidence=list(probe.reason_evidence),
            objection_evidence=list(probe.objection_evidence),
            legal_move_count=legal_move_count,
        )
    ]
    if len(eligible) <= candidate_limit:
        return sorted(eligible, key=lambda candidate: (-candidate.score, candidate.uci))

    score_budget = max(1, candidate_limit // 2)
    selected: dict[str, MoveProbe] = {}
    for probe in sorted(eligible, key=lambda candidate: (-candidate.score, candidate.uci))[:score_budget]:
        selected[probe.uci] = probe
    for probe in sorted(
        eligible,
        key=lambda candidate: (
            forced_reply_mate_risk_sort_key(candidate.objection_evidence)
            if search_depth == 0
            else 1,
            search_refutation_sort_key(candidate.objection_evidence),
            -candidate.score,
            candidate.uci,
        ),
    ):
        selected.setdefault(probe.uci, probe)
        if len(selected) >= candidate_limit:
            break
    return list(selected.values())


def forced_reply_mate_risk_sort_key(objections: tuple[ArgumentEvidence, ...]) -> int:
    if any(
        objection.objection_kind
        in {
            ObjectionKind.FLANK_PAWN_WEAKENING,
            ObjectionKind.CASTLED_FLANK_PAWN_WEAKENING,
            ObjectionKind.FLANK_PAWN_LUNGE,
            ObjectionKind.QUEEN_FLANK_INVASION,
            ObjectionKind.UNANSWERED_ADVANCED_FLANK_PAWN,
            ObjectionKind.OPENING_KING_WALK,
            ObjectionKind.OPENING_KING_CENTER_FLIGHT,
            ObjectionKind.QUEEN_BLUNDER,
        }
        for objection in objections
    ):
        return 0
    return 1


def should_consider_forced_reply_mate_candidate(
    probe: MoveProbe,
    *,
    dialectic_depth: int,
    search_depth: int,
) -> bool:
    if search_depth == 0 and dialectic_depth != 0:
        return not (probe.gives_check or probe.captured_value > 0 or probe.promotion_value > 0)
    return True


def forced_reply_mate_depths(
    probe: MoveProbe,
    *,
    board: OwnedBoard,
    move: Any,
    search_depth: int,
    scan_depth_one_mate_three: bool,
    scan_depth_zero_positive_mate_three: bool,
    legal_move_count: int,
) -> tuple[int, ...]:
    if scan_depth_one_mate_three:
        return (2, 3)
    if scan_depth_zero_positive_mate_three and probe.score > 0:
        return (2, 3)
    if search_depth == 1 and is_deeply_refuted_major_move(board, move, probe.objection_evidence):
        return (2, 3)
    if (
        search_depth == 1
        and legal_move_count <= 20
        and has_search_refutation_at_most(list(probe.objection_evidence), -700)
    ):
        return (2, 3)
    if search_depth in {0, 1}:
        return (2,)
    return (2, 3)


def is_deeply_refuted_major_move(
    board: OwnedBoard,
    move: Any,
    objections: tuple[ArgumentEvidence, ...],
) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.lower() not in {"q", "r"}:
        return False
    return has_search_refutation_at_most(list(objections), -1_500)


def search_refutation_sort_key(objections: tuple[ArgumentEvidence, ...]) -> int:
    scores = [
        score
        for objection in objections
        if (score := objection.search_refutation_score) is not None
    ]
    return min(scores, default=0)


def should_scan_reply_mate(
    search_depth: int,
    board: OwnedBoard,
    move: Any,
    *,
    captured_value: int,
    reason_evidence: list[ArgumentEvidence],
    objection_evidence: list[ArgumentEvidence],
) -> bool:
    if search_depth == 0:
        return True
    if search_depth != 1:
        return False
    piece = board.piece_at(move.from_square)
    if piece is not None and piece.lower() == "k":
        return True
    if captured_value >= OWNED_PIECE_VALUE["n"]:
        return True
    if piece is not None and piece.lower() in {"q", "r"} and has_tactical_threat_reason(reason_evidence):
        return True
    return any(
        objection.objection_kind
        in {
            ObjectionKind.FLANK_PAWN_WEAKENING,
            ObjectionKind.CASTLED_FLANK_PAWN_WEAKENING,
            ObjectionKind.FLANK_PAWN_LUNGE,
            ObjectionKind.QUEEN_FLANK_INVASION,
            ObjectionKind.UNANSWERED_ADVANCED_FLANK_PAWN,
            ObjectionKind.OPENING_KING_WALK,
            ObjectionKind.OPENING_KING_CENTER_FLIGHT,
            ObjectionKind.OPENING_MINOR_RETREAT,
        }
        for objection in objection_evidence
    )


def should_scan_reply_forced_mate(
    search_depth: int,
    board: OwnedBoard,
    move: Any,
    *,
    reason_evidence: list[ArgumentEvidence],
    objection_evidence: list[ArgumentEvidence],
    legal_move_count: int | None = None,
) -> bool:
    if search_depth not in {0, 1, 2}:
        return False
    piece = board.piece_at(move.from_square)
    if piece is None:
        return False
    if search_depth == 0:
        return True
    if search_depth == 1:
        if piece.lower() == "k":
            return True
        if piece.lower() in {"q", "r"} and has_search_refutation_at_most(objection_evidence, -100):
            return True
        if has_search_refutation_at_most(objection_evidence, -200):
            return True
        if (
            legal_move_count is not None
            and legal_move_count <= 2
            and board.in_check(board.turn)
            and has_search_refutation_at_most(objection_evidence, -700)
        ):
            return True
        if (
            legal_move_count is not None
            and legal_move_count <= 20
            and has_search_refutation_at_most(objection_evidence, -700)
        ):
            return True
        has_threat_reason = has_tactical_threat_reason(reason_evidence)
        if (
            piece.lower() != "p"
            and has_threat_reason
            and has_search_refutation_at_most(objection_evidence, -100)
        ):
            return True
        return has_search_refutation_at_most(objection_evidence, -700) and (
            piece.lower() != "p" or has_threat_reason
        )
    if piece.lower() == "k":
        return True
    if has_large_search_refutation(objection_evidence):
        return True
    if piece.lower() in {"n", "b", "r", "q"} and has_material_capture_at_least(
        reason_evidence,
        OWNED_PIECE_VALUE["n"],
    ):
        return True
    if piece.lower() in {"q", "r"} and has_search_refutation_at_most(objection_evidence, -400):
        return True
    has_threat_reason = has_tactical_threat_reason(reason_evidence)
    if not has_threat_reason:
        return False
    if piece.lower() in {"q", "r"}:
        return True
    return any(
        objection.objection_kind == ObjectionKind.MOVED_PIECE_EN_PRIS
        for objection in objection_evidence
    )


def has_large_search_refutation(objections: list[ArgumentEvidence]) -> bool:
    return has_search_refutation_at_most(objections, -1_000)


def has_material_capture_at_least(reasons: list[ArgumentEvidence], threshold: int) -> bool:
    return any(
        reason.world == EvidenceWorld.MATERIAL
        and reason.support_strength >= material_support_strength(threshold)
        for reason in reasons
    )


def has_tactical_threat_reason(reasons: list[ArgumentEvidence]) -> bool:
    return any(reason.tactical_threat_value > 0 for reason in reasons)


def has_search_refutation_at_most(objections: list[ArgumentEvidence], threshold: int) -> bool:
    for objection in objections:
        score = objection.search_refutation_score
        if score is not None and score <= threshold:
            return True
    return False


def ensure_owned_board(board: Any) -> OwnedBoard:
    if isinstance(board, OwnedBoard):
        return board
    return owned_board_from_fen(board.fen())


def owned_board_from_fen(fen: str) -> OwnedBoard:
    return OwnedBoard.from_fen(fen)


def positional_reason_labels(board: OwnedBoard, move: Any, child: OwnedBoard) -> EvidenceLabels:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return EvidenceLabels(())
    labels: list[str] = []
    evidence: list[ArgumentEvidence] = []
    move_text = move.uci()
    kind = piece.lower()
    color = piece_color(piece)
    from_rank = rank_of(move.from_square)
    to_rank = rank_of(move.to_square)

    if kind == "p" and file_of(move.from_square) in {3, 4} and abs(to_rank - from_rank) == 2:
        label = f"development:{move_text}:center_pawn"
        labels.append(label)
        evidence.append(support(label, world=EvidenceWorld.POSITIONAL, counts_as_positional=True, argument_value="positional", strength=1, support_kind=SupportKind.DEVELOPMENT))
    if kind in {"n", "b"} and from_rank == (0 if color == "w" else 7):
        label = f"development:{move_text}:minor_piece"
        labels.append(label)
        evidence.append(support(label, world=EvidenceWorld.POSITIONAL, counts_as_positional=True, argument_value="positional", strength=1, support_kind=SupportKind.DEVELOPMENT))
    if move.kind == "castle":
        label = f"king_safety:{move_text}:castle"
        labels.append(label)
        evidence.append(support(label, world=EvidenceWorld.POSITIONAL, counts_as_positional=True, argument_value="positional", strength=1))

    center_count = moved_piece_center_control(child, move.to_square, piece)
    if center_count:
        label = f"center_control:{move_text}:{center_count}"
        labels.append(label)
        evidence.append(support(label, world=EvidenceWorld.POSITIONAL, counts_as_positional=True, argument_value="positional", strength=1))
    activity_gain = moved_piece_activity_gain(board, child, move.from_square, move.to_square, piece)
    if activity_gain > 0:
        label = f"piece_activity:{move_text}:mobility_gain:{activity_gain}"
        labels.append(label)
        evidence.append(support(label, world=EvidenceWorld.POSITIONAL, counts_as_positional=True, argument_value="positional", strength=1))
    if kind == "p" and is_passed_pawn(child, move.to_square, color):
        label = f"pawn_structure:{move_text}:passed_pawn"
        labels.append(label)
        evidence.append(support(label, world=EvidenceWorld.POSITIONAL, counts_as_positional=True, argument_value="positional", strength=1))
    if kind in {"r", "q"} and controls_open_file(child, move.to_square):
        label = f"file_control:{move_text}:open_file"
        labels.append(label)
        evidence.append(support(label, world=EvidenceWorld.POSITIONAL, counts_as_positional=True, argument_value="positional", strength=1))
    if kind == "n" and is_supported_outpost(child, move.to_square, color):
        label = f"outpost:{move_text}:supported"
        labels.append(label)
        evidence.append(support(label, world=EvidenceWorld.POSITIONAL, counts_as_positional=True, argument_value="positional", strength=1))
    return EvidenceLabels(tuple(labels), tuple(evidence))


def moved_piece_center_control(board: OwnedBoard, source_square: int, piece: str) -> int:
    return sum(
        1
        for target in (
            square_index("d4"),
            square_index("e4"),
            square_index("d5"),
            square_index("e5"),
        )
        if moved_piece_attacks_square(board, source_square, target, piece)
    )


def controls_open_file(board: OwnedBoard, square: int) -> bool:
    file_index = file_of(square)
    return all(
        piece is None or piece.lower() != "p"
        for index, piece in enumerate(board.squares)
        if file_of(index) == file_index
    )


def moved_piece_activity_gain(
    before: OwnedBoard,
    after: OwnedBoard,
    from_square: int,
    to_square: int,
    piece: str,
) -> int:
    before_activity = moved_piece_activity(before, from_square, piece)
    after_activity = moved_piece_activity(after, to_square, piece)
    return after_activity - before_activity


def moved_piece_activity(board: OwnedBoard, square: int, piece: str) -> int:
    return sum(
        1
        for target in range(64)
        if target != square and moved_piece_attacks_square(board, square, target, piece)
    )


def is_passed_pawn(board: OwnedBoard, square: int, color: str) -> bool:
    opponent_pawn = "p" if color == "w" else "P"
    start_rank = rank_of(square) + (1 if color == "w" else -1)
    stop_rank = 8 if color == "w" else -1
    step = 1 if color == "w" else -1
    for file_index in range(max(0, file_of(square) - 1), min(7, file_of(square) + 1) + 1):
        for rank_index in range(start_rank, stop_rank, step):
            if board.piece_at(rank_index * 8 + file_index) == opponent_pawn:
                return False
    return True


def is_supported_outpost(board: OwnedBoard, square: int, color: str) -> bool:
    rank = rank_of(square)
    if color == "w" and rank < 3:
        return False
    if color == "b" and rank > 4:
        return False
    support_rank = rank - 1 if color == "w" else rank + 1
    support_piece = "P" if color == "w" else "p"
    for file_delta in (-1, 1):
        support_file = file_of(square) + file_delta
        if 0 <= support_file < 8:
            support_square = support_rank * 8 + support_file
            if 0 <= support_square < 64 and board.piece_at(support_square) == support_piece:
                return True
    return False
