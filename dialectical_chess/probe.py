"""Move probing for dialectical chess."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from dialectical_chess.arguments import MoveProbe
from dialectical_chess.board import OwnedBoard
from dialectical_chess.evidence import (
    ArgumentEvidence,
    DefeaterKind,
    EvidenceWorld,
    ObjectionKind,
    defeater_evidence,
    defeater_strength,
)
from dialectical_chess.heuristics.evidence import display_evidence, objection, search_refutation_strength, support, material_support_strength
from dialectical_chess.heuristics.standard import (
    advanced_flank_pawn_response_labels,
    draw_objections,
    flank_pawn_weakening_objections,
    fork_witness_labels,
    has_reply_mate_in_one_objection,
    ignored_hanging_piece_objections,
    king_escape_square_reasons,
    moved_piece_safety_labels,
    moved_piece_threat_labels,
    opening_development_objections,
    opening_king_safety_objections,
    opening_minor_retreat_objections,
    positional_reason_labels,
    queen_flank_invasion_objections,
    unsupported_major_drift_objections,
)
from dialectical_chess.reply_mate_scan import (
    reply_mate_in_one_objections,
    scan_forced_reply_mates_for_candidate_moves,
    should_scan_reply_mate,
)
from dialectical_chess.search import (
    OWNED_PIECE_VALUE,
    ReplyAnalysisCache,
    ReplyAnalysisSettings,
    SearchSettings,
    bounded_reply_attack_evidence,
    owned_capture_value,
    owned_is_capture,
    owned_is_checkmate,
    owned_is_draw,
    owned_is_stalemate,
    owned_is_threefold_repetition,
    append_position_history,
    root_search_result,
)
from dialectical_chess.smt import SmtSettings, smt_fork_witnesses, smt_mate_in_one_moves
from dialectical_chess.tuning import (
    CHECKMATE_SCORE,
    CHECK_SCORE,
    POSITIONAL_REASON_SCORE,
    SEARCH_REPLY_MATE_TRIGGER_SCORE,
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
            score += CHECKMATE_SCORE
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
            score += CHECK_SCORE
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
                score += POSITIONAL_REASON_SCORE * len(positional.labels)
                reasons.extend(positional.labels)
                reason_evidence.extend(positional.evidence)
        smt_witnesses: list[str] = []
        if not is_draw and move.uci() in smt_mate_moves:
            score += CHECKMATE_SCORE
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
                and search_result.score <= SEARCH_REPLY_MATE_TRIGGER_SCORE
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


def ensure_owned_board(board: Any) -> OwnedBoard:
    if isinstance(board, OwnedBoard):
        return board
    return owned_board_from_fen(board.fen())


def owned_board_from_fen(fen: str) -> OwnedBoard:
    return OwnedBoard.from_fen(fen)


