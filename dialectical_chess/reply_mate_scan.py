"""Forced reply-mate scan helpers for move probing."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import chess

from dialectical_chess.arguments import MoveProbe
from dialectical_chess.board import OwnedBoard
from dialectical_chess.evidence import (
    ArgumentEvidence,
    EvidenceWorld,
    ObjectionKind,
    ObjectionEvidence,
    SupportEvidence,
    DefeaterEvidence,
    has_search_refutation_at_most,
)
from dialectical_chess.heuristics.evidence import EvidenceLabels, material_support_strength, objection
from dialectical_chess.loss_mining import has_forced_mate
from dialectical_chess.search import OWNED_PIECE_VALUE, owned_is_checkmate
from dialectical_chess.tuning import REPLY_MATE_REFUTATION_SCORE, SEARCH_REPLY_MATE_TRIGGER_SCORE

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
        REPLY_MATE_REFUTATION_SCORE,
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
        REPLY_MATE_REFUTATION_SCORE,
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
        isinstance(objection, ObjectionEvidence)
        and
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
        and has_search_refutation_at_most(list(probe.objection_evidence), SEARCH_REPLY_MATE_TRIGGER_SCORE)
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
        if isinstance(objection, ObjectionEvidence)
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
        isinstance(objection, ObjectionEvidence)
        and
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
            and has_search_refutation_at_most(objection_evidence, SEARCH_REPLY_MATE_TRIGGER_SCORE)
        ):
            return True
        if (
            legal_move_count is not None
            and legal_move_count <= 20
            and has_search_refutation_at_most(objection_evidence, SEARCH_REPLY_MATE_TRIGGER_SCORE)
        ):
            return True
        has_threat_reason = has_tactical_threat_reason(reason_evidence)
        if (
            piece.lower() != "p"
            and has_threat_reason
            and has_search_refutation_at_most(objection_evidence, -100)
        ):
            return True
        return has_search_refutation_at_most(objection_evidence, SEARCH_REPLY_MATE_TRIGGER_SCORE) and (
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
        isinstance(objection, ObjectionEvidence)
        and objection.objection_kind == ObjectionKind.MOVED_PIECE_EN_PRIS
        for objection in objection_evidence
    )
def has_large_search_refutation(objections: list[ArgumentEvidence]) -> bool:
    return has_search_refutation_at_most(objections, -1_000)
def has_material_capture_at_least(reasons: list[ArgumentEvidence], threshold: int) -> bool:
    return any(
        isinstance(reason, SupportEvidence | DefeaterEvidence)
        and reason.world == EvidenceWorld.MATERIAL
        and reason.support_strength >= material_support_strength(threshold)
        for reason in reasons
    )


def has_tactical_threat_reason(reasons: list[ArgumentEvidence]) -> bool:
    return any(
        isinstance(reason, SupportEvidence | DefeaterEvidence)
        and reason.tactical_threat_value > 0
        for reason in reasons
    )

