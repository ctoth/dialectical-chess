"""Piece Safety heuristic label/evidence producers."""

from __future__ import annotations

from typing import Any

from dialectical_chess.board import (
    OwnedBoard,
    file_of,
    opposite,
    piece_color,
    rank_of,
    square_index,
    square_name,
)
from dialectical_chess.evidence import (
    ArgumentEvidence,
    DefeaterKind,
    EvidenceWorld,
    ObjectionKind,
    SupportKind,
    defeater_evidence,
    defeater_strength,
    material_cost_objection_strength,
)
from dialectical_chess.heuristics.evidence import (
    EvidenceLabels,
    defended_piece_support_strength,
    display_evidence,
    material_support_strength,
    objection,
    support,
)
from dialectical_chess.search import OWNED_PIECE_VALUE, owned_is_threefold_repetition
from dialectical_chess.smt import moved_piece_attacks_square
from dialectical_chess.tuning import (
    KING_ESCAPE_SCORE,
    MAJOR_PIECE_VALUE,
    MOVED_PIECE_DEFENDED_SCORE,
    QUEEN_BLUNDER_EXCHANGE_THRESHOLD,
    UNSUPPORTED_MAJOR_DRIFT_PENALTY,
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
                support_magnitude=moved_value,
                support_kind=SupportKind.PIECE_DEFENDED,
            )
        )
        score += MOVED_PIECE_DEFENDED_SCORE
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
                    support_magnitude=max(1, exchange_gain),
                    support_kind=SupportKind.CHECKING_EXCHANGE_PRESSURE,
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
            if moved_value >= OWNED_PIECE_VALUE["q"] and exchange_gain <= QUEEN_BLUNDER_EXCHANGE_THRESHOLD:
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
    if target_value < MAJOR_PIECE_VALUE:
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
                support_magnitude=target_value,
                support_kind=SupportKind.TACTICAL_THREAT,
            ),
        ),
        min(target_value, 700),
    )

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
