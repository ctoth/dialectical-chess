"""King Safety heuristic label/evidence producers."""

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
    FLANK_PAWN_LUNGE_PENALTY,
    FLANK_PAWN_WEAKENING_PENALTY,
    KING_ESCAPE_SCORE,
    MAJOR_PIECE_VALUE,
    MOVED_PIECE_DEFENDED_SCORE,
    QUEEN_BLUNDER_EXCHANGE_THRESHOLD,
    UNSUPPORTED_MAJOR_DRIFT_PENALTY,
)


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
        KING_ESCAPE_SCORE,
    )

def king_adjacent(left: int, right: int) -> bool:
    return max(
        abs(file_of(left) - file_of(right)),
        abs(rank_of(left) - rank_of(right)),
    ) == 1

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
        score += FLANK_PAWN_WEAKENING_PENALTY
    elif king_square in {square_index("c1"), square_index("c8")} and from_file in {0, 1, 2}:
        label = f"king_safety:castled_flank_pawn_weakening:{move.uci()}"
        labels.append(label)
        evidence.append(objection(label, kind=ObjectionKind.CASTLED_FLANK_PAWN_WEAKENING, strength=1))
        score += FLANK_PAWN_WEAKENING_PENALTY
    elif from_file in {6, 7}:
        label = f"king_safety:flank_pawn_weakening:{move.uci()}"
        labels.append(label)
        evidence.append(objection(label, kind=ObjectionKind.FLANK_PAWN_WEAKENING, strength=1))
        score += FLANK_PAWN_WEAKENING_PENALTY
    if labels and abs(rank_of(move.to_square) - rank_of(move.from_square)) == 2:
        label = f"king_safety:flank_pawn_lunge:{move.uci()}"
        labels.append(label)
        evidence.append(objection(label, kind=ObjectionKind.FLANK_PAWN_LUNGE, strength=1))
        score += FLANK_PAWN_LUNGE_PENALTY
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
                # Carry the material magnitude (the value of the pawn the queen
                # is now en-prise to / threatening on the flank square) so the
                # FACT route in ``core_labels.core_objection_label`` produces
                # ``obj:loses_exchange:{n}`` instead of falling through to
                # the HEURISTIC dispatcher. ``captured`` was just verified to
                # be a pawn at L206; ``OWNED_PIECE_VALUE`` gives the canonical
                # centipawn value (pawn = 100) and keeps the lookup robust if
                # the captured-piece guard is later loosened to cover knights
                # or bishops on the flank.
                en_pris_value = OWNED_PIECE_VALUE.get(captured.lower(), 0)
                evidence.append(
                    objection(
                        label,
                        kind=ObjectionKind.QUEEN_FLANK_INVASION,
                        strength=9,
                        moved_piece_en_pris_value=en_pris_value,
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
