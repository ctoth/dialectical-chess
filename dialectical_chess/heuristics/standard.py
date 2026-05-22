"""Standard heuristic label/evidence producers for move probing."""

from __future__ import annotations

from typing import Any

from dialectical_chess.board import OwnedBoard, file_of, opposite, piece_color, rank_of, square_index, square_name
from dialectical_chess.evidence import ArgumentEvidence, DefeaterKind, EvidenceWorld, ObjectionKind, SupportKind, defeater_evidence, defeater_strength, material_cost_objection_strength
from dialectical_chess.heuristics.evidence import EvidenceLabels, display_evidence, objection, support, material_support_strength, defended_piece_support_strength
from dialectical_chess.search import OWNED_PIECE_VALUE, owned_is_threefold_repetition
from dialectical_chess.smt import moved_piece_attacks_square
from dialectical_chess.tuning import (
    KING_ESCAPE_SCORE,
    MAJOR_PIECE_VALUE,
    MOVED_PIECE_DEFENDED_SCORE,
    QUEEN_BLUNDER_EXCHANGE_THRESHOLD,
    REPLY_MATE_REFUTATION_SCORE,
    UNSUPPORTED_MAJOR_DRIFT_PENALTY,
)

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
        UNSUPPORTED_MAJOR_DRIFT_PENALTY,
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
        KING_ESCAPE_SCORE,
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
