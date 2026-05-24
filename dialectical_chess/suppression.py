"""Chess-side suppression and material-safety policy (the chess cartridge).

This module is the chess half of the cartridge seam. The generic
argumentation machinery (``opinion_graph``, ``decide``) is game-agnostic — it
reads only ``evidence.tier`` and ``evidence`` strengths. Every chess-specific
rule that used to be fused into that machinery lives here:

* **Defeater suppression (design D3).** ``suppressing_defeaters`` is the
  chess suppression policy — the hard-coded ``QUEEN_BLUNDER`` /
  ``MOVED_PIECE_EN_PRIS`` / ``OPENING_PREMATURE_MINOR_CHECK`` /
  ``FLANK_PAWN_*`` rules that used to sit inline in the generic graph
  builder's ``objection_defeater_evidence``. The generic builder now calls
  this policy through a single hook; it never names a chess objection kind.

* **Material-safety FACT classification (design D2).** ``fact_material_loss``
  is the chess policy that scores a move's worst proven material loss. Chess's
  ``material_safety`` penalties used to be smuggled into the opinion
  base-rate and the argmax key; they are now an honest FACT-tier decision
  term the generic lexicographic decider consults through this one function.

The generic layer depends on this module; this module depends on the chess
evidence vocabulary — never the reverse. That is the seam.
"""

from __future__ import annotations

from dialectical_chess.arguments import MoveProbe
from dialectical_chess.evidence import (
    ArgumentEvidence,
    COMPENSATING_TACTICAL_THREAT_THRESHOLD,
    DefeaterEvidence,
    DefeaterKind,
    EvidenceWorld,
    ObjectionEvidence,
    ObjectionKind,
    ReplyEvidence,
    SupportEvidence,
    SupportKind,
    defeater_evidence as make_defeater_evidence,
    has_search_refutation_at_most,
)

# --- defeater suppression policy (design D3) --------------------------------
#
# A defeater suppresses an objection by cancelling part of its strength at
# aggregation time (it is never a graph node — design v2 §1e). The four rules
# below are the chess-specific suppression knowledge. The generic graph builder
# calls ``suppressing_defeaters`` and reads only the returned defeater
# strengths; it never sees a chess objection kind.

DEFEATER_SEARCH_SUPPORT_STRENGTH = 97
DEFEATER_COMPENSATION_STRENGTH = 33
DEFEATER_TACTICAL_PRESSURE_STRENGTH = 17


def defeater_strength_for(defeater_kind: DefeaterKind) -> int:
    """Return the suppression strength a synthetic defeater kind carries."""
    match defeater_kind:
        case DefeaterKind.SEARCH_SUPPORT:
            return DEFEATER_SEARCH_SUPPORT_STRENGTH
        case (
            DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE
            | DefeaterKind.COMPENSATING_FORCING_PRESSURE
            | DefeaterKind.FORCING_MATERIAL_GAIN
        ):
            return DEFEATER_COMPENSATION_STRENGTH
        case DefeaterKind.COMPENSATING_TACTICAL_PRESSURE:
            return DEFEATER_TACTICAL_PRESSURE_STRENGTH


def synthetic_defeater_evidence(defeater_kind: DefeaterKind) -> ArgumentEvidence:
    """Build a synthetic chess defeater for a suppression rule that fired."""
    return make_defeater_evidence(
        f"defeater:{defeater_kind.value}",
        world=EvidenceWorld.PROCEDURAL,
        defeater_kind=defeater_kind,
        defeater_strength=defeater_strength_for(defeater_kind),
    )


def suppressing_defeaters(
    probe: MoveProbe,
    objection: ArgumentEvidence,
) -> tuple[ArgumentEvidence, ...]:
    """Return the chess defeaters that suppress ``objection`` on ``probe``.

    The chess suppression policy (design D3). A ``ReplyEvidence`` carrying a
    positive ``defense_strength`` suppresses itself (a defended reply). An
    ``ObjectionEvidence`` is suppressed by the chess-specific rules below:

    * ``QUEEN_BLUNDER`` + compensating forcing pressure;
    * ``MOVED_PIECE_EN_PRIS`` (en-pris value >= 300) + compensating tactical
      pressure and/or a forcing material gain;
    * ``OPENING_PREMATURE_MINOR_CHECK`` + a typed search-support defeater;
    * ``FLANK_PAWN_WEAKENING`` / ``FLANK_PAWN_LUNGE`` + a typed advanced-flank
      -pawn-response defeater.

    These four rules used to be fused inline into the generic graph builder;
    they are the chess cartridge's knowledge and live only here.
    """
    if isinstance(objection, ReplyEvidence):
        if objection.defense_strength > 0:
            return (objection,)
        return ()
    if not isinstance(objection, ObjectionEvidence):
        return ()

    defeaters: list[ArgumentEvidence] = []
    if (
        objection.objection_kind == ObjectionKind.QUEEN_BLUNDER
        and has_compensating_forcing_pressure(probe)
    ):
        defeaters.append(
            synthetic_defeater_evidence(DefeaterKind.COMPENSATING_FORCING_PRESSURE)
        )
    if (
        objection.objection_kind == ObjectionKind.MOVED_PIECE_EN_PRIS
        and objection.moved_piece_en_pris_value is not None
        and objection.moved_piece_en_pris_value >= 300
    ):
        if has_compensating_tactical_pressure(probe):
            defeaters.append(
                synthetic_defeater_evidence(
                    DefeaterKind.COMPENSATING_TACTICAL_PRESSURE
                )
            )
        if has_forcing_material_gain(probe):
            defeaters.append(
                synthetic_defeater_evidence(DefeaterKind.FORCING_MATERIAL_GAIN)
            )
    if (
        objection.objection_kind == ObjectionKind.OPENING_PREMATURE_MINOR_CHECK
        and has_typed_reason_defeater(probe, DefeaterKind.SEARCH_SUPPORT)
    ):
        defeaters.append(synthetic_defeater_evidence(DefeaterKind.SEARCH_SUPPORT))
    if (
        objection.objection_kind
        in {ObjectionKind.FLANK_PAWN_WEAKENING, ObjectionKind.FLANK_PAWN_LUNGE}
        and has_typed_reason_defeater(
            probe, DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE
        )
    ):
        defeaters.append(
            synthetic_defeater_evidence(DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE)
        )
    return tuple(defeaters)


def has_compensating_tactical_pressure(probe: MoveProbe) -> bool:
    """A reason carries a tactical threat at the compensation threshold."""
    return any(
        isinstance(evidence, SupportEvidence | DefeaterEvidence)
        and evidence.tactical_threat_value >= COMPENSATING_TACTICAL_THREAT_THRESHOLD
        for evidence in probe.reason_evidence
    )


def has_compensating_forcing_pressure(probe: MoveProbe) -> bool:
    """Compensating tactical pressure that is also forcing (check or gain)."""
    return has_compensating_tactical_pressure(probe) and (
        probe.gives_check or material_or_promotion_gain(probe) > 0
    )


def has_forcing_material_gain(probe: MoveProbe) -> bool:
    """The move gives check and nets immediate material / promotion."""
    return probe.gives_check and material_or_promotion_gain(probe) > 0


def has_typed_reason_defeater(probe: MoveProbe, defeater_kind: DefeaterKind) -> bool:
    """A reason carries a typed defeater of ``defeater_kind``."""
    return any(
        isinstance(evidence, DefeaterEvidence)
        and evidence.defeater_kind == defeater_kind
        for evidence in probe.reason_evidence
    )


def material_or_promotion_gain(probe: MoveProbe) -> int:
    """The immediate material the move captures plus any promotion value."""
    return probe.captured_value + probe.promotion_value


def has_development_reason(probe: MoveProbe) -> bool:
    """A reason is a development support witness."""
    development_kinds = {
        SupportKind.DEVELOPMENT,
        SupportKind.DEVELOPMENT_CENTER_PAWN,
        SupportKind.DEVELOPMENT_MINOR_PIECE,
    }
    return any(
        isinstance(evidence, SupportEvidence | DefeaterEvidence)
        and evidence.support_kind in development_kinds
        for evidence in probe.reason_evidence
    )


# --- material-safety FACT policy (design D2) --------------------------------
#
# Chess's ``material_safety`` penalties used to be smuggled into the opinion
# base-rate (``material_safety_prior_penalty_for``) and the argmax selection
# key (``material_safety_selection_penalty``). Design D2 reframes them as a
# single honest FACT-tier decision term: ``fact_material_loss`` scores the
# worst proven material loss a move walks into. The generic lexicographic
# decider consults this as its top-priority FACT term — a move with a proven
# material loss is ordered strictly after every move without one.

# A queen-flank invasion is a proven king-safety loss worth a knight-scale
# penalty; the en-pris / hanging losses are scaled by the lost material.
QUEEN_FLANK_INVASION_LOSS = 300
IGNORED_HANGING_PIECE_LOSS = 300
EN_PRIS_LOSS_SCALE = 4

# The search-refutation thresholds the FACT material-loss rules gate on.
EN_PRIS_FACT_REFUTATION_THRESHOLD = -400
HANGING_FACT_REFUTATION_THRESHOLD = -400
COMBINED_FACT_REFUTATION_THRESHOLD = -300

# A search-confirmed combined material collapse (an en-pris piece and an
# ignored hanging piece, both with a >= -300 search refutation) is the
# worst-case proven loss; it dominates every finite scaled magnitude.
COMBINED_MATERIAL_COLLAPSE_LOSS = 10**6


def _objection_prior_loss(
    probe: MoveProbe,
    objection: ObjectionEvidence,
) -> int:
    """The proven material loss a single objection contributes on ``probe``.

    The reframed ``material_safety_prior_penalty_for``: a queen-flank invasion
    (unless answered by development with no search refutation), a
    search-refuted en-pris piece, or a search-refuted ignored hanging piece.
    """
    if objection.objection_kind == ObjectionKind.QUEEN_FLANK_INVASION:
        if has_development_reason(probe) and not has_search_refutation_at_most(
            probe.objection_evidence, -300
        ):
            return 0
        return QUEEN_FLANK_INVASION_LOSS
    if (
        objection.objection_kind == ObjectionKind.MOVED_PIECE_EN_PRIS
        and objection.moved_piece_en_pris_value is not None
        and has_search_refutation_at_most(
            probe.objection_evidence, EN_PRIS_FACT_REFUTATION_THRESHOLD
        )
    ):
        return EN_PRIS_LOSS_SCALE * objection.moved_piece_en_pris_value
    if (
        objection.objection_kind == ObjectionKind.IGNORED_HANGING_PIECE
        and has_search_refutation_at_most(
            probe.objection_evidence, HANGING_FACT_REFUTATION_THRESHOLD
        )
    ):
        return IGNORED_HANGING_PIECE_LOSS
    return 0


def _has_moved_piece_en_pris_objection(probe: MoveProbe) -> bool:
    """The move walks a >= 300-value piece en pris."""
    return any(
        isinstance(evidence, ObjectionEvidence)
        and evidence.objection_kind == ObjectionKind.MOVED_PIECE_EN_PRIS
        and evidence.moved_piece_en_pris_value is not None
        and evidence.moved_piece_en_pris_value >= 300
        for evidence in probe.objection_evidence
    )


def _has_ignored_hanging_piece_objection(probe: MoveProbe) -> bool:
    """The move ignores a hanging piece."""
    return any(
        isinstance(evidence, ObjectionEvidence)
        and evidence.objection_kind == ObjectionKind.IGNORED_HANGING_PIECE
        for evidence in probe.objection_evidence
    )


def fact_material_loss(probe: MoveProbe) -> int:
    """Return the worst proven material loss ``probe`` walks into (design D2).

    The chess FACT-tier material-safety term. Non-zero exactly when the move
    walks into a proven material loss — a search-refuted en-pris piece, a
    search-refuted ignored hanging piece, an unanswered queen-flank invasion,
    or a search-confirmed combined material collapse. The generic decider
    consults this as its top-priority FACT term; a larger magnitude is a worse
    proven loss.

    This single function subsumes both the old ``material_safety`` penalties:
    the base-rate ``material_safety_prior_penalty_for`` (its per-objection
    magnitudes) and the argmax ``material_safety_selection_penalty`` (the
    combined-collapse case). Expressing the loss as an ordered FACT magnitude
    — rather than a base-rate nudge plus a flat argmax penalty — is the
    design-D2 reframing: an honest FACT decision term, not a smuggled one.
    """
    loss = max(
        (
            _objection_prior_loss(probe, evidence)
            for evidence in probe.objection_evidence
            if isinstance(evidence, ObjectionEvidence)
        ),
        default=0,
    )

    # The combined material collapse — a search-refuted en-pris piece AND a
    # search-refuted ignored hanging piece — is qualitatively the worst case
    # (the old ``material_safety_selection_penalty`` flat demotion). Rank it
    # above every finite scaled magnitude.
    if (
        has_search_refutation_at_most(
            probe.objection_evidence, COMBINED_FACT_REFUTATION_THRESHOLD
        )
        and _has_moved_piece_en_pris_objection(probe)
        and _has_ignored_hanging_piece_objection(probe)
    ):
        loss = max(loss, COMBINED_MATERIAL_COLLAPSE_LOSS)
    return loss
