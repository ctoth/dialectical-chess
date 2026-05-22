"""Chess cartridge — lift chess ``MoveProbe`` data into generic move arguments.

This module is the chess-specific half of the cartridge seam. The generic
argumentation machinery (:mod:`~dialectical_chess.opinion_graph`,
:mod:`~dialectical_chess.decide`) consumes only generic
:class:`~dialectical_chess.move_argument.MoveArgument` values; this module is
the one place that turns a chess :class:`~dialectical_chess.arguments.MoveProbe`
into one. Every chess-specific input the generic layer used to reach for is
computed here and handed across as a generic typed value:

* the move-node base rate — chess's squashed
  :func:`~dialectical_chess.static_prior.static_prior`;
* each support / objection — a generic
  :class:`~dialectical_chess.move_argument.Evidence`, with chess defeater
  suppression already applied to the objection strengths
  (:mod:`~dialectical_chess.suppression`, design D3);
* the worst proven material loss — chess's
  :func:`~dialectical_chess.suppression.fact_material_loss`, lifted to a
  FACT-tier objection :class:`~dialectical_chess.move_argument.Evidence` *and*
  carried as the move argument's generic FACT magnitude (design D2);
* the empty-survivor slowest-loss distance — a forced-mate scan over the
  post-move chess board, supplied as a *lazy callback* the generic decider
  invokes only on the empty-survivor fallback path (never eagerly per probe),
  bounded by the engine's critical-clock deadline.

The chess-facing wrappers (:func:`build_argumentation_artifacts`,
:func:`choose_move_argumentation`) keep the ``MoveProbe`` call surface the rest
of the chess engine and its tests use; they convert to generic move arguments
and delegate to the generic core.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import chess
from doxa import Opinion

from dialectical_chess.arguments import MoveProbe
from dialectical_chess.evidence import (
    ArgumentEvidence,
    DefeaterEvidence,
    ObjectionEvidence,
    ReplyEvidence,
    SupportEvidence,
    forced_mate_refutation_distance,
    is_forced_mate_refutation,
)
from dialectical_chess.decide import decide
from dialectical_chess.loss_mining import has_forced_mate
from dialectical_chess.move_argument import Evidence, MoveArgument, Role
from dialectical_chess.opinion_graph import (
    MoveArgumentationArtifacts,
    build_argumentation_artifacts as build_generic_artifacts,
)
from dialectical_chess.scheme import Tier
from dialectical_chess.static_prior import squash, static_prior
from dialectical_chess.suppression import (
    fact_material_loss,
    suppressing_defeaters,
)

SLOWEST_LOSS_MAX_MATE_DEPTH = 4

# A proven material loss is FACT-tier objection evidence, but it does NOT hard
# -defeat the move in the crisp filter — it feeds the decider's FACT term. The
# label under which the lifted material-loss objection is recorded.
MATERIAL_LOSS_LABEL = "fact:material_loss"


# --- evidence lifting -------------------------------------------------------


def _support_strength(evidence: ArgumentEvidence) -> int:
    """The aggregate support strength a reason contributes, after chess rules.

    A :class:`SupportEvidence` / :class:`DefeaterEvidence` contributes its
    ``support_strength`` when it positively supports the move; everything else
    contributes nothing.
    """
    if isinstance(evidence, SupportEvidence | DefeaterEvidence):
        if evidence.supports_argument and evidence.support_strength > 0:
            return evidence.support_strength
    return 0


def _residual_objection_strength(
    probe: MoveProbe,
    objection: ArgumentEvidence,
) -> int:
    """The objection's strength after chess-policy defeater suppression.

    The chess suppression policy (design D3): an objection's strength is
    cancelled, at lifting time, by the suppression strength of every defeater
    the chess :mod:`~dialectical_chess.suppression` policy returns for it. The
    generic layer never sees the suppression — it reads only the residual
    strength carried on the lifted generic :class:`Evidence`.
    """
    strength = 0
    if isinstance(objection, ObjectionEvidence):
        strength = objection.objection_strength
    elif isinstance(objection, ReplyEvidence):
        strength = objection.reply_attack_strength
    if strength <= 0:
        return 0
    suppression = sum(
        _defeater_strength_value(defeater)
        for defeater in suppressing_defeaters(probe, objection)
    )
    return max(0, strength - suppression)


def _defeater_strength_value(evidence: ArgumentEvidence) -> int:
    """The suppression strength a defeater / defended reply carries.

    The chess suppression policy returns either a synthetic
    :class:`DefeaterEvidence` (carrying ``defeater_strength``) or — for a
    defended reply that suppresses itself — the :class:`ReplyEvidence` (whose
    ``defense_strength`` is the suppression). Both cases are read here.
    """
    if isinstance(evidence, DefeaterEvidence):
        return evidence.defeater_strength
    if isinstance(evidence, ReplyEvidence):
        return evidence.defense_strength
    return 0


def _lift_support(evidence: ArgumentEvidence) -> Evidence:
    """Lift a chess support reason into a generic SUPPORT :class:`Evidence`."""
    return Evidence(
        label=evidence.label,
        role=Role.SUPPORT,
        tier=evidence.tier,
        strength=_support_strength(evidence),
        source=evidence,
    )


def _lift_objection(probe: MoveProbe, objection: ArgumentEvidence) -> Evidence:
    """Lift a chess objection / reply attack into a generic OBJECTION evidence.

    Chess defeater suppression is applied here, so the generic ``strength`` is
    the residual. ``refutes`` is set for a forced-mate / proven-refutation
    objection — the chess crisp hard gate, lifted to the generic
    :attr:`Evidence.refutes` flag. ``magnitude`` carries the proven forced-mate
    distance when the objection encodes one.
    """
    return Evidence(
        label=objection.label,
        role=Role.OBJECTION,
        tier=objection.tier,
        strength=_residual_objection_strength(probe, objection),
        magnitude=forced_mate_refutation_distance(objection),
        refutes=is_forced_mate_refutation(objection),
        source=objection,
    )


def _material_loss_objection(probe: MoveProbe) -> Evidence | None:
    """Build the FACT-tier material-loss objection for a move, if it has one.

    Design D2: chess's ``material_safety`` is reframed as an honest FACT-tier
    objection-evidence record. When the move walks into a proven material loss
    (:func:`~dialectical_chess.suppression.fact_material_loss` is positive),
    this returns a generic FACT :class:`Evidence` whose ``magnitude`` is that
    proven loss. It is FACT-tier but does *not* ``refute`` — a material loss
    feeds the decider's FACT term, it does not crisp-filter the move (chess's
    crisp filter has always gated only on forced mate). ``None`` when the move
    walks into no proven material loss.
    """
    loss = fact_material_loss(probe)
    if loss <= 0:
        return None
    return Evidence(
        label=f"{MATERIAL_LOSS_LABEL}:{probe.uci}",
        role=Role.OBJECTION,
        tier=Tier.FACT,
        strength=0,
        magnitude=loss,
        refutes=False,
        source=None,
    )


def _slowest_loss_distance(
    probe: MoveProbe,
    *,
    deadline: float | None = None,
) -> int:
    """The proven mate distance a hard-refuted move walks into (empty-survivor).

    A larger distance is a slower loss — better, under the empty-survivor
    fallback. Reads the post-move chess board (when the probe carries one) to
    scan for a forced mate, falling back to the forced-mate distance encoded in
    the move's objection / reply-attack evidence.

    This is chess board logic and runs an unbounded recursive proof, so it is
    deliberately **lazy**: it is invoked only on the empty-survivor fallback
    path (see :func:`choose_move_argumentation`), never eagerly per probe. The
    engine's ``deadline`` is threaded into ``has_forced_mate`` so a critical
    clock caps the proof even when it does run.
    """
    if probe.post_fen is not None:
        board = chess.Board(probe.post_fen)
        for mate_depth in range(1, SLOWEST_LOSS_MAX_MATE_DEPTH + 1):
            if has_forced_mate(board, mate_depth=mate_depth, deadline=deadline):
                return mate_depth
    distances = [
        distance
        for evidence in (*probe.objection_evidence, *probe.reply_attack_evidence)
        if (distance := forced_mate_refutation_distance(evidence)) is not None
    ]
    return min(distances, default=0)


def move_argument_for(probe: MoveProbe) -> MoveArgument:
    """Lift one chess :class:`MoveProbe` into a generic :class:`MoveArgument`.

    This is the seam: every chess-specific computation — the squashed static
    prior, defeater suppression, the FACT material-loss classification —
    happens here, and the result is a purely generic :class:`MoveArgument` the
    core argumentation layer consumes.

    The empty-survivor slowest-loss distance is **not** computed here. It runs
    an unbounded forced-mate proof and is needed only on the rare
    empty-survivor fallback path, so it is supplied separately, lazily, as a
    callback to the generic decider (see :func:`choose_move_argumentation` and
    :func:`empty_survivor_loss_for`). ``empty_survivor_loss_distance`` is left
    at its default 0.
    """
    supports = tuple(_lift_support(reason) for reason in probe.reason_evidence)
    lifted_objections = [
        _lift_objection(probe, objection)
        for objection in (*probe.objection_evidence, *probe.reply_attack_evidence)
    ]
    material_loss = _material_loss_objection(probe)
    if material_loss is not None:
        lifted_objections.append(material_loss)
    return MoveArgument(
        move_id=probe.uci,
        prior=squash(static_prior(probe)),
        supports=supports,
        objections=tuple(lifted_objections),
    )


def move_arguments_for(probes: list[MoveProbe]) -> list[MoveArgument]:
    """Lift a list of chess probes into generic move arguments."""
    return [move_argument_for(probe) for probe in probes]


def empty_survivor_loss_for(
    probes: list[MoveProbe],
    *,
    deadline: float | None = None,
) -> Callable[[MoveArgument], int]:
    """Build the empty-survivor loss-distance callback for the generic decider.

    Returns a callable mapping a generic :class:`MoveArgument` back to its
    chess probe and computing :func:`_slowest_loss_distance` for it — the
    unbounded forced-mate proof. The generic decider invokes this **only** on
    the empty-survivor fallback path, and then only for the moves it ranks, so
    the proof never runs when a clean crisp survivor exists. The engine
    ``deadline`` is closed over and threaded into the proof.
    """
    probe_by_uci = {probe.uci: probe for probe in probes}

    def loss_distance(argument: MoveArgument) -> int:
        probe = probe_by_uci.get(argument.move_id)
        if probe is None:
            return 0
        return _slowest_loss_distance(probe, deadline=deadline)

    return loss_distance


# --- chess-facing decision surface ------------------------------------------


@dataclass(frozen=True)
class ArgumentationDecision:
    """The chess opinion-valued decision over one legal-move probe set.

    The chess-facing decision: :attr:`selected` is the chosen chess
    :class:`MoveProbe`. The generic core decides over
    :class:`~dialectical_chess.move_argument.MoveArgument` values; this wrapper
    maps the chosen move id back to its originating probe.
    """

    selected: MoveProbe
    empty_survivors: bool
    move_opinion: dict[str, Opinion]


def build_argumentation_artifacts(
    probes: list[MoveProbe],
) -> MoveArgumentationArtifacts:
    """Build the argumentation artifacts for a list of chess probes.

    The chess-facing wrapper: lifts the probes to generic move arguments and
    delegates to the generic graph builder.
    """
    return build_generic_artifacts(move_arguments_for(probes))


def choose_move_argumentation(
    probes: list[MoveProbe],
    *,
    deadline: float | None = None,
) -> ArgumentationDecision:
    """Return the argumentation decision for the input chess probes.

    The chess-facing wrapper around the generic decider: lifts the probes to
    generic move arguments, runs the generic core decision, and maps the
    chosen move id back to its chess probe.

    ``deadline`` is the engine's critical-clock budget; it is threaded into
    the empty-survivor slowest-loss callback so the forced-mate proof — which
    runs only on the empty-survivor fallback path — is bounded.
    """
    if not probes:
        raise ValueError("position has no legal moves")
    arguments = move_arguments_for(probes)
    decision = decide(
        arguments,
        empty_survivor_loss=empty_survivor_loss_for(probes, deadline=deadline),
    )
    probe_by_uci = {probe.uci: probe for probe in probes}
    return ArgumentationDecision(
        selected=probe_by_uci[decision.selected.move_id],
        empty_survivors=decision.empty_survivors,
        move_opinion=decision.move_opinion,
    )
