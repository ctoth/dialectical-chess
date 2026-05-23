"""Phase 2 thesis tests — failing specs for the opinion-valued argumentation
decision (chunk P2.2).

These tests encode the **locked design v2** — `reports/argdriven-phase2-design-v2.md`
— as executable specifications. They are written FIRST and fail NOW: the Phase-2
modules they target (`dialectical_chess.opinion_graph`,
`dialectical_chess.skeptical_filter`, `dialectical_chess.decide`,
`dialectical_chess.static_prior`) and the `doxa` package do not exist / are not
yet a dependency. The whole module therefore raises `ModuleNotFoundError` at
import time. That RED state is the spec — the Phase-2 coder (chunks P2.4-P2.7)
makes it green by building exactly what design v2 specifies, and these tests
must not be relaxed to do so.

Design-v2 anchors (section -> property under test):
  * §1a  aggregation — one leaf per move per role, strengths summed (Codex C1).
  * §1b  the computed leaf-opinion table (k -> E).
  * §1d  k=0 evidence omitted entirely; no double-count (Codex C2).
  * §1e  defeaters as residual suppression — restoration, never boost (Codex M1).
  * §2   move node intrinsic = Opinion.vacuous(tau); unargued move E == tau.
  * §5   the Dung skeptical hard-filter — kept and mandatory.
  * §5c  forced-mate refutations are non-counterdefeatable (Codex C4).
  * §5d  empty-survivor fallback exposes `empty_survivors` (Codex Minor 3).
  * §6   the decision rule — argmax expectation(), no probe.score term.
  * §7   the single artifact builder, MoveArgumentationArtifacts (Codex C3).

Every numeric value asserted here is taken verbatim from design v2's worked,
computed examples against the real `doxa` package.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

# Core Phase 3 chunk B-E intermediate: this module asserts the chess-local
# Phase-2 surface (`dialectical_chess.opinion_graph`,
# `dialectical_chess.skeptical_filter`, `dialectical_chess.argumentation_cartridge`)
# that chunk B deleted. The tests get rewritten in chunk F against the
# `dialectical_games` core. Skip module collection until then.
pytest.skip(
    "core phase 3 in-flight: chess Phase-2 surface deleted; "
    "thesis tests rewritten against dialectical-games core in chunk F",
    allow_module_level=True,
)

# --- Phase-2 target modules — ABSENT until P2.4-P2.7. The import below is the
#     RED trigger: until `doxa` is pinned as a dependency (P2.5) and the
#     opinion-valued modules are written, this raises ModuleNotFoundError and
#     every test in this file is collected as an error. That is correct. ---
from doxa import Opinion  # noqa: E402
from doxa.argumentation import BipolarOpinionGraph, evaluate  # noqa: E402
from argumentation.dung import ArgumentationFramework, grounded_extension  # noqa: E402

from dialectical_chess.opinion_graph import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    A_ROLE,
    EV,
    BipolarMoveGraph,
    MoveArgumentationArtifacts,
    leaf_intrinsic,
)
from dialectical_chess.skeptical_filter import skeptical_survivors  # noqa: E402  # pyright: ignore[reportMissingImports]
from dialectical_chess.argumentation_cartridge import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    ArgumentationDecision,
    build_argumentation_artifacts,
    choose_move_argumentation,
)
from dialectical_chess.static_prior import (  # noqa: E402
    TAU_CLAMP,
    TAU_SCALE,
    squash,
    static_prior,
)

# --- `dialectical_chess` modules that already exist — importable today. ---
from dialectical_chess.arguments import MoveProbe  # noqa: E402
from dialectical_chess.evidence import (  # noqa: E402
    EvidenceWorld,
    ObjectionEvidence,
    ObjectionKind,
    ReplyEvidence,
    SupportKind,
    objection_evidence,
    reply_evidence,
    support_evidence,
    is_forced_mate_refutation,
)
from dialectical_chess.probe import owned_board_from_fen, probe_moves  # noqa: E402


# --------------------------------------------------------------------------
# Helpers — synthetic MoveProbe construction.
#
# MoveProbe derives `reason_evidence` / `objection_evidence` /
# `reply_attack_evidence` from the label tuples in __post_init__, so a probe is
# fully specified by its label lists. `score` is set to a sentinel value the
# decider must NOT read (§6 / §8c — the no-probe.score rule).
# --------------------------------------------------------------------------

SCORE_SENTINEL = 999_999  # design v6: probe.score appears NOWHERE in the rule.


def make_probe(
    uci: str,
    *,
    reasons: tuple[str, ...] = (),
    objections: tuple[str, ...] = (),
    reply_attacks: tuple[str, ...] = (),
    score: int = SCORE_SENTINEL,
    is_checkmate: bool = False,
    gives_check: bool = False,
    is_capture: bool = False,
    captured_value: int = 0,
    promotion_value: int = 0,
) -> MoveProbe:
    """Build a synthetic MoveProbe from explicit label lists."""
    return MoveProbe(
        uci=uci,
        san=uci,
        score=score,
        is_checkmate=is_checkmate,
        gives_check=gives_check,
        is_capture=is_capture,
        captured_value=captured_value,
        promotion_value=promotion_value,
        reasons=reasons,
        objections=objections,
        reply_attacks=reply_attacks,
        reason_evidence=tuple(synthetic_reason_evidence(label) for label in reasons),
        objection_evidence=tuple(synthetic_objection_evidence(label) for label in objections),
        reply_attack_evidence=tuple(synthetic_reply_evidence(label) for label in reply_attacks),
    )


def synthetic_reason_evidence(label: str):
    if label.startswith("development:"):
        return support_evidence(
            label,
            world=EvidenceWorld.POSITIONAL,
            counts_as_positional=True,
            argument_value="positional",
            support_strength=1,
            support_kind=SupportKind.DEVELOPMENT,
        )
    if label.startswith("center_control:") or label.startswith("piece_activity:"):
        return support_evidence(
            label,
            world=EvidenceWorld.POSITIONAL,
            counts_as_positional=True,
            argument_value="positional",
            support_strength=1,
        )
    if label.startswith("material:capture:900"):
        return support_evidence(label, world=EvidenceWorld.MATERIAL, counts_as_tactical=True, argument_value="tactical", support_strength=9)
    if label.startswith("material:capture:500"):
        return support_evidence(label, world=EvidenceWorld.MATERIAL, counts_as_tactical=True, argument_value="tactical", support_strength=9)
    if label == "tactical:check":
        return support_evidence(label, world=EvidenceWorld.TACTICAL, counts_as_tactical=True, argument_value="tactical", support_strength=7)
    if label.startswith("tactical:threat:"):
        return support_evidence(
            label,
            world=EvidenceWorld.TACTICAL,
            counts_as_tactical=True,
            argument_value="tactical",
            support_strength=6,
            tactical_threat_value=900,
        )
    return support_evidence(label, world=EvidenceWorld.PROCEDURAL)


def synthetic_objection_evidence(label: str):
    if label == "objection:no_immediate_tactical_warrant":
        return objection_evidence(
            label,
            world=EvidenceWorld.PROCEDURAL,
            objection_kind=ObjectionKind.NO_IMMEDIATE_TACTICAL_WARRANT,
            objection_strength=0,
        )
    if label.startswith("safety:moved_piece_en_pris:"):
        value = int(label.rsplit(":", 1)[1])
        return objection_evidence(
            label,
            world=EvidenceWorld.MATERIAL,
            objection_kind=ObjectionKind.MOVED_PIECE_EN_PRIS,
            objection_strength=0 if value < 300 else 97 if value >= 900 else 17,
            moved_piece_en_pris_value=value,
            argument_value="material_safety",
        )
    if label.startswith("safety:queen_blunder:"):
        return objection_evidence(
            label,
            world=EvidenceWorld.MATERIAL,
            objection_kind=ObjectionKind.QUEEN_BLUNDER,
            objection_strength=2,
            argument_value="material_safety",
        )
    if label.startswith("king_safety:queen_flank_invasion:"):
        return objection_evidence(
            label,
            world=EvidenceWorld.POSITIONAL,
            objection_kind=ObjectionKind.QUEEN_FLANK_INVASION,
            objection_strength=9,
        )
    if label.startswith("tactical:allows_reply_mate_in_one:"):
        return objection_evidence(
            label,
            world=EvidenceWorld.TACTICAL,
            objection_kind=ObjectionKind.REPLY_MATE_IN_ONE,
            objection_strength=6,
            forced_mate_distance=1,
            argument_value="reply_refutation",
        )
    if label.startswith("tactical:allows_reply_forced_mate_in_"):
        depth = int(label.removeprefix("tactical:allows_reply_forced_mate_in_").split(":", 1)[0])
        return objection_evidence(
            label,
            world=EvidenceWorld.TACTICAL,
            objection_kind=ObjectionKind.REPLY_FORCED_MATE,
            objection_strength=6 if depth == 2 else 3,
            forced_mate_distance=depth,
            argument_value="reply_refutation",
        )
    return objection_evidence(
        label,
        world=EvidenceWorld.UNKNOWN,
        objection_kind=ObjectionKind.NO_IMMEDIATE_TACTICAL_WARRANT,
        objection_strength=0,
    )


def synthetic_reply_evidence(label: str):
    if label.startswith("reply_mate:undefended:"):
        return reply_evidence(label, reply_attack_strength=7, forced_mate_distance=1)
    if label.startswith("reply_mate:defended:"):
        return reply_evidence(label, reply_attack_strength=7, defense_strength=13)
    return reply_evidence(label, reply_attack_strength=1)


def expectation_of(decision: ArgumentationDecision, uci: str) -> float:
    """The resolved move-node expectation() the decider exposes for `uci`."""
    return decision.move_opinion[uci].expectation()


# A position where `c2c4` walks into a reply mate-in-one and `f2f3` does not —
# the canonical skeptical-filter case (existing engine tests use this FEN).
QUEEN_GRAB_INTO_MATE_FEN = "6nr/n4pp1/k6p/8/3p4/1P6/1PPP1PPP/r1B3K1 w - - 0 22"


# ==========================================================================
# §1a / §1b — the leaf-opinion encoding and per-role aggregation (Codex C1)
# ==========================================================================


@pytest.mark.unit
def test_leaf_intrinsic_constants_match_design_v2() -> None:
    """§1f — the encoding is a two-constant family: EV=2.0, A_ROLE=0.5."""
    assert EV == 2.0
    assert A_ROLE == 0.5


@pytest.mark.unit
@pytest.mark.parametrize(
    ("k", "b", "u", "expectation"),
    [
        (1, 0.500000, 0.500000, 0.750000),
        (2, 0.666667, 0.333333, 0.833333),
        (3, 0.750000, 0.250000, 0.875000),
        (4, 0.800000, 0.200000, 0.900000),
        (9, 0.900000, 0.100000, 0.950000),
        (17, 0.944444, 0.055556, 0.972222),
        (33, 0.970588, 0.029412, 0.985294),
        (97, 0.989796, 0.010204, 0.994898),
    ],
)
def test_leaf_intrinsic_table(k: int, b: float, u: float, expectation: float) -> None:
    """§1b — `leaf_intrinsic(k)` reproduces the design-v2 computed table.

    Closed form (s=0, W=2): b = k*EV/(k*EV+2), d = 0, u = 2/(k*EV+2).
    """
    leaf = leaf_intrinsic(k)
    assert leaf.b == pytest.approx(b, abs=1e-6)
    assert leaf.d == pytest.approx(0.0, abs=1e-9)
    assert leaf.u == pytest.approx(u, abs=1e-6)
    assert leaf.a == pytest.approx(A_ROLE, abs=1e-9)
    assert leaf.expectation() == pytest.approx(expectation, abs=1e-6)


@pytest.mark.unit
def test_leaf_intrinsic_rejects_zero_strength() -> None:
    """§1d — there is no vacuous-leaf branch; strength must be > 0.

    A zero-strength aggregate must be omitted by the builder before
    `leaf_intrinsic` is ever called (Codex C2). The helper raises.
    """
    with pytest.raises(ValueError):
        leaf_intrinsic(0)
    with pytest.raises(ValueError):
        leaf_intrinsic(-3)


@pytest.mark.property
def test_aggregation_two_reasons_beat_one() -> None:
    """§1a / Codex C1 — two same-band support reasons resolve STRICTLY higher
    than one. The CCF-idempotence collapse (1/2/4 reasons all flat 0.775) is
    defeated by per-role strength summation.

    Design v2 §1a worked example (tau ~ 0.55, one support edge): one k=1
    reason -> E 0.775; two k=1 reasons -> E 0.850.
    """
    one_reason = make_probe(
        "a2a3", reasons=("development:a2a3:center_pawn",)
    )
    two_reasons = make_probe(
        "b2b3",
        reasons=(
            "development:b2b3:center_pawn",
            "center_control:b2b3:1",
        ),
    )
    decision = choose_move_argumentation([one_reason, two_reasons])
    e_one = expectation_of(decision, "a2a3")
    e_two = expectation_of(decision, "b2b3")
    assert e_two > e_one
    # The two-reason move is the one chosen.
    assert decision.selected.uci == "b2b3"


@pytest.mark.property
def test_aggregation_is_single_leaf_per_role() -> None:
    """§1a / §7a — the builder emits AT MOST one support leaf and one objection
    leaf per move; the individual reasons are summed into it, not turned into
    one leaf each.
    """
    probe = make_probe(
        "c2c3",
        reasons=(
            "development:c2c3:center_pawn",
            "center_control:c2c3:1",
            "piece_activity:c2c3:mobility_gain:2",
        ),
    )
    artifacts = build_argumentation_artifacts([probe])
    graph = artifacts.graph.graph
    support_leaves = {
        arg for arg in graph.arguments if arg.startswith("support:")
    }
    # Exactly one aggregate support leaf for the single move, despite 3 reasons.
    assert support_leaves == {"support:c2c3"}
    support_edges = {edge for edge in graph.supports if edge[1] == "move:c2c3"}
    assert support_edges == {("support:c2c3", "move:c2c3")}
    # All three reasons are retained in the trace for explainability (§1a).
    traced = artifacts.evidence_trace["support:c2c3"]
    assert len(traced) == 3


# ==========================================================================
# §2 — the move node: vacuity. An unargued move resolves to E == tau.
# ==========================================================================


@pytest.mark.property
def test_unargued_move_resolves_to_tau() -> None:
    """§2 / checklist 15 — a move with no reasons and no objections has a
    vacuous intrinsic Opinion.vacuous(tau); `evaluate` resolves it to
    expectation() == tau exactly.
    """
    probe = make_probe("d2d3")
    artifacts = build_argumentation_artifacts([probe])
    move_arg = artifacts.move_arg["d2d3"]
    intrinsic = artifacts.graph.graph.intrinsic[move_arg]
    # Move node intrinsic is exactly vacuous: (0, 0, 1, tau).
    assert intrinsic.b == pytest.approx(0.0, abs=1e-9)
    assert intrinsic.d == pytest.approx(0.0, abs=1e-9)
    assert intrinsic.u == pytest.approx(1.0, abs=1e-9)
    tau = intrinsic.a
    resolved = evaluate(artifacts.graph.graph)[move_arg]
    assert resolved.expectation() == pytest.approx(tau, abs=1e-9)


# ==========================================================================
# §1d / §8e — no double-count: removing a reason removes its only effect.
# ==========================================================================


@pytest.mark.property
def test_no_double_count_removing_a_reason_removes_its_only_effect() -> None:
    """§1d / Codex C2 — a support reason's effect on the decision flows
    through exactly one channel (its aggregate leaf). Removing the reason
    removes that effect and nothing else; the rest of the decision is
    unchanged.
    """
    with_reason = make_probe(
        "e2e4", reasons=("development:e2e4:center_pawn",)
    )
    without_reason = make_probe("e2e4")
    other = make_probe("g1f3")

    e_with = expectation_of(
        choose_move_argumentation([with_reason, other]), "e2e4"
    )
    e_without = expectation_of(
        choose_move_argumentation([without_reason, other]), "e2e4"
    )
    # The supporting reason raised E. Removing it lowers E back.
    assert e_with > e_without
    # The unrelated move is untouched by mutating e2e4's reasons.
    e_other_a = expectation_of(
        choose_move_argumentation([with_reason, other]), "g1f3"
    )
    e_other_b = expectation_of(
        choose_move_argumentation([without_reason, other]), "g1f3"
    )
    assert e_other_a == pytest.approx(e_other_b, abs=1e-12)


@pytest.mark.property
def test_zero_strength_evidence_never_reaches_the_graph() -> None:
    """§1d / Codex C2 — a zero-strength evidence item produces NO leaf and NO
    edge. The corrupting `k=0`-vacuous-support leaf of the original design
    (which shifted a mixed-conflict result by +0.1575) is never built.

    `objection:no_immediate_tactical_warrant` carries objection_strength 0;
    it must not yield an objection leaf.
    """
    probe = make_probe(
        "h2h3", objections=("objection:no_immediate_tactical_warrant",)
    )
    artifacts = build_argumentation_artifacts([probe])
    graph = artifacts.graph.graph
    objection_leaves = {
        arg for arg in graph.arguments if arg.startswith("objection:")
    }
    assert objection_leaves == set()
    # No attack edge for the zero-strength objection.
    assert all(edge[1] != "move:h2h3" for edge in graph.attacks)
    # The move resolves exactly as an unargued move (E == tau).
    move_arg = artifacts.move_arg["h2h3"]
    tau = graph.intrinsic[move_arg].a
    assert evaluate(graph)[move_arg].expectation() == pytest.approx(tau, abs=1e-9)


# ==========================================================================
# §8 — static_prior: disjoint from probe.score and from evidence labels.
# ==========================================================================


@pytest.mark.unit
def test_squash_centered_and_monotone() -> None:
    """§8a — squash(0) == 0.5 exactly; squash is strictly increasing."""
    assert squash(0.0) == pytest.approx(0.5, abs=1e-12)
    assert squash(100.0) == pytest.approx(0.622459, abs=1e-6)
    assert squash(300.0) == pytest.approx(0.817574, abs=1e-6)
    assert squash(400.0) == pytest.approx(0.880797, abs=1e-6)
    assert squash(-400.0) == pytest.approx(0.119203, abs=1e-6)
    values = [squash(p) for p in (-800, -200, 0, 200, 800)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


@pytest.mark.property
def test_squash_stays_strictly_inside_the_open_interval() -> None:
    """§8a — squash output is strictly inside (0.01, 0.99) so
    Opinion.vacuous(squash(x)) never raises 'a not in (0, 1)'.
    """
    lo, hi = TAU_CLAMP
    assert (lo, hi) == (0.01, 0.99)
    for prior in (-1e6, -1e3, -1.0, 0.0, 1.0, 1e3, 1e6):
        tau = squash(prior)
        assert lo <= tau <= hi
        # The clamped tau is always a legal Opinion base rate.
        Opinion.vacuous(tau)


@pytest.mark.property
def test_static_prior_ignores_probe_score() -> None:
    """§8c — two probes with the same post-move board but different
    `probe.score` produce the SAME `static_prior`. `probe.score` is an
    accumulated mix of node-represented facts and must not feed `tau`.
    """
    board = owned_board_from_fen(QUEEN_GRAB_INTO_MATE_FEN)
    probes = probe_moves(board, search_depth=0, smt_fork=False)
    base = probes[0]
    mutated = replace(base, score=base.score + 50_000)
    assert static_prior(mutated) == pytest.approx(static_prior(base), abs=1e-9)


@pytest.mark.property
def test_static_prior_ignores_evidence_labels() -> None:
    """§8c — `static_prior` is invariant under mutating the typed-evidence
    label tuples while the board is held fixed. The prior reads board state,
    never `probe.reasons` / `probe.objections` / `probe.reply_attacks`.
    """
    board = owned_board_from_fen(QUEEN_GRAB_INTO_MATE_FEN)
    probes = probe_moves(board, search_depth=0, smt_fork=False)
    base = probes[0]
    mutated = replace(
        base,
        reasons=base.reasons + ("development:zz9z9z:center_pawn",),
        objections=base.objections + ("safety:queen_blunder:zz9z9z:580",),
    )
    assert static_prior(mutated) == pytest.approx(static_prior(base), abs=1e-9)


# ==========================================================================
# §1a / §6 — aggregation test against design-v2's exact computed E values.
# ==========================================================================


@pytest.mark.property
def test_aggregation_resolved_expectations_match_design_v2() -> None:
    """§1a — design v2's computed proof: against a vacuous move node at
    tau=0.55 with a single full-trust support edge, one k=1 support leaf
    resolves to E 0.775 and two k=1 leaves (k=2 aggregate) to E 0.850.

    The static prior of the position below must put `tau` at 0.55 for the
    numbers to land exactly; the test pins the *gap* (0.850 - 0.775 = 0.075)
    and the strict ordering, which hold whatever the precise tau, and pins
    the absolute values when tau == 0.55.
    """
    one = make_probe("a2a3", reasons=("development:a2a3:center_pawn",))
    two = make_probe(
        "a2a4",
        reasons=(
            "development:a2a4:center_pawn",
            "center_control:a2a4:1",
        ),
    )
    decision = choose_move_argumentation([one, two])
    e_one = expectation_of(decision, "a2a3")
    e_two = expectation_of(decision, "a2a4")
    # Strictly monotone in aggregate strength (the C1 fix).
    assert e_two > e_one
    tau_one = decision.move_opinion["a2a3"].a
    if tau_one == pytest.approx(0.55, abs=1e-6):
        assert e_one == pytest.approx(0.775000, abs=1e-6)
        assert e_two == pytest.approx(0.850000, abs=1e-6)


# ==========================================================================
# §5a — uncertainty: a contested move resolves to higher u than an
#       unargued move (CCF routes support/attack conflict into uncertainty).
# ==========================================================================


@pytest.mark.property
def test_contested_move_has_higher_uncertainty_than_unargued_move() -> None:
    """§5a — a move with both a supporter and an objection resolves to a
    strictly higher uncertainty `u` than an unargued move. CCF routes
    total support/attack conflict into uncertainty, not disbelief.
    """
    contested = make_probe(
        "f1c4",
        reasons=("development:f1c4:minor_piece",),
        objections=("safety:moved_piece_en_pris:300",),
    )
    unargued = make_probe("g1f3")
    decision = choose_move_argumentation([contested, unargued])
    u_contested = decision.move_opinion["f1c4"].u
    u_unargued = decision.move_opinion["g1f3"].u
    # An unargued move has a vacuous resolution: u == 1.0.
    assert u_unargued == pytest.approx(1.0, abs=1e-9)
    # The contested move's support+objection drive u strictly below 1.0 but
    # CCF keeps it high — the conflict lands in uncertainty.
    assert u_contested < u_unargued
    assert u_contested > 0.0


@pytest.mark.property
def test_hanging_piece_objections_scale_with_material_cost() -> None:
    moved_minor = synthetic_objection_evidence("safety:moved_piece_en_pris:330")
    ignored_minor = objection_evidence(
        "safety:ignored_hanging_piece:f2f4:b3:330",
        world=EvidenceWorld.MATERIAL,
        objection_kind=ObjectionKind.IGNORED_HANGING_PIECE,
        objection_strength=17,
        argument_value="material_safety",
    )
    moved_pawn = synthetic_objection_evidence("safety:moved_piece_en_pris:100")
    moved_queen = synthetic_objection_evidence("safety:moved_piece_en_pris:900")
    queen_flank = synthetic_objection_evidence("king_safety:queen_flank_invasion:f7f5:g7")

    assert moved_minor.objection_strength == 17
    assert ignored_minor.objection_strength == 17
    assert moved_pawn.objection_strength == 0
    assert moved_queen.objection_strength == 97
    assert queen_flank.objection_strength == 9


# ==========================================================================
# §6 / discrimination — the argumentation decider provably diverges from a
#       plain probe.score sort.
# ==========================================================================


@pytest.mark.differential
def test_decider_diverges_from_probe_score_sort() -> None:
    """§6 — the decision rule contains NO `probe.score` term. A position
    constructed so the `probe.score` argmax and the argumentation argmax
    disagree must be decided by the argumentation rule.

    `loud` carries the largest `probe.score` but only a single weak reason;
    `quiet` carries a smaller `probe.score` but a strong aggregate of
    high-band support. The score-sort would pick `loud`; the argumentation
    decider must pick `quiet`.
    """
    loud = make_probe(
        "a1a2",
        reasons=("development:a1a2:center_pawn",),
        score=10_000,
    )
    quiet = make_probe(
        "h1h2",
        reasons=(
            "material:capture:900",
            "material:capture:500",
        ),
        score=10,
    )
    # The plain probe.score sort would rank `loud` first.
    by_score = sorted([loud, quiet], key=lambda p: (-p.score, p.uci))
    assert by_score[0].uci == "a1a2"
    # The argumentation decider ranks the strongly-supported move first.
    decision = choose_move_argumentation([loud, quiet])
    assert decision.selected.uci == "h1h2"
    assert expectation_of(decision, "h1h2") > expectation_of(decision, "a1a2")


# ==========================================================================
# §5 / §5b — the skeptical filter: a forced-mate refutation removes the move.
# ==========================================================================


@pytest.mark.differential
def test_skeptical_filter_removes_forced_mate_refuted_move() -> None:
    """§5 / §5b — the queen-grab-into-mate case. On a real position, a move
    with an undefended forced-mate refutation is excluded from the survivor
    pool and is NOT chosen, even when its expectation() is high.

    Position: `c2c4` walks into `tactical:allows_reply_mate_in_one:c2c4:a1c1`.
    """
    board = owned_board_from_fen(QUEEN_GRAB_INTO_MATE_FEN)
    probes = probe_moves(board, search_depth=0, smt_fork=False)
    by_uci = {p.uci: p for p in probes}
    # Precondition: c2c4 really does carry a reply-mate-in-one objection.
    assert any(
        obj.startswith("tactical:allows_reply_mate_in_one:c2c4")
        for obj in by_uci["c2c4"].objections
    )

    artifacts = build_argumentation_artifacts(probes)
    survivors = skeptical_survivors(artifacts)
    # The hard-refuted move is not a survivor.
    assert "c2c4" not in survivors
    # The decider never selects it.
    decision = choose_move_argumentation(probes)
    assert decision.selected.uci != "c2c4"
    assert decision.empty_survivors is False


@pytest.mark.differential
def test_undefended_reply_mate_from_reply_attack_analyzer_is_hard_filtered() -> None:
    """§5b / M-1 — an undefended reply mate found only by the reply-attack
    analyzer is a hard refutation, not a soft objection.

    The candidate has no `tactical:allows_reply_mate_in_one:` objection. Its
    only mate fact is `reply_mate:undefended:...`, the label family emitted by
    bounded reply-attack analysis at depths where the reply-mate objection scan
    can be skipped. The filter must exclude it before expectation ranking.
    """
    into_mate = make_probe(
        "d1d8",
        reasons=("material:capture:900", "material:capture:500"),
        reply_attacks=("reply_mate:undefended:a1a2",),
    )
    sound = make_probe("g1f3")

    assert into_mate.objection_evidence == ()
    assert [ev.label for ev in into_mate.reply_attack_evidence] == [
        "reply_mate:undefended:a1a2"
    ]

    artifacts = build_argumentation_artifacts([into_mate, sound])
    survivors = skeptical_survivors(artifacts)
    assert "d1d8" not in survivors
    assert "g1f3" in survivors

    decision = choose_move_argumentation([into_mate, sound])
    assert decision.selected.uci != "d1d8"
    assert decision.empty_survivors is False


@pytest.mark.differential
def test_defended_reply_mate_is_suppressed_not_hard_filtered() -> None:
    """m-2 — `reply_mate:defended:` is answered by its defense flag.

    A defended reply mate is not a filter refutation, and its reply-attack
    strength is fully suppressed before an objection leaf is built.
    """
    defended = make_probe(
        "d1d8",
        reply_attacks=("reply_mate:defended:a1a2",),
    )
    evidence = defended.reply_attack_evidence[0]
    assert isinstance(evidence, ReplyEvidence)
    assert evidence.reply_attack_strength == 7
    assert evidence.defense_strength == 13
    assert not is_forced_mate_refutation(evidence)

    artifacts = build_argumentation_artifacts([defended])
    assert skeptical_survivors(artifacts) == {"d1d8"}
    assert not artifacts.filter_af.defeats
    assert all(not arg.startswith("objection:") for arg in artifacts.graph.graph.arguments)


@pytest.mark.differential
def test_filter_graph_contains_only_refute_to_move_edges() -> None:
    """§5b — `filter_af` is a pure-attack Dung framework: every defeat pair
    has a `refute:` attacker and a `move:` target. The forced-mate
    refutation node attacks the move and nothing else.
    """
    board = owned_board_from_fen(QUEEN_GRAB_INTO_MATE_FEN)
    probes = probe_moves(board, search_depth=0, smt_fork=False)
    artifacts = build_argumentation_artifacts(probes)
    filter_af = artifacts.filter_af
    assert isinstance(filter_af, ArgumentationFramework)
    assert filter_af.defeats  # the position has at least one refutation.
    for attacker, target in filter_af.defeats:
        assert attacker.startswith("refute:")
        assert target.startswith("move:")


# ==========================================================================
# §5c — counterdefeater policy (Codex C4): forced-mate refutations are
#       non-counterdefeatable — the filter graph has NO counterdefeater edges.
# ==========================================================================


@pytest.mark.differential
def test_filter_graph_has_no_counterdefeater_edges() -> None:
    """§5c / Codex C4 — a forced-mate refutation is a terminal fact and is
    never counter-defeated in Phase 2. The filter `ArgumentationFramework`
    contains no pair whose TARGET is a `refute:` node.
    """
    board = owned_board_from_fen(QUEEN_GRAB_INTO_MATE_FEN)
    probes = probe_moves(board, search_depth=0, smt_fork=False)
    artifacts = build_argumentation_artifacts(probes)
    for _attacker, target in artifacts.filter_af.defeats:
        assert not target.startswith("refute:")
    # The refutation node is therefore always in the grounded extension.
    grounded = grounded_extension(artifacts.filter_af)
    refute_nodes = {
        arg for arg in artifacts.filter_af.arguments if arg.startswith("refute:")
    }
    assert refute_nodes <= grounded


@pytest.mark.unit
def test_forced_mate_refutation_predicate_is_the_filter_membership_rule() -> None:
    """§5b — filter membership is exactly `is_forced_mate_refutation`. A
    reply-mate-in-one objection qualifies; a soft positional objection does
    not.
    """
    board = owned_board_from_fen(QUEEN_GRAB_INTO_MATE_FEN)
    probes = probe_moves(board, search_depth=0, smt_fork=False)
    by_uci = {p.uci: p for p in probes}
    mate_objections = [
        ev
        for ev in by_uci["c2c4"].objection_evidence
        if isinstance(ev, ObjectionEvidence)
        if ev.objection_kind == ObjectionKind.REPLY_MATE_IN_ONE
    ]
    assert mate_objections
    assert all(is_forced_mate_refutation(ev) for ev in mate_objections)


# ==========================================================================
# §1e — defeater-as-suppression (Codex M1): a defeated objection restores the
#       move to its no-objection baseline, never above it.
# ==========================================================================


@pytest.mark.property
def test_defeated_objection_restores_to_baseline_never_above() -> None:
    """§1e / Codex M1 — a move with one objection PLUS a full defeater of
    that objection resolves to exactly its no-objection baseline. A defeater
    suppresses the objection's strength at aggregation time; it is not a
    graph node and never converts a (defeated) attack into positive belief.

    `contested_then_defeated` carries a `queen_blunder` objection AND a
    compensating forcing pressure (a >=700 tactical threat reason plus a
    check) that synthesizes a COMPENSATING_FORCING_PRESSURE defeater. The
    defeater fully cancels the objection (defeater strength 33 >> objection
    strength 2), so the move returns to its no-objection result.
    """
    baseline = make_probe(
        "d1h5",
        reasons=("tactical:threat:targets:1:value:900",),
        gives_check=True,
    )
    contested_then_defeated = make_probe(
        "d1h5",
        reasons=("tactical:threat:targets:1:value:900",),
        objections=("safety:queen_blunder:d1h5:580",),
        gives_check=True,
    )
    e_baseline = expectation_of(
        choose_move_argumentation([baseline]), "d1h5"
    )
    e_defeated = expectation_of(
        choose_move_argumentation([contested_then_defeated]), "d1h5"
    )
    # Restoration, not boost: the defeated objection lands EXACTLY on the
    # no-objection baseline and never above it.
    assert e_defeated == pytest.approx(e_baseline, abs=1e-9)
    assert e_defeated <= e_baseline + 1e-9


@pytest.mark.property
def test_fully_defeated_objection_yields_no_objection_leaf() -> None:
    """§1d / §1e — a fully-defeated objection has effective strength 0 and,
    by the C2 omission rule, produces no objection leaf and no attack edge.
    """
    probe = make_probe(
        "d1h5",
        reasons=("tactical:threat:targets:1:value:900",),
        objections=("safety:queen_blunder:d1h5:580",),
        gives_check=True,
    )
    artifacts = build_argumentation_artifacts([probe])
    graph = artifacts.graph.graph
    objection_leaves = {
        arg for arg in graph.arguments if arg.startswith("objection:")
    }
    assert objection_leaves == set()
    assert all(edge[1] != "move:d1h5" for edge in graph.attacks)


# ==========================================================================
# §5d — empty-survivor fallback (Codex Minor 3).
# ==========================================================================


@pytest.mark.property
def test_empty_survivors_when_every_move_hard_refuted() -> None:
    """§5d — a position where every legal move carries a forced-mate
    refutation. The grounded extension excludes every move; the decider sets
    `empty_survivors=True`, falls back to ranking all moves, and still
    returns one.
    """
    refuted_a = make_probe(
        "a1a2",
        objections=("tactical:allows_reply_mate_in_one:a1a2:h8h1",),
    )
    refuted_b = make_probe(
        "b1b2",
        objections=("tactical:allows_reply_mate_in_one:b1b2:h8h1",),
    )
    artifacts = build_argumentation_artifacts([refuted_a, refuted_b])
    survivors = skeptical_survivors(artifacts)
    assert survivors == set()

    decision = choose_move_argumentation([refuted_a, refuted_b])
    assert decision.empty_survivors is True
    # A lost position still yields a least-bad move — one of the inputs.
    assert decision.selected.uci in {"a1a2", "b1b2"}


@pytest.mark.property
def test_empty_survivors_prefer_slowest_proven_loss() -> None:
    fast_loss = make_probe(
        "a1a2",
        reasons=("tactical:check",),
        objections=("tactical:allows_reply_mate_in_one:a1a2:h8h1",),
    )
    slow_loss = make_probe(
        "b1b2",
        objections=("tactical:allows_reply_forced_mate_in_3:b1b2",),
    )

    decision = choose_move_argumentation([fast_loss, slow_loss])

    assert decision.empty_survivors is True
    assert decision.selected.uci == "b1b2"


@pytest.mark.property
def test_over_filtered_position_detected() -> None:
    """§5d — a soft reply objection must not be silently hard-filtered.

    This guard fails if a future predicate widening excludes a move that is not
    proved lost by `is_forced_mate_refutation`.
    """
    soft_reply_objection = make_probe(
        "a1a2",
        reply_attacks=("reply_captures_moved_piece:undefended:a2a7:320",),
    )
    hard_refuted = make_probe(
        "b1b2",
        objections=("tactical:allows_reply_mate_in_one:b1b2:h8h1",),
    )

    artifacts = build_argumentation_artifacts([soft_reply_objection, hard_refuted])
    survivors = skeptical_survivors(artifacts)
    assert "a1a2" in survivors
    assert "b1b2" not in survivors
    assert all(target != "move:a1a2" for _attacker, target in artifacts.filter_af.defeats)


@pytest.mark.unit
def test_decider_rejects_empty_probe_list() -> None:
    """§7c — `choose_move_argumentation([])` raises; a position with no legal
    moves is a caller error, not an empty-survivor fallback.
    """
    with pytest.raises(ValueError):
        choose_move_argumentation([])


# ==========================================================================
# §7 — the single-artifact contract (Codex C3): one builder, one artifact.
# ==========================================================================


@pytest.mark.differential
def test_artifact_is_built_once_from_real_probes() -> None:
    """§7a / Codex C3 — `build_argumentation_artifacts` produces ONE
    `MoveArgumentationArtifacts` carrying the opinion graph, the move index,
    the filter framework, and the evidence trace, from real `probe_moves`
    output. The decider and the filter both consume this artifact.
    """
    board = owned_board_from_fen(QUEEN_GRAB_INTO_MATE_FEN)
    probes = probe_moves(board, search_depth=0, smt_fork=False)
    artifacts = build_argumentation_artifacts(probes)

    assert isinstance(artifacts, MoveArgumentationArtifacts)
    assert isinstance(artifacts.graph, BipolarMoveGraph)
    assert isinstance(artifacts.graph.graph, BipolarOpinionGraph)
    assert isinstance(artifacts.filter_af, ArgumentationFramework)

    # Every probe has a `move:{uci}` argument in the opinion graph and the
    # move index mirrors the graph index.
    for probe in probes:
        move_arg = f"move:{probe.uci}"
        assert artifacts.move_arg[probe.uci] == move_arg
        assert move_arg in artifacts.graph.graph.arguments
    assert artifacts.move_arg == artifacts.graph.move_arg

    # The opinion graph is accepted by doxa and `evaluate` resolves it.
    opinions = evaluate(artifacts.graph.graph)
    for probe in probes:
        assert artifacts.move_arg[probe.uci] in opinions

    # The evidence trace keys cover every leaf and every refute: node.
    for arg in artifacts.graph.graph.arguments:
        if arg.startswith(("support:", "objection:")):
            assert arg in artifacts.evidence_trace
    for arg in artifacts.filter_af.arguments:
        if arg.startswith("refute:"):
            assert arg in artifacts.evidence_trace


@pytest.mark.property
def test_resolved_opinion_ordering_on_a_known_position() -> None:
    """§6 — on a constructed position the resolved per-move `expectation()`
    induces a total order, and `choose_move_argumentation` selects the
    argmax. A move with a strong material support leaf outranks an unargued
    move, which outranks an objection-only move.
    """
    strong = make_probe("a1a2", reasons=("material:capture:900",))
    quiet = make_probe("b1b2")
    weak = make_probe(
        "c1c2", objections=("safety:moved_piece_en_pris:500",)
    )
    decision = choose_move_argumentation([strong, quiet, weak])
    e_strong = expectation_of(decision, "a1a2")
    e_quiet = expectation_of(decision, "b1b2")
    e_weak = expectation_of(decision, "c1c2")
    assert e_strong > e_quiet > e_weak
    assert decision.selected.uci == "a1a2"


@pytest.mark.unit
def test_tie_break_selects_lexicographically_largest_uci() -> None:
    """§6 / Codex Minor 4 — on an exact `expectation()` tie, the
    lexicographically LARGEST UCI wins. Two unargued moves with the same
    static prior resolve to identical `E`; the decider is a pure function of
    the probe set via the (expectation, uci) max key.
    """
    move_a = make_probe("a1a2")
    move_b = make_probe("b1b2")
    decision = choose_move_argumentation([move_a, move_b])
    # Both unargued -> equal expectation -> largest uci ("b1b2") wins.
    assert expectation_of(decision, "a1a2") == pytest.approx(
        expectation_of(decision, "b1b2"), abs=1e-12
    )
    assert decision.selected.uci == "b1b2"
