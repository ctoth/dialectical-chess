"""Phase 1 — the cut cartridge seam and the explicit ``Tier`` (core extraction).

These tests pin the Phase-1 cleanup: chess now has an explicit, typed
:class:`~dialectical_chess.scheme.Tier`, and the generic argumentation
machinery (``opinion_graph``, ``decide``) is cleanly seamed off from the
chess-specific tactics — it reads only generic typed evidence, never a chess
objection-kind name. The chess suppression / material-safety policy lives in
``dialectical_chess.suppression``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from dialectical_chess.evidence import (
    EvidenceWorld,
    ObjectionKind,
    objection_evidence,
    objection_tier,
    reply_evidence,
    support_evidence,
)
from dialectical_chess.scheme import Tier
from dialectical_chess.suppression import fact_material_loss, suppressing_defeaters

_PACKAGE = Path(__file__).resolve().parents[1] / "dialectical_chess"


# ==========================================================================
# D1 — the explicit Tier.
# ==========================================================================


@pytest.mark.unit
def test_tier_has_exactly_fact_and_heuristic() -> None:
    """D1 — ``Tier`` is the two-member FACT / HEURISTIC enum, modelled on
    dialectical-checkers' ``Tier``."""
    assert {tier.name for tier in Tier} == {"FACT", "HEURISTIC"}
    assert Tier.FACT.value == "fact"
    assert Tier.HEURISTIC.value == "heuristic"


@pytest.mark.unit
def test_forced_mate_and_search_objections_are_fact_tier() -> None:
    """D1 — a proven loss (a forced-mate / search refutation) is FACT-tier."""
    for kind in (
        ObjectionKind.SEARCH_REFUTATION,
        ObjectionKind.REPLY_MATE_IN_ONE,
        ObjectionKind.REPLY_FORCED_MATE,
    ):
        assert objection_tier(kind) is Tier.FACT


@pytest.mark.unit
def test_material_safety_objections_are_fact_tier() -> None:
    """D1 / D2 — chess's material-safety objections are honest FACT-tier
    evidence (the reframing of the smuggled ``material_safety`` penalties)."""
    for kind in (
        ObjectionKind.MOVED_PIECE_EN_PRIS,
        ObjectionKind.IGNORED_HANGING_PIECE,
        ObjectionKind.QUEEN_BLUNDER,
        ObjectionKind.QUEEN_FLANK_INVASION,
    ):
        assert objection_tier(kind) is Tier.FACT


@pytest.mark.unit
def test_positional_objections_are_heuristic_tier() -> None:
    """D1 — a positional judgement (opening play, flank-pawn structure,
    drift) is HEURISTIC-tier, not FACT."""
    for kind in (
        ObjectionKind.OPENING_KING_WALK,
        ObjectionKind.OPENING_PREMATURE_QUEEN,
        ObjectionKind.FLANK_PAWN_WEAKENING,
        ObjectionKind.FLANK_PAWN_LUNGE,
        ObjectionKind.UNSUPPORTED_MAJOR_DRIFT,
        ObjectionKind.NO_IMMEDIATE_TACTICAL_WARRANT,
    ):
        assert objection_tier(kind) is Tier.HEURISTIC


@pytest.mark.unit
def test_objection_factory_sets_tier_from_kind() -> None:
    """D1 — the ``objection_evidence`` factory tags each objection with the
    tier of its kind; the tier is carried on the typed evidence."""
    fact = objection_evidence(
        "safety:moved_piece_en_pris:900",
        world=EvidenceWorld.MATERIAL,
        objection_kind=ObjectionKind.MOVED_PIECE_EN_PRIS,
        objection_strength=97,
        moved_piece_en_pris_value=900,
    )
    heuristic = objection_evidence(
        "opening:king_walk:e1e2",
        world=EvidenceWorld.POSITIONAL,
        objection_kind=ObjectionKind.OPENING_KING_WALK,
        objection_strength=1,
    )
    assert fact.tier is Tier.FACT
    assert heuristic.tier is Tier.HEURISTIC


@pytest.mark.unit
def test_reply_evidence_tier_tracks_forced_mate_proof() -> None:
    """D1 — a reply that proves a forced mate is FACT-tier; a soft reply
    attack is a HEURISTIC judgement."""
    proven = reply_evidence(
        "reply_mate:undefended:a1a2",
        reply_attack_strength=7,
        forced_mate_distance=1,
    )
    soft = reply_evidence("reply_captures:a1a2", reply_attack_strength=1)
    assert proven.tier is Tier.FACT
    assert soft.tier is Tier.HEURISTIC


@pytest.mark.unit
def test_support_evidence_defaults_to_heuristic_tier() -> None:
    """D1 — a support reason is a positional judgement: HEURISTIC by default."""
    reason = support_evidence(
        "development:e2e4:center_pawn",
        world=EvidenceWorld.POSITIONAL,
        counts_as_positional=True,
        support_strength=1,
    )
    assert reason.tier is Tier.HEURISTIC


# ==========================================================================
# The cut seam — the generic argumentation layer imports NOTHING chess-specific.
#
# These tests verify the REAL seam, not name-spelling. ``opinion_graph`` and
# ``decide`` are the would-be ``dialectical-games`` generic core: they must
# import only generic types (the generic ``Evidence`` / ``MoveArgument`` /
# ``Role``, the generic ``Tier``, the doxa / Dung argumentation libraries) and
# they must genuinely READ ``Tier`` as the FACT/HEURISTIC discriminator. If the
# seam regresses — a ``chess`` import creeps back, a chess policy module is
# imported, the concrete chess ``MoveProbe`` is reached for, or the generic
# layer stops keying on ``Tier`` — one of these tests fails.
# ==========================================================================

# The generic core files that must be game-agnostic and extractable as-is.
_GENERIC_CORE_FILES = ("opinion_graph.py", "decide.py")

# Chess-specific module names the generic core must NEVER import. ``chess`` is
# the python-chess board library; the rest are chess cartridge policy modules.
_FORBIDDEN_IMPORT_MODULES = frozenset(
    {
        "chess",
        "chess.pgn",
        "chess.engine",
        "dialectical_chess.static_prior",
        "dialectical_chess.suppression",
        "dialectical_chess.loss_mining",
        "dialectical_chess.probe",
        "dialectical_chess.board",
        "dialectical_chess.search",
        "dialectical_chess.evidence",
        "dialectical_chess.argumentation_cartridge",
    }
)

# The concrete chess per-move carrier. The generic core consumes the generic
# ``MoveArgument`` instead; it must never import the concrete chess ``MoveProbe``.
_FORBIDDEN_IMPORT_NAMES = frozenset({"MoveProbe"})


def _imported_modules(module_path: Path) -> set[str]:
    """Every module name a source file imports (``import`` and ``from`` forms)."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _imported_names(module_path: Path) -> set[str]:
    """Every bare name a source file imports via ``from ... import name``."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
    return names


def _names_used(module_path: Path) -> set[str]:
    """Every attribute / name referenced in a module's source."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


@pytest.mark.unit
@pytest.mark.parametrize("source_file", _GENERIC_CORE_FILES)
def test_generic_core_imports_nothing_chess_specific(source_file: str) -> None:
    """The seam — the generic core (``opinion_graph``, ``decide``) imports no
    chess-specific module: not ``chess``, not ``static_prior`` / ``suppression``
    / ``loss_mining``, not any chess board / probe / evidence module. Every
    chess-specific input is computed cartridge-side and handed in as a generic
    typed value. This is the import scan that proves the seam, not a name scan.
    """
    imported = _imported_modules(_PACKAGE / source_file)
    leaked = imported & _FORBIDDEN_IMPORT_MODULES
    assert leaked == set(), (
        f"{source_file} imports chess-specific modules: {sorted(leaked)}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("source_file", _GENERIC_CORE_FILES)
def test_generic_core_does_not_import_concrete_chess_move_probe(
    source_file: str,
) -> None:
    """The seam — the generic core never imports the concrete chess
    ``MoveProbe``. It consumes the generic ``MoveArgument`` carrier instead, so
    a second game's cartridge can feed the same core unchanged."""
    imported = _imported_names(_PACKAGE / source_file)
    leaked = imported & _FORBIDDEN_IMPORT_NAMES
    assert leaked == set(), (
        f"{source_file} imports a concrete chess carrier: {sorted(leaked)}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("source_file", _GENERIC_CORE_FILES)
def test_generic_core_imports_only_generic_dialectical_modules(
    source_file: str,
) -> None:
    """The seam — every ``dialectical_chess`` module the generic core imports
    is itself generic (the scheme ``Tier``, the generic ``move_argument`` and
    ``opinion_graph`` / ``skeptical_filter`` modules, the tuning constants). No
    chess cartridge module is in the import set."""
    generic_dialectical_modules = {
        "dialectical_chess.scheme",
        "dialectical_chess.move_argument",
        "dialectical_chess.opinion_graph",
        "dialectical_chess.skeptical_filter",
        "dialectical_chess.tuning",
    }
    imported = _imported_modules(_PACKAGE / source_file)
    dialectical = {m for m in imported if m.startswith("dialectical_chess")}
    unexpected = dialectical - generic_dialectical_modules
    assert unexpected == set(), (
        f"{source_file} imports non-generic dialectical_chess modules: "
        f"{sorted(unexpected)}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("source_file", _GENERIC_CORE_FILES)
def test_generic_core_reads_tier(source_file: str) -> None:
    """The seam — the generic core genuinely READS ``Tier``: it imports the
    generic ``Tier`` enum and references it in its source. ``Tier`` is the
    FACT/HEURISTIC discriminator the crisp filter and the decider key off; a
    generic layer that never names ``Tier`` is not genuinely keyed on it."""
    imported = _imported_names(_PACKAGE / source_file)
    assert "Tier" in imported, f"{source_file} does not import the generic Tier"
    names = _names_used(_PACKAGE / source_file)
    assert "Tier" in names, f"{source_file} imports Tier but never reads it"
    assert "FACT" in names, (
        f"{source_file} never references Tier.FACT — the crisp/decider FACT key"
    )


@pytest.mark.unit
def test_generic_graph_builder_filter_keys_on_fact_tier() -> None:
    """The seam — ``opinion_graph``'s crisp filter selects FACT evidence by
    ``tier is Tier.FACT``, not by any chess-specific predicate. The generic
    ``MoveArgument.crisp_refutations`` (the filter's input) is defined exactly
    as ``tier is Tier.FACT and refutes``."""
    from dialectical_chess.move_argument import Evidence, MoveArgument, Role

    fact_refutation = Evidence(
        label="fact_refute",
        role=Role.OBJECTION,
        tier=Tier.FACT,
        strength=6,
        refutes=True,
    )
    heuristic_objection = Evidence(
        label="soft",
        role=Role.OBJECTION,
        tier=Tier.HEURISTIC,
        strength=1,
        refutes=True,
    )
    argument = MoveArgument(
        move_id="e2e4",
        prior=0.5,
        objections=(fact_refutation, heuristic_objection),
    )
    # Only the FACT-tier objection is a crisp refutation; the HEURISTIC one,
    # even with refutes=True, is excluded — the gate keys on Tier.FACT.
    assert argument.crisp_refutations == (fact_refutation,)


def _fact_loss_objection(label: str, magnitude: int):
    """A generic FACT-tier objection carrying a proven-loss magnitude."""
    from dialectical_chess.move_argument import Evidence, Role

    return Evidence(
        label=label,
        role=Role.OBJECTION,
        tier=Tier.FACT,
        strength=0,
        magnitude=magnitude,
        refutes=False,
    )


@pytest.mark.unit
def test_generic_decider_fact_term_reads_generic_fact_tier_evidence() -> None:
    """The seam — ``decide``'s lexicographic FACT term is computed by reading
    the generic FACT-tier (``Tier.FACT``) objection evidence's magnitude, not
    by a chess policy call. A move with a larger proven loss is ordered
    strictly after one with a smaller loss, and both after a move with no
    proven FACT-tier loss."""
    from dialectical_chess.decide import (
        expectation_selection_key,
        worst_fact_objection_magnitude,
    )
    from dialectical_chess.move_argument import MoveArgument
    from dialectical_chess.opinion_graph import build_argumentation_artifacts
    from doxa.argumentation import evaluate

    clean = MoveArgument(move_id="a1a1", prior=0.5)
    small_loss = MoveArgument(
        move_id="b1b1",
        prior=0.5,
        objections=(_fact_loss_objection("loss:b1b1", 300),),
    )
    big_loss = MoveArgument(
        move_id="c1c1",
        prior=0.5,
        objections=(_fact_loss_objection("loss:c1c1", 900),),
    )
    # The FACT term is read off the FACT-tier objection magnitude.
    assert worst_fact_objection_magnitude(clean) == 0
    assert worst_fact_objection_magnitude(small_loss) == 300
    assert worst_fact_objection_magnitude(big_loss) == 900

    artifacts = build_argumentation_artifacts([clean, small_loss, big_loss])
    opinions = evaluate(artifacts.graph.graph)
    key_clean = expectation_selection_key(clean, artifacts, opinions)
    key_small = expectation_selection_key(small_loss, artifacts, opinions)
    key_big = expectation_selection_key(big_loss, artifacts, opinions)
    # The FACT term is the first key component (negated magnitude): clean
    # outranks small loss outranks big loss.
    assert key_clean > key_small > key_big


@pytest.mark.unit
def test_generic_decider_fact_term_ignores_heuristic_tier_objections() -> None:
    """The seam — the decider's FACT term keys strictly on ``Tier.FACT``: a
    HEURISTIC-tier objection, even one carrying a magnitude, never contributes
    to the FACT term (it is graded, not proven)."""
    from dialectical_chess.decide import worst_fact_objection_magnitude
    from dialectical_chess.move_argument import Evidence, MoveArgument, Role

    heuristic_with_magnitude = Evidence(
        label="soft:loss",
        role=Role.OBJECTION,
        tier=Tier.HEURISTIC,
        strength=4,
        magnitude=500,
    )
    argument = MoveArgument(
        move_id="d1d1", prior=0.5, objections=(heuristic_with_magnitude,)
    )
    assert worst_fact_objection_magnitude(argument) == 0


# ==========================================================================
# D3 — the defeater-suppression policy is the chess cartridge.
# ==========================================================================


@pytest.mark.unit
def test_suppression_policy_suppresses_defended_reply() -> None:
    """D3 — a defended reply suppresses itself; an undefended one does not."""
    defended = reply_evidence(
        "reply_mate:defended:a1a2", reply_attack_strength=7, defense_strength=13
    )
    undefended = reply_evidence(
        "reply_mate:undefended:a1a2", reply_attack_strength=7
    )
    probe = _probe_with(reply_attacks=(defended, undefended))
    assert suppressing_defeaters(probe, defended) == (defended,)
    assert suppressing_defeaters(probe, undefended) == ()


@pytest.mark.unit
def test_suppression_policy_leaves_undefeated_objection_unsuppressed() -> None:
    """D3 — an objection with no firing chess suppression rule is returned
    unsuppressed (no defeaters)."""
    objection = objection_evidence(
        "opening:king_walk:e1e2",
        world=EvidenceWorld.POSITIONAL,
        objection_kind=ObjectionKind.OPENING_KING_WALK,
        objection_strength=1,
    )
    probe = _probe_with(objections=(objection,))
    assert suppressing_defeaters(probe, objection) == ()


# ==========================================================================
# D2 — fact_material_loss is the honest FACT material-safety term.
# ==========================================================================


@pytest.mark.unit
def test_fact_material_loss_zero_for_a_clean_move() -> None:
    """D2 — a move with no material-safety objection has zero FACT loss."""
    probe = _probe_with(objections=())
    assert fact_material_loss(probe) == 0


@pytest.mark.unit
def test_fact_material_loss_scores_search_refuted_en_pris() -> None:
    """D2 — a search-refuted en-pris piece is a proven material loss; the
    FACT term scores it (scaled by the lost material)."""
    probe = _probe_with(
        objections=(
            objection_evidence(
                "safety:moved_piece_en_pris:500",
                world=EvidenceWorld.MATERIAL,
                objection_kind=ObjectionKind.MOVED_PIECE_EN_PRIS,
                objection_strength=17,
                moved_piece_en_pris_value=500,
                search_refutation_score=-600,
            ),
        )
    )
    assert fact_material_loss(probe) > 0


@pytest.mark.unit
def test_fact_material_loss_ignores_un_refuted_en_pris() -> None:
    """D2 — an en-pris objection with no search refutation is not yet a
    *proven* loss; the FACT term stays 0 (the graded layer still sees it)."""
    probe = _probe_with(
        objections=(
            objection_evidence(
                "safety:moved_piece_en_pris:500",
                world=EvidenceWorld.MATERIAL,
                objection_kind=ObjectionKind.MOVED_PIECE_EN_PRIS,
                objection_strength=17,
                moved_piece_en_pris_value=500,
            ),
        )
    )
    assert fact_material_loss(probe) == 0


# --- helpers ---------------------------------------------------------------


def _probe_with(*, objections=(), reply_attacks=()):
    """A minimal MoveProbe carrying the given typed objection / reply evidence."""
    from dialectical_chess.arguments import MoveProbe

    return MoveProbe(
        uci="e2e4",
        san="e4",
        score=0,
        is_checkmate=False,
        gives_check=False,
        is_capture=False,
        captured_value=0,
        promotion_value=0,
        reasons=(),
        objections=tuple(ev.label for ev in objections),
        reply_attacks=tuple(ev.label for ev in reply_attacks),
        objection_evidence=tuple(objections),
        reply_attack_evidence=tuple(reply_attacks),
    )
