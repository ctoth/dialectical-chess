"""Chess cartridge: :class:`ChessGradedPolicy` (chunk H').

Implements the core ``dialectical_games.arguments.GradedPolicy`` Protocol.
A per-build policy bound to the survivor probe set; the bound policy carries
two per-position caches:

* a per-label-prefix MATERIAL CDF over sibling magnitudes (the chess-only
  MATERIAL class), built from the survivor probes;
* a per-position rank-fraction over sibling ``child_eval`` values (the
  move base rate), built from the same probes.

Both caches are populated in :meth:`ChessGradedPolicy.with_probes`, which
the generic builder calls once at entry (chunk H'.a Protocol extension).

Chunk H' (Core Phase 3, 2026-05-24) dissolved the tuned chunk-G.1
:data:`_WITNESS_*` / :data:`_TAU_*` constants in favour of principled
beta-binomial / per-position Hazen rank-fraction derivations. Witness
opinions classify the label into one of four classes and dispatch:

* **MATERIAL** (chess-only): a centipawn-scale ``:{n}`` magnitude on one
  of the four prefixes :data:`_MATERIAL_PREFIXES`. The opinion's belief is
  the Hazen rank-fraction of this magnitude among the survivor probes'
  same-prefix MATERIAL magnitudes; the rest is uncertainty.
* **COUNT**: a count-scale ``:{n}`` magnitude on any other HEURISTIC label.
  ``Opinion.from_evidence(n, 0, MAX_ENT_PRIOR)``.
* **BOOLEAN**: a FIXED HEURISTIC label with no magnitude.
  ``Opinion.from_evidence(1, 0, MAX_ENT_PRIOR)``.
* **SEARCH**: a ``search_support:{backend}:{score}`` label. No core
  translation exists yet (chunk-H' plan §6-G); the policy returns a
  vacuous opinion, deferring SEARCH-class translation to a later cycle.

The ONE literal that survives in the witness/policy path is
:data:`MAX_ENT_PRIOR` (= 0.5, the max-entropy binary prior). Everything
else is derived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence

from doxa import Opinion

from dialectical_games.arguments import GradedPolicy, MoveProbe


#: The max-entropy binary prior. The ONLY literal that survives in the
#: witness / policy path under chunk H' -- every other number is derived
#: from witness semantics through :func:`doxa.Opinion.from_evidence` or
#: from a per-position Hazen rank-fraction.
MAX_ENT_PRIOR: float = 0.5


#: Label prefixes that carry a centipawn-scale ``:{n}`` magnitude treated
#: as MATERIAL by the chess policy. Per-prefix CDFs are built over the
#: survivor probes' same-prefix magnitudes; the resulting rank-fraction
#: is the witness opinion's belief.
_MATERIAL_PREFIXES: frozenset[str] = frozenset({
    "pro:piece_safety:defended",
    "pro:tactical:threat",
    "pro:smt:fork",
    "obj:smt:fork:moved_piece_en_pris",
})


class _WitnessClass(Enum):
    """Four-class witness taxonomy (chunk H' D2)."""

    MATERIAL = auto()
    COUNT = auto()
    BOOLEAN = auto()
    SEARCH = auto()


def _label_prefix(label: str) -> str:
    """The label prefix -- everything up to the final ``:{n}`` segment.

    A magnitude-carrying label is ``<prefix>:<n>``; ``label.rpartition(":")[0]``
    returns ``<prefix>``. For a fixed label (no ``:{n}``) this returns
    ``<label>``-without-the-final-segment, which is acceptable for the
    classifier (only MATERIAL_PREFIXES membership matters here).
    """
    return label.rpartition(":")[0]


def _classify(label: str, magnitude: int | None) -> _WitnessClass:
    """The chunk H' witness class of ``(label, magnitude)`` (chess cartridge).

    * SEARCH: any label starting with ``search_support:``. (No core
      taxonomy entry today; chunk-H' plan §6-G defers translation to a
      later cycle. The classifier still names the class so the policy
      can return a vacuous opinion.)
    * MATERIAL: a magnitude-carrying label whose prefix is in
      :data:`_MATERIAL_PREFIXES`.
    * COUNT: a magnitude-carrying label outside MATERIAL.
    * BOOLEAN: a no-magnitude FIXED label.
    """
    if label.startswith("search_support:"):
        return _WitnessClass.SEARCH
    if magnitude is None:
        return _WitnessClass.BOOLEAN
    if _label_prefix(label) in _MATERIAL_PREFIXES:
        return _WitnessClass.MATERIAL
    return _WitnessClass.COUNT


def _parse_int_after_last_colon(label: str) -> int | None:
    """Parse the integer trailing ``:{n}`` segment from ``label``."""
    head, sep, tail = label.rpartition(":")
    if not sep or not tail:
        return None
    try:
        return int(tail)
    except ValueError:
        return None


@dataclass(frozen=True)
class _MaterialCdf:
    """The per-label-prefix Hazen rank-fraction lookup.

    ``magnitudes`` is the sorted (ascending) tuple of sibling magnitudes for
    this prefix across the survivor probe set; mid-rank tie handling for
    equal magnitudes. ``rank_fraction(m)`` returns the Hazen plotting
    position ``rank(m) / (N + 1)`` strictly in ``(0, 1)``.
    """

    magnitudes: tuple[int, ...] = ()

    @property
    def size(self) -> int:
        """The number of sibling magnitudes pooled in this CDF."""
        return len(self.magnitudes)

    def rank_fraction(self, magnitude: int) -> float:
        """The Hazen rank-fraction of ``magnitude`` in this corpus.

        Mid-rank for ties (probes with equal magnitude share the same
        rank-fraction). The result is in ``(0, 1)`` strictly by Hazen's
        formula ``rank / (N + 1)``.

        With an empty corpus (``size == 0``) returns
        :data:`MAX_ENT_PRIOR` -- a defensive vacuous fallback; the
        construction path ensures every queried magnitude is in the
        corpus, so this branch is unreachable in practice.
        """
        n = self.size
        if n == 0:
            return MAX_ENT_PRIOR
        # Mid-rank: count strict-less + (1 + equal-count)/2.
        less = sum(1 for m in self.magnitudes if m < magnitude)
        equal = sum(1 for m in self.magnitudes if m == magnitude)
        if equal == 0:
            # magnitude not in corpus -- defensive insertion-rank.
            mid = less + 1.0
        else:
            # ranks are (less+1) .. (less+equal); mid = average.
            mid = less + (1 + equal) / 2.0
        return mid / (n + 1)


def _search_opinion() -> Opinion:
    """The SEARCH-class opinion under chunk H'.

    No core taxonomy entry translates ``search_support:{backend}:{score}``
    today (chunk-H' plan §6-G). Return the vacuous opinion -- honest "we
    don't have a derivation for this witness class yet". Chunk-H' plan
    §6-G recommends ``Opinion.from_probability(score_winrate, depth, 0.5)``
    when the SEARCH translation is added in a later cycle.
    """
    return Opinion.vacuous(MAX_ENT_PRIOR)


@dataclass(frozen=True)
class ChessGradedPolicy:
    """``GradedPolicy`` impl for chess (chunk H').

    The bound root ``board`` is retained on the dataclass for backward
    compatibility with the orchestrator's
    ``Cartridge.make_graded_policy(board)`` signature, but is not read.
    The per-position caches (``_material_cdfs`` and ``_child_eval_ranks``)
    are populated only by :meth:`with_probes`; an unbound policy
    (no :meth:`with_probes` call yet) falls back to neutral defaults.
    """

    board: Any = None
    _material_cdfs: dict[str, _MaterialCdf] = field(default_factory=dict)
    _child_eval_ranks: dict[str, float] = field(default_factory=dict)
    _child_eval_count: int = 0

    def with_probes(self, probes: Sequence[MoveProbe]) -> "ChessGradedPolicy":
        """Return a policy bound to ``probes`` (chunk H' D1, D4).

        Walks every survivor probe's reasons/objections/reply_attacks/
        defenses; for each MATERIAL-classified label, parses the magnitude
        and pushes it onto a per-prefix list. After the walk, builds the
        per-prefix Hazen CDF. Also builds the per-position
        ``child_eval`` rank-fraction cache (chess ``child_eval`` is mover-
        relative: LARGER is better for the mover, so the largest gets the
        highest rank-fraction -- ASCENDING orientation).
        """
        n = len(probes)
        if n == 0:
            return ChessGradedPolicy(board=self.board)

        # --- MATERIAL per-prefix CDF ----------------------------------------
        per_prefix: dict[str, list[int]] = {}
        for probe in probes:
            all_labels: tuple[str, ...] = (
                probe.reasons + probe.objections + probe.reply_attacks + probe.defenses
            )
            for label in all_labels:
                prefix = _label_prefix(label)
                if prefix not in _MATERIAL_PREFIXES:
                    continue
                magnitude = _parse_int_after_last_colon(label)
                if magnitude is None:
                    continue
                per_prefix.setdefault(prefix, []).append(magnitude)

        material_cdfs: dict[str, _MaterialCdf] = {
            prefix: _MaterialCdf(magnitudes=tuple(sorted(mags)))
            for prefix, mags in per_prefix.items()
        }

        # --- child_eval rank-fraction CDF (ASCENDING for chess) -------------
        # Chess child_eval is mover-relative: larger is better for the mover
        # (probe.py:427 sets it as int(squash(static_prior) * 1000), monotone
        # in the static prior). Smallest child_eval gets the LOWEST rank
        # fraction; largest gets the HIGHEST.
        sorted_probes = sorted(probes, key=lambda p: p.child_eval)
        ranks: dict[str, float] = {}
        i = 0
        while i < n:
            j = i + 1
            while (
                j < n
                and sorted_probes[j].child_eval == sorted_probes[i].child_eval
            ):
                j += 1
            mid = ((i + 1) + j) / 2.0
            frac = mid / (n + 1)
            for k in range(i, j):
                ranks[sorted_probes[k].move_id] = frac
            i = j

        return ChessGradedPolicy(
            board=self.board,
            _material_cdfs=material_cdfs,
            _child_eval_ranks=ranks,
            _child_eval_count=n,
        )

    def move_base_rate(self, probe: MoveProbe) -> float:
        """The move node's base rate ``a`` -- per-position rank-fraction.

        Reads the per-position CDF :meth:`with_probes` built. Chess
        ``child_eval`` is mover-relative; larger is better for the mover;
        the largest ``child_eval`` gets the highest rank-fraction. With an
        unbound policy (no :meth:`with_probes` call yet) the base rate is
        the neutral max-entropy prior :data:`MAX_ENT_PRIOR`.
        """
        rank_fraction = self._child_eval_ranks.get(probe.move_id)
        if rank_fraction is None:
            return MAX_ENT_PRIOR
        return rank_fraction

    def witness_opinion(
        self,
        *,
        probe: MoveProbe,
        label: str,
        magnitude: int | None,
    ) -> Opinion:
        """A HEURISTIC witness opinion (chunk H' D3).

        Classifies ``(label, magnitude)`` into one of four classes and
        dispatches:

        * BOOLEAN -> ``Opinion.from_evidence(1, 0, MAX_ENT_PRIOR)``.
        * COUNT   -> ``Opinion.from_evidence(magnitude, 0, MAX_ENT_PRIOR)``.
        * MATERIAL -> ``Opinion(b, 0, 1 - b, MAX_ENT_PRIOR)`` where ``b``
          is the Hazen rank-fraction of ``magnitude`` in the per-prefix
          CDF :meth:`with_probes` built. With a single-observation corpus
          (``N <= 1``) falls back to the BOOLEAN shape -- "one
          observation, no comparison" is honestly the BOOLEAN regime
          (chunk-H' plan §2 single-observation case).
        * SEARCH  -> :func:`_search_opinion` (vacuous; chunk-H' plan
          §6-G defers translation).
        """
        del probe  # the per-probe CDF lookup uses the label prefix, not the probe id
        cls = _classify(label, magnitude)
        if cls is _WitnessClass.BOOLEAN:
            return Opinion.from_evidence(1.0, 0.0, MAX_ENT_PRIOR)
        if cls is _WitnessClass.COUNT:
            assert magnitude is not None  # classify guarantees it
            return Opinion.from_evidence(float(magnitude), 0.0, MAX_ENT_PRIOR)
        if cls is _WitnessClass.MATERIAL:
            assert magnitude is not None  # classify guarantees it
            prefix = _label_prefix(label)
            cdf = self._material_cdfs.get(prefix)
            if cdf is None or cdf.size <= 1:
                # Single-observation case (chunk-H' plan §2): the rank space
                # collapses to "largest of one observation" -- no comparison.
                # Fall back to the BOOLEAN shape -- one observation, no
                # comparison, honest beta-binomial.
                return Opinion.from_evidence(1.0, 0.0, MAX_ENT_PRIOR)
            b = cdf.rank_fraction(magnitude)
            # Hazen rank-fraction is strictly in (0, 1) for any rank in
            # [1, N], so u = 1 - b is strictly in (0, 1) -- the doxa
            # non-dogmatic invariant holds.
            return Opinion(b, 0.0, 1.0 - b, MAX_ENT_PRIOR)
        # SEARCH or unknown class.
        return _search_opinion()

    @property
    def edge_trust(self) -> Opinion:
        """The (witness -> move) edge trust opinion.

        Edges are facts of the graph, not measured beliefs; structural
        trust is dogmatic-true. The base rate is the max-entropy binary
        prior :data:`MAX_ENT_PRIOR`.
        """
        return Opinion.dogmatic_true(MAX_ENT_PRIOR)


def make_graded_policy(board: Any = None) -> ChessGradedPolicy:
    """Construct a per-build chess graded policy bound to ``board``.

    Under chunk H' the constructor no longer caches board-derived position
    features -- every per-position aggregate (the MATERIAL CDFs and the
    survivor ``child_eval`` rank-fractions) is built in
    :meth:`ChessGradedPolicy.with_probes`, which the generic builder calls
    once at entry. ``board`` is retained for call-site backward
    compatibility but is not read.
    """
    return ChessGradedPolicy(board=board)


__all__ = [
    "ChessGradedPolicy",
    "MAX_ENT_PRIOR",
    "make_graded_policy",
]
