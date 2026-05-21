# dialectical-chess — Phase 2 move-decision semantics

This document specifies how `dialectical-chess` decides which move to play, as
shipped at the end of Phase 2. The decision is **opinion-valued**: the engine
builds an argument graph per position, evaluates it with a subjective-logic
gradual semantics, applies a Dung skeptical hard-filter, and picks the move with
the highest resolved expectation.

It is the "as shipped" companion to the design lock
`reports/argdriven-phase2-design-v2.md`. Where this document and the design lock
disagree, the design lock's numeric worked examples are the provenance; this
document describes the code that exists.

---

## 1. The pipeline

```
legal moves
  -> probe_moves(...)                      # dialectical_chess/probe.py
       per move: a MoveProbe carrying typed ArgumentEvidence
  -> build_argumentation_artifacts(probes) # dialectical_chess/opinion_graph.py
       one MoveArgumentationArtifacts:
         - graph:        a doxa BipolarOpinionGraph + move-argument index
         - filter_af:    a pure-attack Dung ArgumentationFramework
         - evidence_trace: argument id -> the ArgumentEvidence behind it
  -> evaluate(graph)                       # doxa.argumentation.evaluate
       dict[argument id -> resolved Opinion]
  -> skeptical_survivors(artifacts)        # dialectical_chess/skeptical_filter.py
       grounded extension of filter_af -> the set of admissible move UCIs
  -> argmax expectation()                  # dialectical_chess/decide.py
       bestmove = the survivor (or, if none survive, the move) whose move
       node has the largest Opinion.expectation()
```

`arguments.choose_move(probes)` is a thin wrapper that calls
`choose_move_argumentation(probes)` and returns the selected `MoveProbe`;
`engine.DialecticalChessEngine.analyze` consumes only `selected.uci`. There is
no scalar HCE decider any more — `build_root_argument_graph` and the old
lexicographic `selection_key` decider were deleted in chunk P2.3.

---

## 2. Why opinion-valued

The decision is not a sort over a centipawn scalar. Each move node carries a
**subjective-logic `Opinion`** `(b, d, u, a)` — belief, disbelief, uncertainty,
base rate — and the engine maximises `expectation() = b + a*u`.

This buys three things a single scalar cannot express:

- **Uncertainty is first-class.** A move with one weak reason and a move with no
  reasons are different even when their material is identical: the unargued move
  resolves to a vacuous opinion `(0, 0, 1, tau)` whose expectation is exactly
  its prior `tau`; the argued move's belief mass is non-zero. Total
  support-versus-attack conflict resolves into *uncertainty* `u`, not into a
  misleading mid-point disbelief.
- **Reasons aggregate as evidence mass.** One supporting reason, two, and four
  produce strictly increasing belief (design lock §1a: resolved expectation
  0.775 / 0.850 / 0.910 for one/two/four `k=1` reasons). The graph distinguishes
  a thinly-argued move from a well-argued one.
- **The argument graph governs the decision.** `probe.score`, the legacy HCE
  scalar, appears **nowhere** in the decision rule. The only scalar input is
  `tau`, and `tau` comes from a *disjoint* `static_prior` re-evaluation of the
  post-move board (§6) — never from `probe.score` and never from the evidence
  labels that become graph nodes.

The result is genuinely argumentation-driven: change the typed evidence a probe
emits and the move node's opinion — and therefore the decision — changes,
through the graph, by construction.

---

## 3. What `doxa.argumentation` provides

`doxa.argumentation` (package `doxa`, module `argumentation.py`) is the
chess-agnostic gradual-semantics layer. It supplies:

- **`BipolarOpinionGraph(arguments, intrinsic, supports, attacks,
  edge_opinions)`** — a frozen dataclass. `arguments` is a `frozenset[str]`;
  `intrinsic` maps each argument to its own `Opinion`; `supports` and `attacks`
  are disjoint `frozenset`s of `(source, target)` pairs; `edge_opinions` carries
  a per-edge trust `Opinion`. Construction validates that the intrinsic map
  covers exactly `arguments`, that every edge is declared, that supports and
  attacks are disjoint, and that there are no self-loops.
- **`evaluate(graph) -> dict[str, Opinion]`** — a Kahn bottom-up traversal that
  resolves every argument to a final `Opinion`. It raises `CyclicGraphError` on
  a cycle; the chess graph is a strict DAG (leaves -> move nodes), so it never
  cycles.
- **Model C accrual.** For each argument, `evaluate` per-edge discounts each
  child opinion by the edge trust, negates discounted *attackers*, assembles a
  source pool (the node's own intrinsic leads the pool only if it is
  non-vacuous), fuses the pool with cumulative-belief fusion `Opinion.ccf`, and
  re-stamps the base rate to the node's `tau`. An empty pool yields
  `Opinion.vacuous(tau)`.
- **`Opinion`** — `(b, d, u, a)` with `b + d + u = 1`, all of `b, d, u` in
  `[0, 1]`, and `a` strictly inside `(0, 1)`. `expectation() = b + a*u`.
  `Opinion.from_evidence(r, s, a)` builds an opinion from Beta evidence counts
  with prior weight `W = 2`. `Opinion.vacuous(a) = (0, 0, 1, a)`.

`dialectical-chess` constructs `BipolarOpinionGraph`s and reads `evaluate`'s
output. It does not re-implement fusion, discounting, or topological traversal.

The Dung skeptical filter (§5) uses a separate package, `argumentation`
(`argumentation.dung`), for `ArgumentationFramework` and `grounded_extension`.

---

## 4. The chess -> opinion-graph encoding (as shipped)

`build_argumentation_artifacts` (in `dialectical_chess/opinion_graph.py`) builds,
in one pass over the probe list, every Phase-2 artifact. Per probe:

### 4a. The move node

A `move:{uci}` argument with intrinsic
`Opinion.vacuous(squash(static_prior(probe)))`. It has **no evidence of its
own** — `tau = squash(static_prior(probe))` is its base rate, and an unargued
move therefore resolves to `expectation() == tau` exactly.

### 4b. The support leaf — one per move

All of a probe's `reason_evidence` items with `supports_argument` and
`support_strength > 0` are **aggregated**: the integer strengths are summed, and
**one** `support:{uci}` leaf is built with
`leaf_intrinsic(sum) = Opinion.from_evidence(sum * EV, 0.0, A_ROLE)`
(`EV = 2.0`, `A_ROLE = 0.5`, from `tuning.py`). One support edge
`(support, move)` is added. If the aggregate is `0`, **no leaf and no edge** —
a zero-strength leaf is never built (it would corrupt the fusion pool).

### 4c. The objection leaf — one per move

All `objection_evidence` and `reply_attack_evidence` items are aggregated the
same way into **one** `objection:{uci}` leaf with an attack edge
`(objection, move)`. Before aggregation, **defeater suppression** applies:
`effective_objection_strength` subtracts every applicable defeater's strength
from the objection's strength (`max(0, obj - sum(defeater))`). A fully-defeated
objection contributes `0` and is omitted, returning the move *exactly* to its
no-objection baseline — defeating an objection restores the move, it never
boosts it above baseline.

Defeaters are **not** graph nodes. `objection_defeater_evidence` synthesises a
defeater for exactly four objection kinds — `QUEEN_BLUNDER` with compensating
forcing pressure, `MOVED_PIECE_EN_PRIS` (value >= 300) with compensating
tactical pressure or forcing material gain, `OPENING_PREMATURE_MINOR_CHECK` with
a search-support reason, and `FLANK_PAWN_WEAKENING`/`FLANK_PAWN_LUNGE` with an
advanced-flank-pawn response — plus any objection that carries its own defense
strength. They are consumed at aggregation time only.

### 4d. Edges

Every support and attack edge carries the trust opinion
`Opinion.dogmatic_true(0.5)` — fully trusted. All discrimination lives in the
aggregate leaf strength; the edge layer is structurally required by `doxa` but
is a constant.

### 4e. The filter graph

For every objection / reply-attack evidence item for which
`is_forced_mate_refutation` is true, a `refute:{uci}:{label}` argument is added
to a separate, pure-attack Dung `ArgumentationFramework` with a single defeat
edge `(refute, move)`. There are **no counterdefeater edges**: a proven forced
mate is a terminal fact and is never counter-defeated in Phase 2. Soft
objections never enter the filter graph.

### 4f. The evidence trace

`evidence_trace` records, for each leaf and each `refute:` node, the
`ArgumentEvidence` items that fed it. The filter's `refute:` nodes and the
opinion graph's objection leaf are derived from the *same* aggregation pass, so
the two artifacts cannot disagree about which objections are sound refutations.
The trace is the explainability surface — every individual reason is retained
even though the graph presents one aggregate leaf per role.

---

## 5. The skeptical filter

The opinion graph alone is not enough. A queen-grab that walks into a forced
mate can still out-score a sound quiet move on `expectation()` whenever its
prior `tau` is high — `doxa`'s fusion routes total support-versus-attack
conflict into *uncertainty*, and a high `tau * high u` keeps the expectation
high (design lock §5a, worked example B: an into-mate queen grab resolves to
`E ~ 0.82` and beats a sound alternative at `E = 0.50`).

So a **Dung grounded-extension hard-filter** runs over the pure-attack
`filter_af`. `skeptical_survivors` computes the grounded extension; a move whose
`move:{uci}` node is not in the grounded extension is removed from the decision
pool. The `refute:` nodes have in-degree 0, so they are always in the grounded
extension and always fire — a move attacked by a `refute:` node is always
excluded.

**Empty-survivor fallback.** When *every* legal move is hard-refuted (every move
walks into a proven forced mate — a fully lost position), the survivor set is
empty. The decider sets `empty_survivors = True` and falls back to the full move
set, still ranked by `expectation()`: a lost position still gets a move played.

---

## 6. The static prior

`tau` is the move node's base rate and the only scalar input to the decision.
`static_prior(probe)` (in `dialectical_chess/static_prior.py`) is a **disjoint**
centipawn-scale re-evaluation of the post-move board: material balance plus a
small positional-geometry term, read **only** from `probe.post_fen`. It
deliberately ignores `probe.score` and every evidence label — material,
tactical, search, objection, reply, and defeater facts are the graph's job, and
counting them in `tau` as well would double-book them. `squash` maps the
unbounded centipawn prior into the open interval `[0.01, 0.99]` via
`0.5 + 0.5*tanh(prior / TAU_SCALE)` (`TAU_SCALE = 400.0`), so
`Opinion.vacuous(squash(...))` never raises.

---

## 7. The decision rule

```python
artifacts = build_argumentation_artifacts(probes)
opinions  = evaluate(artifacts.graph.graph)
survivors = skeptical_survivors(artifacts)
pool      = survivors or {p.uci for p in probes}      # empty-survivor fallback
bestmove  = max(
    (p for p in probes if p.uci in pool),
    key=lambda p: (opinions[move_arg[p.uci]].expectation(), p.uci),
)
```

The maximised quantity is the move node's resolved `expectation()`. On an exact
expectation tie, the lexicographically largest UCI wins — a pure determinism
device, making the decider a pure function of the probe set. No `probe.score`
term, no centipawn tiebreaker.

---

## 8. Resolved tensions

The opinion-valued design resolved four tensions the earlier weighted-graph
design left open (design lock, Codex review C1-C4 / M1):

- **Same-strength reasons no longer collapse (C1).** `doxa`'s `ccf` is
  idempotent on identical opinions, so one leaf per reason meant N equal-strength
  reasons fused to one. Aggregating per move per role *before* opinion
  construction makes one/two/four reasons strictly distinct.
- **Zero-strength evidence cannot corrupt fusion (C2).** A vacuous *child*
  arriving over an edge is appended to the fusion pool — Model C drops a vacuous
  *intrinsic* but not a vacuous edge source. The builder omits every
  zero-strength item entirely.
- **One artifact, no drift (C3).** A single builder produces the opinion graph,
  the filter framework, and the trace. The filter and the decider both consume
  it and never re-parse probes, so they cannot disagree on which objections are
  sound refutations.
- **Defeating an objection restores, never boosts (M1).** Encoding a defeater as
  an attack-on-the-objection turned the defeated objection's negation into
  positive belief for the move, pushing it *above* its no-objection baseline.
  Residual strength suppression at aggregation time lands a fully-defeated
  objection exactly on the baseline.

The value layer (Bench-Capon audience-relative value preferences) was **dropped**:
graded objection strength already does the soft-attack grading, and the one
binary thing that genuinely needs to be binary — a sound forced mate must
dominate regardless of supporters — is the Dung skeptical filter's job. Checklist
item 12 (audience-configurable value ordering) is honestly recorded as deferred,
not re-scoped as satisfied.

---

## 9. Axiom profile (headline)

`doxa.argumentation`'s gradual semantics, as used here:

| Property | Status | Note |
|---|---|---|
| Vacuity | holds | an unargued move resolves to `expectation() == tau` exactly |
| Balance | holds | belief moves above `tau` when supported, below when attacked |
| Monotone in aggregate strength | holds | more independent reasons -> strictly higher expectation (0.775 / 0.850 / 0.910 for 1 / 2 / 4 `k=1` reasons) |
| Strict monotonicity (textbook) | does not hold | an exact-duplicate leaf is absorbed by `ccf` idempotence; the engine emits each reason once, so this is not reached in practice |
| Bounded | holds | `Opinion` enforces `b, d, u in [0, 1]`; `expectation() in [0, 1]` |
| Anonymity / independence of argument names | holds modulo tie-break | the UCI tie-break is a deliberate determinism device |

The exhaustive Bonzon-16 / Baroni-11 axiom audit is deferred to Phase 3; this
table is the Phase-2 headline profile.

---

## 10. Scope and follow-ups

Phase 2 delivers the opinion-valued decision, the skeptical filter, the disjoint
static prior, and the one-artifact builder. Deferred to Phase 3: the
discriminated-union `ArgumentEvidence` (P3.1), carrying typed evidence directly
from `probe.py` and deleting the `evidence.py` string parsers (P3.2), a central
constants table (P3.3), and the exhaustive axiom audit. The strength/EPD
measurement gate is degenerate at the current 0%-versus-Stockfish-2000 Phase 1
baseline and is recommended as a separate follow-up (see
`reports/argdriven-phase2-8b.md`).
