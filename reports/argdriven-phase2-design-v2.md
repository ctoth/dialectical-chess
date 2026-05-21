# Phase 2 P2.1 — Design-lock v2: the chess→opinion-graph mapping (computed)

Date: 2026-05-20 · Status: **DESIGN LOCKED (revision v2).** Design note only —
no production code written, no file under `dialectical_chess/` or `tests/`
created or modified. Throwaway compute scripts were run against the real `doxa`
package (`C:\Users\Q\code\doxa`, HEAD `f076502184990a42f1531d3d7a27a5bcf606074c`)
and the real `argumentation` package (`C:\Users\Q\code\argumentation`), then
deleted — not committed or left in any repo.

This is the **self-contained replacement** for `reports/argdriven-phase2-design.md`.
The Phase 2 build (chunks P2.2–P2.8) uses **this** note, not the original. It
incorporates every fix from the Codex review (`reports/codex-phase2-design-review.md`):
four Criticals (C1–C4), four Majors (M1–M4), and four Minors. Section 12 maps each
finding to its resolution.

Every numeric claim below is shown with the computed worked example behind it.
All `doxa`-side numbers were produced by `Opinion.from_evidence`, `Opinion.ccf`,
and `doxa.argumentation.evaluate`; all Dung-side numbers by
`argumentation.dung.grounded_extension`.

Inputs read in full: `reviews/PLAN-argumentation-driven.md` (all of Phase 2),
`reports/codex-phase2-design-review.md`, `reports/argdriven-phase2-design.md`
(the original being revised), `reports/doxa-argumentation-gateA2.md`,
`C:\Users\Q\code\doxa\src\doxa\argumentation.py`,
`C:\Users\Q\code\doxa\src\doxa\opinion.py`,
`C:\Users\Q\code\argumentation\src\argumentation\dung.py`,
`reports/argdriven-research-engine.md`, `dialectical_chess/evidence.py`,
`dialectical_chess/probe.py`, `dialectical_chess/arguments.py`.

---

## 0. What `doxa.argumentation` actually provides — verified from source

Read directly from `doxa/src/doxa/argumentation.py` and `opinion.py`. This is the
surface Phase 2 maps onto; it is fixed and not negotiable.

- `BipolarOpinionGraph(arguments, intrinsic, supports, attacks, edge_opinions)` —
  five required fields, `@dataclass(frozen=True)`:
  - `arguments: frozenset[str]`
  - `intrinsic: Mapping[str, Opinion]` — per-argument own opinion; `tau` of an
    argument is `intrinsic[x].a`.
  - `supports: frozenset[tuple[str, str]]` — `(supporter, target)`.
  - `attacks: frozenset[tuple[str, str]]` — `(attacker, target)`.
  - `edge_opinions: Mapping[tuple[str, str], Opinion]` — per-edge trust/strength.
- Six construction validations (`__post_init__`, `argumentation.py:53-117`):
  intrinsic covers exactly `arguments`; support edges declared; attack edges
  declared; `supports ∩ attacks = ∅`; `edge_opinions` keys cover
  `supports ∪ attacks` exactly; no self-loops. Each is an explicit
  `raise ValueError`. Acyclicity is **not** a construction check.
- `evaluate(graph) -> dict[str, Opinion]` — Kahn bottom-up traversal over a
  sorted ready set; raises `CyclicGraphError(ValueError)` on a cycle.
- **Model C accrual** (`_accrue`, `argumentation.py:182-262`): for argument `x`,
  per-edge discount each child (`edge_opinion.discount(child)`), negate each
  discounted attacker (`~`), assemble the source pool; the node's intrinsic
  **leads the pool iff `intrinsic.u < 1.0 - 1e-9`** (a non-vacuous intrinsic is
  evidence; a vacuous one is dropped); fuse with `Opinion.ccf(*pool)`; re-stamp
  `a = tau_x`. Empty pool → `Opinion.vacuous(tau_x)`.
- `Opinion(b, d, u, a)` (`opinion.py:35-61`): `b+d+u ≈ 1`; `b,d,u ∈ [0,1]`;
  **`a` strictly in `(0,1)`** — `a = 0.0` or `a = 1.0` raises
  `ValueError("a=… not in (0, 1)")`. `expectation() = b + a·u`.
- `Opinion.from_evidence(r, s, a)` → `BetaEvidence(r,s,a).to_opinion()` →
  `b = r/(r+s+W)`, `d = s/(r+s+W)`, `u = W/(r+s+W)` with the kernel prior weight
  `W = 2` (`opinion.py:14`). This is the constructor the leaf encoding uses.
- `discount(source)`: receiver is the **trust** opinion. A `dogmatic_true(0.5)`
  edge is full trust — `dogmatic_true(0.5).discount(child)` returns `child`
  unchanged in `(b,d,u)`.
- CCF (`_ccf_binomial`, `opinion.py:587-714`) is **non-associative** and
  **idempotent on identical sources** (`ccf(op, op) == op`). This idempotence is
  the root cause of Codex finding C1 — see §1.

---

## 1. Leaf-opinion encoding — LOCKED (revised for C1, C2)

### 1a. The aggregation rule (Codex C1 fix — load-bearing)

**The problem.** The original design mapped *each* typed `ArgumentEvidence` item
to its own leaf argument and relied on CCF accrual to combine same-role leaves.
CCF is idempotent on identical opinions, and many engine reasons share exactly
the same integer strength `k` and base rate `a`. So N identical-`k` leaves fuse to
exactly one. Computed against real `doxa`, one support edge of `k=1`,
`tau=0.55`:

| # of separate `k=1` support leaves | resolved `E(move)` |
|---:|---:|
| 1 | 0.775000 |
| 2 | 0.775000 |
| 4 | 0.775000 |

The graph could not distinguish a move with one weak reason from a move with
four independent weak reasons. **This is corrected by aggregating evidence per
move per role *before* opinion construction.**

**The locked rule.** For a given move and a given role (support / objection), sum
the integer strengths of every non-zero evidence item of that role, then build
**one** leaf opinion from the summed strength:

```
agg_support_k(move) = Σ { e.support_strength : e ∈ reason_evidence(move),
                          e.supports_argument, e.support_strength > 0 }
agg_objection_k(move) = Σ { effective_objection_strength(e) :
                            e ∈ objection_evidence(move) ∪ reply_attack_evidence(move) }
```

`effective_objection_strength` is the residual after defeater suppression — see
§1d (the M1 fix). At most **one support leaf** and **one objection leaf** per
move. The leaf opinion is then:

```
leaf_intrinsic(k, a_role) = Opinion.from_evidence(r = k · EV, s = 0.0, a = a_role)
```

with `EV = 2.0` and `a_role = 0.5` (§1c). Closed form (since `s=0`, `W=2`):
`b = kEV/(kEV+2)`, `d = 0`, `u = 2/(kEV+2)`.

**Computed proof the aggregation fix gives distinct, monotone, sane values**
(`tau=0.55`, support edge, full trust):

| reasons | summed `k` | resolved `E(move)` |
|---|---:|---:|
| one `k=1` | 1 | 0.775000 |
| two `k=1` | 2 | 0.850000 |
| four `k=1` | 4 | 0.910000 |
| `k=3` + `k=1` + `k=1` | 5 | 0.925000 |
| one `k=3` | 3 | 0.887500 |
| one `k=9` | 9 | 0.955000 |

1 vs 2 vs 4 reasons now produce **distinct, strictly increasing** `E`. The result
is also exactly what a single leaf of the summed strength produces, by
construction — there is one leaf, so CCF idempotence cannot collapse anything.

**Why sum the strengths (not the Beta `r`).** Summing `k` then multiplying by `EV`
is identical to summing `r = k·EV` directly (`r` is linear in `k`); the rule is
stated in `k` because `k` is the integer the engine emits. The aggregation is a
*count-like* accrual: independent reasons of the same role add evidence mass.
This is the intended "multiple reasons aggregated as a set" property (checklist
item 21) and it replaces the rev-1 copy-multiplication (`add_typed_attack`
`:strength:i` loop), which P2.3 deletes.

**Explainability is preserved.** The individual `ArgumentEvidence` items are not
discarded — the artifact builder (§7, C3) keeps a trace table from each
`ArgumentEvidence` to the aggregate leaf it contributed to. The opinion graph
presents one leaf per role to CCF; the trace presents every reason to the
explanation surface.

### 1b. The leaf-opinion table — computed against real `doxa`

`leaf_intrinsic(k, a=0.5)` for every aggregate strength the engine can produce,
run against `doxa.Opinion.from_evidence`. The relevant range of `k` is now an
*aggregate* and can exceed the single-item bands:

| `k` | `b` | `d` | `u` | `E` (@a=0.5) |
|----:|----:|----:|----:|----:|
| 0 | 0.000000 | 0.0 | 1.000000 | 0.500000 |
| 1 | 0.500000 | 0.0 | 0.500000 | 0.750000 |
| 2 | 0.666667 | 0.0 | 0.333333 | 0.833333 |
| 3 | 0.750000 | 0.0 | 0.250000 | 0.875000 |
| 4 | 0.800000 | 0.0 | 0.200000 | 0.900000 |
| 6 | 0.857143 | 0.0 | 0.142857 | 0.928571 |
| 7 | 0.875000 | 0.0 | 0.125000 | 0.937500 |
| 9 | 0.900000 | 0.0 | 0.100000 | 0.950000 |
| 13 | 0.928571 | 0.0 | 0.071429 | 0.964286 |
| 17 | 0.944444 | 0.0 | 0.055556 | 0.972222 |
| 33 | 0.970588 | 0.0 | 0.029412 | 0.985294 |
| 97 | 0.989796 | 0.0 | 0.010204 | 0.994898 |

The family is monotone in `k`: belief rises, uncertainty falls, `E` rises
strictly. (`k=33` and `k=97` are the corrected defeater numbers — Minor 1; the
original note's "≈0.970 / ≈0.990" were stale. The exact values are
`E(33)=0.985294` and `E(97)=0.994898`.)

### 1c. Per-role base rate `a_role` — LOCKED

Every leaf needs an `a ∈ (0,1)`. A leaf's `a` only affects its own
`expectation()` display and, via CCF's confidence-weighted base-rate average, the
*fused* base rate of intermediate nodes — but **never the move node's base
rate**, which `evaluate` always re-stamps to `tau_x`. **Locked: `a_role = 0.5`
for every leaf role.** This keeps the encoding a one-parameter family (`EV`
only). The `evidence.argument_value` string is **not** consumed as a base rate —
the value layer is dropped (§4).

### 1d. `k=0` evidence is omitted entirely (Codex C2 fix — load-bearing)

**The problem.** The original design said a `k=0` item maps to a *vacuous* leaf
that "Model C drops." That is true only for a node's *own* intrinsic. A vacuous
*child* opinion arriving over a support or attack **edge** is appended to the
CCF pool before fusion — Model C does not drop edge sources. Computed against
real `doxa`, `tau=0.55`:

| graph | resolved `E(move)` |
|---|---:|
| `supports=[k9]`, `attacks=[k1]` | 0.725000 |
| `supports=[k9, k0-vacuous]`, `attacks=[k1]` | 0.882500 |

Adding a connected `k=0` "vacuous" support leaf corrupted the mixed-conflict
result by `+0.1575`. (Codex reported `0.725 → 0.8825`; reproduced exactly at
`tau=0.55`. At `tau=0.5` the same effect is `0.700 → 0.880`.)

**The locked rule.** `build_argumentation_artifacts` (§7) **omits every
zero-strength evidence item entirely** — no leaf, no edge. A `k=0` item never
reaches the graph. After per-role aggregation (§1a), if a move's
`agg_support_k == 0` there is **no support leaf and no support edge** for that
move; likewise for objections. The `leaf_intrinsic` helper is documented to
require `strength > 0` and raises `ValueError` on `0` — there is no
"`leaf_intrinsic(0) → Opinion.vacuous`" public branch (the original API's
zero-branch is deleted). This matters because aggregation can also *produce* a
zero: a move whose every objection has been fully defeated (§1e) has
`agg_objection_k == 0` and must carry no objection leaf at all.

### 1e. Defeaters / defenses: residual suppression (Codex M1 fix — load-bearing)

**The problem.** The original design encoded a defeater as a leaf that *attacks an
objection leaf*. Computed against real `doxa`, a move with one `k=1` objection
and a `k=17` defeater of that objection resolves to `E(move)=0.722222`; with a
`k=97` defeater, `E(move)=0.744898`. The no-objection baseline at `tau=0.5` is
`E=0.500000`. So defeating an objection did not *neutralize* it — it pushed the
move **above** its no-objection baseline. Model C negates the (now defeated)
objection at the target, and a defeated objection's `~`-image becomes positive
belief for the move. That is wrong: answering an objection should restore the
move to where it would be *without* that objection, never reward it.

**Three encodings were computed.** (a) Defeater as a support edge into the move:
still boosts above baseline (`E=0.722222` / `0.744898`) — identical pathology.
(b) Defeater attacks the objection leaf (the original): boosts above baseline.
(c) **Residual suppression** — the defeater reduces the objection's *strength*
before the objection leaf is built. Computed: this is the only encoding where
`[one objection + its defeater]` lands at-or-below the no-objection baseline.

**The locked encoding — residual suppression at aggregation time.** A defeater is
**not** a graph node. It is consumed by the evidence-aggregation step (§1a). For
each objection, the engine's synthesized defeater evidence
(`objection_defeater_evidence`, `arguments.py:187-216`) reduces that objection's
contribution:

```
effective_objection_strength(obj) = max(0, obj.objection_strength
                                            - Σ defeater_strength(d)
                                              for d in defeaters_of(obj))
```

The aggregate objection leaf is then built from
`agg_objection_k = Σ effective_objection_strength(obj)`. A fully-defeated
objection contributes `0` and, by the C2 rule (§1d), is omitted — the move
returns **exactly** to its no-objection result. Defenses against reply-attacks
work identically: `defense_strength` (13) subtracts from the
`reply_attack_strength` of the reply-attack it answers.

**Computed proof — restoration, not boost** (`leaf` encoding, objection base
`k=6`, varying defeater strength `d`, `tau=0.5`, no supporter):

| defeater `d` | residual obj `k = max(0,6−d)` | resolved `E(move)` |
|---:|---:|---:|
| 0 | 6 | 0.071429 |
| 2 | 4 | 0.100000 |
| 4 | 2 | 0.166667 |
| 6 | 0 (omitted) | 0.500000 |
| 17 | 0 (omitted) | 0.500000 |

Monotone in `d`; a stronger defeater drives `E` toward the **baseline 0.5**, and
a fully-answering defeater lands *exactly* on it — never above. Across
`tau ∈ {0.3, 0.5, 0.7}` the `[objection fully defeated]` result equals the
`[no objection]` baseline to machine precision. A partially-defeated objection
(base `k=6`, defeater `d=4`, residual `k=2`) sits strictly between the
full-objection result and the baseline: `tau=0.5` → full `0.071429` < residual
`0.166667` ≤ baseline `0.500000`. This is the locked M1 semantics: **defeating an
objection restores the move toward its no-objection baseline; it never boosts
the move above it.**

A consequence for the engine side: `defeater_strength` (17/33/33/33/97) is large
relative to `objection_strength` (1–6), so any synthesized defeater fully cancels
its objection (`d ≥ obj` always). Partial residuals only arise if a future phase
introduces weaker defeaters; the `max(0, …)` rule already covers that case
correctly and is locked now so P2.5 needs no change later.

### 1f. The `tuning.py` constants

The integer strength tables in `evidence.py` (`OBJECTION_STRENGTHS`,
`DEFEATER_STRENGTHS`, the `support_strength` bands) **stay as the integer
strength source** — only their consumption changes. The single new Phase-2
intrinsic-opinion parameter family is:

```
EV       = 2.0     # evidence units per strength point  (tuning.py)
A_ROLE   = 0.5     # base rate of every leaf argument   (tuning.py)
```

Two constants, both calibratable, both with the §1b table as provenance.

---

## 2. The move node — LOCKED

A `move:{uci}` argument has **no evidence of its own**. Its intrinsic is:

```
intrinsic[move:{uci}] = Opinion.vacuous(tau)
```

where `tau = squash(static_prior(probe))` — the squashed static prior, fully
specified in §8 (the M3 fix). `Opinion.vacuous(tau) = (0, 0, 1, tau)`.

**`tau` must be strictly in `(0,1)`.** `Opinion.vacuous(0.0)` and
`Opinion.vacuous(1.0)` both raise `ValueError("a=… not in (0, 1)")`. `squash`
clamps to `[0.01, 0.99]` (§8).

### Confirmed: an unargued move resolves to `expectation() == tau`

Computed against `evaluate` for a move node with a vacuous intrinsic and no
edges, over `tau ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`: `E == tau` exactly in every case.
The vacuous intrinsic is dropped from the CCF pool by Model C; the empty-pool
branch returns `Opinion.vacuous(tau_x)`; `E = b + a·u = 0 + tau·1 = tau`. The
move node's base rate is always re-stamped to `tau` even when argued. This is the
SC1 vacuity property (checklist item 15).

---

## 3. The edge-opinion model — LOCKED

Every edge in `supports ∪ attacks` needs an `Opinion` in `edge_opinions`. The
edge opinion is the **trust** opinion — `discount`'s receiver.

**Locked default: `Opinion.dogmatic_true(0.5)` — fully trusted — for every
support and attack edge.** `dogmatic_true(0.5).discount(child)` returns the child
unchanged in `(b,d,u)`. All discrimination lives in the aggregate leaf strength
`k` (§1) — one calibrated quantity per move-role, not two. Grading the edge and
grading the leaf would be two knobs for one job; the plan's tuning discipline
(J9/J10) forbids that. The edge layer is kept structurally (`edge_opinions` is a
required field) but is a constant. The `a` of an edge opinion is irrelevant to
`discount`; locked at `0.5` for tidiness.

---

## 4. The value layer — LOCKED: **DROPPED**

The rev-1 design introduced a 6-value Bench-Capon ordering and computed
`defeat = attack + valpref`. **Locked decision: the value layer is dropped from
the opinion-valued Phase 2.** No `values.py`, no value vocabulary, no
`preference_undercut_attacks`. Three computed reasons:

### 4a. Opinion strength subsumes graded value-relative defeat

Bench-Capon's purpose is "a weak-value attack should not fully defeat a
strong-value target." In the opinion-valued model an attack's effect is *already*
graded by the aggregate objection strength. Computed: a weak `objection_k=1`
objection against a move with a `support_k=9` supporter, `tau=0.6`:

| weak `k=1` objection | resolved `E(move)` |
|---|---:|
| kept as an attack edge | 0.750000 |
| dropped (value-filtered out) | 0.960000 |

The objection moves `E` by `−0.21` — the correct graded outcome of a real-but-
minor objection, not a binary "defeats / does not defeat." The opinion model's
strength channel **is** the graded-defeat mechanism Bench-Capon approximated.

### 4b. The one thing values gave that strength does not — is given by §5 instead

The single Bench-Capon device strength does *not* replicate is the
audience-independent **dominating value** (`SOUNDNESS`): a sound forced-mate
refutation must defeat regardless of how strong the move's supporters are. §5
shows — computed — the opinion model cannot do this through strength. §5's answer
is the **Dung skeptical hard-filter**, a cleaner binary mechanism for the one
binary thing that genuinely needs to be binary. The value layer's only
irreplaceable job is reassigned to §5's filter; its other job (grading soft
attacks) is done by §1's strength encoding. Nothing is left for a value layer.

### 4c. Honest accounting for checklist item 12 (Codex M2 fix)

Checklist item 12 ("value ordering configurable; changing it can change the
move") asked for an *audience-relative* value permutation. The original note
claimed the `EV`/`TAU_SCALE` tuning parameters satisfy this. **Codex M2 is
accepted: that claim is false.** `EV` and `TAU_SCALE` are *global* calibration
knobs; a Bench-Capon value ordering is *audience-relative per argument value*.
They are not equivalent. A global knob that shifts every move's strength is not
the same capability as an audience permutation that can reorder *which* move wins
because it values, say, `KING_SAFETY` over `MATERIAL`.

**Locked: checklist item 12 is marked DEFERRED / NOT-SATISFIED, not "re-scoped
as satisfied."** Phase 2 does not deliver audience/value reordering in any form.
The §8 checklist triage and the P2.8 exit gate record item 12 as **deferred**
with the honest reason "the value layer is dropped (§4); no audience mechanism
exists in Phase 2." If a later phase wants it, it is a Phase-3 addition. There is
no false claim that a tuning knob substitutes for it.

### Consequence for the plan

Chunk **P2.6** ("Value layer / skeptical filter — *if the P2.1 design keeps
them*") is **half-dropped**: no `values.py`. P2.6 is **not empty** — it still
builds the skeptical filter (§5).

---

## 5. The skeptical filter — LOCKED: **KEPT** (Dung grounded hard-filter)

### 5a. The decisive computation — the filter is mandatory

Does opinion `expectation()` alone drive an into-mate move low enough, or is a
Dung `grounded_extension` hard-filter still required? **Computed answer: the
hard-filter is REQUIRED**, and this conclusion survives the C1 aggregation fix.

The original note's worked example B used the *broken* one-leaf-per-reason
encoding (`nf3` = two separate `k=1` leaves → `E=0.775`). Under the corrected
aggregation (§1a) `nf3`'s two reasons aggregate to `k=2` → `E=0.850`, which
*beats* the original `qxq` number — so example B as written no longer
demonstrates the failure. **A corrected worked example is computed below.** The
failure mode is unchanged; only the specific numbers move.

**Worked example A — a forced-mate objection alone vs. with one supporter**
(`tau=0.75`, dogmatic mate objection, full-trust edges):

| configuration | resolved `omega_m` | `E(move)` |
|---|---|---:|
| dogmatic mate objection, **no supporter** | `(0, 1, 0, 0.75)` | **0.000000** |
| dogmatic mate objection, **+ one `k=9` supporter** | `(0, 0.100, 0.900, 0.75)` | **0.675000** |

The objection alone kills the move. Add a single genuine supporter and `E` jumps
back to `0.675` — above `tau`-neutral. CCF routes total support/attack conflict
into **uncertainty**, not disbelief; high `tau × high u` keeps `E` high.

**Worked example B (corrected, post-C1-fix) — a queen-grab-into-mate.** With the
aggregated encoding:

- `move:qxq` — grabs a queen (`tau=0.88` from the static prior, which sees the
  queen capture); aggregate support leaf `k=9` (`material:capture:900`);
  aggregate objection leaf `k=6` (reply forced mate). Resolved
  **`E(qxq) = 0.820000`** (`b=0.128571, d=0.085714, u=0.785714`).
- `alt` — an unargued alternative move (`tau=0.50`). Resolved **`E(alt) = 0.500000`**.
- `argmax expectation()` → **`qxq`**. The move that walks into forced mate wins.

Even against a quiet move *with* a real reason, the into-mate capture wins
whenever its prior is high enough: `qxq` (`tau=0.90`, support `k=9`, mate
objection `k=6`) → `E=0.835714`, beating a quiet developing move (`tau=0.55`,
support `k=1`) → `E=0.775000`. And a stronger aggregate support only widens the
gap. This is the K2 soundness failure the plan exists to fix. **Opinion
propagation alone re-introduces it; the skeptical filter is kept.**

### 5b. What the filter is

A Dung **grounded-extension hard-filter**, computed by
`argumentation.dung.grounded_extension` over a **pure-attack** Dung framework — a
*separate, smaller* graph from the `BipolarOpinionGraph`, built from the **same**
artifact (§7, the C3 fix) so the two cannot drift.

The filter graph contains **only** audience-independent, sound refutations as
attacks — the evidence the engine can *prove* forces a loss. Membership is
decided by the single predicate `is_forced_mate_refutation`
(`evidence.py:407-413`), which is true iff:

- `search_refutation_score ≤ −100_000` (the mate sentinel `probe.py` writes for
  `search_refutes:` objections), or
- `objection_kind ∈ {REPLY_MATE_IN_ONE, REPLY_FORCED_MATE}`, or
- the reply-attack analyzer emits an undefended reply checkmate label:
  `reply_mate:undefended:{uci}`.

Each such refutation is a Dung argument `refute:{uci}:{label}` that **attacks**
`move:{uci}`. Soft objections (positional, opening, drift — anything not in the
`is_forced_mate_refutation` set) **do not enter the filter graph**; they live
only in the `BipolarOpinionGraph` and act through aggregate strength (§1, §4). A
move not in the grounded extension of the filter graph is removed from the
decision pool. A `reply_mate:defended:{uci}` label is not a filter refutation;
its defense is handled by residual suppression (§1e).

**Computed confirmation** (real `argumentation.dung`): filter graph
`arguments={move:qxq, move:nf3, refute:qxq:mate}`,
`defeats={(refute:qxq:mate, move:qxq)}` →
`grounded_extension = {move:nf3, refute:qxq:mate}`. `move:qxq` is **excluded**;
`move:nf3` survives. Refutation nodes have in-degree 0 (they are leaves of the
filter graph), so they are always IN the grounded extension and always fire.

### 5c. Counterdefeater policy (Codex C4 fix — settled, not left to the coder)

**The question:** is `is_forced_mate_refutation` evidence ever counter-defeatable
in Phase 2 — does the filter graph need `counterdefeater → refutation` edges?

**Examined the engine evidence model.** Defeater evidence in
`dialectical_chess` is synthesized by `objection_defeater_evidence`
(`arguments.py:187-216`). It emits a defeater for exactly four objection kinds:

- `QUEEN_BLUNDER` (+ `has_compensating_forcing_pressure`),
- `MOVED_PIECE_EN_PRIS` (+ compensating tactical pressure / forcing material gain),
- `OPENING_PREMATURE_MINOR_CHECK` (+ a `SEARCH_SUPPORT` reason defeater),
- `FLANK_PAWN_WEAKENING` / `FLANK_PAWN_LUNGE` (+ `ADVANCED_FLANK_PAWN_RESPONSE`).

**None of those is a forced-mate refutation kind.** `REPLY_MATE_IN_ONE`,
`REPLY_FORCED_MATE`, and the `search_refutes:` mate-sentinel — the entire
`is_forced_mate_refutation` set — **never receive a synthesized defeater.** The
engine produces no counter-evidence against a forced-mate refutation, and it
cannot: a forced mate is a terminal fact, established by an actual checkmate
search (`reply_mate_in_one_objections` finds a real `owned_is_checkmate`;
`reply_forced_mate_objections` calls `has_forced_mate`). A proof of forced mate
is not the kind of thing a heuristic compensation argument can answer.

**Locked:** in Phase 2 the filter graph is **`move ← refutation` only**, with
**no counterdefeater edges**. There is no closed list of filter-level
counterdefeaters because there are none. This is locked, not left to coder
judgment, and **P2.6 must include a test asserting it**:
`test_filter_graph_has_no_counterdefeater_edges` — for every probe set, the
filter `ArgumentationFramework.defeats` relation contains only pairs whose
attacker is a `refute:` node and whose target is a `move:` node; no pair has a
`refute:` node as its target. The computed contrast that motivates the lock: with
no counterdefeater, `defeats={refute→qxq}` → grounded `{move:nf3, refute:qxq:mate}`
(qxq excluded, permanently filtered); *if* a `cdef→refute` edge were ever added,
`defeats={refute→qxq, cdef→refute}` → grounded `{cdef, move:nf3, move:qxq}` (qxq
reinstated). Phase 2 locks the first behavior: a proven forced mate is never
reinstated.

### 5d. Empty-survivor fallback (Codex Minor 3 fix — testable policy)

The grounded extension can exclude *every* move (a position where every legal
move is a proven loss). A chess engine must still return a move. **Locked
policy:** the decider exposes an `empty_survivors: bool` flag on its result. If
the survivor set is empty, `empty_survivors = True` and the pool falls back to
**all moves**, ranked by `expectation()` (§6) — a lost position still has a
least-bad move. `empty_survivors` is a structured field, not a log line, so it is
machine-observable.

**Required tests (P2.6):**
- `test_empty_survivors_when_every_move_hard_refuted` — a constructed position (or
  synthetic probe set) where every move carries a forced-mate refutation; assert
  `skeptical_survivors(...)` is empty, the decider sets `empty_survivors=True`,
  and still returns a move. Computed against real `argumentation.dung`: filter
  `defeats={refute:a→move:a, refute:b→move:b}` → grounded `{refute:a, refute:b}`,
  survivors `{}` — confirmed.
- `test_over_filtered_position_detected` — a constructed position where the filter
  excludes a move that is *not* in fact lost (an over-broad filter), asserting the
  test fails loudly. This guards against the filter silently masking a real
  candidate. (Phase 2's filter membership is exactly `is_forced_mate_refutation`,
  which is sound by construction, so this test is expected to pass trivially
  today; it exists so a future widening of the filter predicate cannot regress
  silently.)

---

## 6. The decision rule — LOCKED (one formula, artifact-based)

```
artifacts = build_argumentation_artifacts(probes)   # §7 — single builder (C3)

opinions  = evaluate(artifacts.graph.graph)         # doxa.argumentation.evaluate
survivors = { uci for uci in artifacts.move_arg
              if f"move:{uci}" in grounded_extension(artifacts.filter_af) }
empty     = (len(survivors) == 0)
pool      = survivors if survivors else { p.uci for p in probes }   # §5d fallback

bestmove  = max( (p for p in probes if p.uci in pool),
                 key = lambda p: ( opinions[artifacts.move_arg[p.uci]].expectation(),
                                   p.uci ) )
```

- The maximised quantity is `expectation() = b + a·u` of the move node's resolved
  `Opinion`. The full `Opinion` (belief vs. uncertainty) is retained on the
  result for explanation.
- **Tie-break (Codex Minor 4 fix).** `max(..., key=(expectation, p.uci))` selects
  the **largest** UCI string on an exact-`expectation` tie (Python `max` returns
  the maximum of the tuple, and the tuple's second element is `p.uci`). The locked
  rule is therefore stated explicitly and consistently: **"on an exact
  expectation tie, the lexicographically largest UCI wins."** This is purely a
  determinism device — it makes the decider a pure function of the probe set,
  independent of `frozenset` iteration order. It is *not* claimed to mirror
  `doxa`'s internal smallest-name-first traversal (that comparison in the
  original note was incorrect; `doxa`'s traversal order is an evaluation-order
  detail, not a tie-break). Exact ties remain possible because two distinct moves
  can have identical `tau` and identical aggregate strengths. If a future phase
  wants smallest-UCI instead, negate the second key element; Phase 2 locks
  largest-UCI for the single reason that it is what `max` does with the natural
  tuple and re-stating it honestly is cheaper than inverting it.
- No `probe.score` term, no centipawn tiebreaker: `probe.score` appears
  **nowhere** in this rule. `tau` comes from the disjoint `static_prior` (§8).
- `pool` is the skeptical-survivor set, falling back to all moves only when the
  survivor set is empty (§5d).

---

## 7. The `dialectical-chess`-side module API — LOCKED (one artifact builder, C3)

### 7a. The single-artifact contract (Codex C3 fix — load-bearing)

The original API exposed two independent builders, `build_opinion_graph(probes)`
and `skeptical_survivors(probes)`. That lets P2.5 and P2.6 each re-parse the
probes and classify evidence independently — they can drift on *which* objections
count as sound refutations. **Locked fix: one builder produces one artifact**;
the decider and the filter both consume that artifact and never re-parse probes.

```python
# dialectical_chess/opinion_graph.py  (chunk P2.5)
from __future__ import annotations

from dataclasses import dataclass

from doxa import Opinion
from doxa.argumentation import BipolarOpinionGraph
from argumentation.dung import ArgumentationFramework

from dialectical_chess.arguments import MoveProbe
from dialectical_chess.evidence import ArgumentEvidence


# --- leaf-opinion encoding (§1) ---

EV: float = 2.0          # evidence units per strength point   (-> tuning.py)
A_ROLE: float = 0.5      # base rate of every leaf argument    (-> tuning.py)


def leaf_intrinsic(strength: int) -> Opinion:
    """Encode a positive aggregate evidence strength as a leaf Opinion.

    REQUIRES strength > 0. Raises ValueError on strength <= 0 — a
    zero-strength item must be omitted by the builder before this is
    called (Codex C2). There is no vacuous-leaf branch.

    strength > 0 -> Opinion.from_evidence(strength * EV, 0.0, A_ROLE).
    """


@dataclass(frozen=True)
class BipolarMoveGraph:
    """The opinion graph for one position plus the move->argument-id index."""
    graph: BipolarOpinionGraph
    move_arg: dict[str, str]            # uci -> "move:{uci}" argument id


@dataclass(frozen=True)
class MoveArgumentationArtifacts:
    """The single artifact every Phase-2 consumer reads. Built once, by
    build_argumentation_artifacts, from one MoveProbe list."""
    graph: BipolarMoveGraph            # the doxa BipolarOpinionGraph + index
    move_arg: dict[str, str]           # uci -> "move:{uci}" (mirrors graph.move_arg)
    filter_af: ArgumentationFramework  # pure-attack Dung AF (refute -> move only)
    evidence_trace: dict[str, list[ArgumentEvidence]]
        # opinion/filter argument id -> the ArgumentEvidence items aggregated
        # into it. Keys cover every leaf and every refute: node. This is the
        # explainability surface (§1a) and the consistency anchor: the filter's
        # refute: nodes and the opinion graph's objection leaves are derived
        # from the SAME aggregation pass, so they cannot disagree about which
        # objections are sound forced-mate refutations.


def build_argumentation_artifacts(
    probes: list[MoveProbe],
) -> MoveArgumentationArtifacts:
    """Build, in ONE pass over `probes`, every Phase-2 argumentation artifact.

    Per probe:
      * a `move:{uci}` argument, intrinsic = Opinion.vacuous(squash(static_prior(probe)));
      * aggregate the reason evidence -> at most ONE support leaf
        `support:{uci}` with strength = sum of positive support_strength,
        one support edge (support leaf -> move). Omitted entirely if the
        aggregate strength is 0 (C2).
      * aggregate the objection + reply-attack evidence, applying defeater /
        defense residual suppression (§1e, M1) -> at most ONE objection leaf
        `objection:{uci}` with strength = sum of effective_objection_strength,
        one attack edge (objection leaf -> move). Omitted entirely if the
        aggregate strength is 0 (C2 / fully-defeated objection).
      * every edge opinion = Opinion.dogmatic_true(0.5) (§3).
      * for every objection-or-reply-attack evidence item with
        is_forced_mate_refutation(item) True, a `refute:{uci}:{label}`
        argument in `filter_af` and a single defeat edge
        (refute -> move). NO counterdefeater edges (§5c, C4).
      * evidence_trace records, for each leaf and each refute: node, the
        ArgumentEvidence items that fed it.

    doxa validates the BipolarOpinionGraph at construction and raises
    CyclicGraphError in evaluate; argumentation.dung validates the
    ArgumentationFramework; this builder relies on those and does not
    re-implement them.
    """
```

`static_prior` and `squash` are `dialectical_chess/static_prior.py` (chunk
P2.4); the full contract is §8 of this note (the M3 fix).

### 7b. The filter consumer

```python
# dialectical_chess/skeptical_filter.py  (chunk P2.6 — the surviving half)
from __future__ import annotations

from argumentation.dung import grounded_extension

from dialectical_chess.opinion_graph import MoveArgumentationArtifacts


def skeptical_survivors(artifacts: MoveArgumentationArtifacts) -> set[str]:
    """{ uci : "move:{uci}" in grounded_extension(artifacts.filter_af) }.

    Consumes the pre-built artifact — does NOT re-parse probes (Codex C3).
    The filter_af is a pure-attack Dung framework of forced-mate refutations
    only (§5b); refute: nodes attack move: nodes and nothing else (§5c).
    """
```

There is **no `build_skeptical_filter(probes)`** — the filter `ArgumentationFramework`
is `artifacts.filter_af`, built by the one builder in 7a. This reconciles the
P2.7 plan sketch with the design (Codex Minor 2): the plan sketch's
`skeptical_survivors(bmg)` and the original note's `skeptical_survivors(probes)`
are both replaced by `skeptical_survivors(artifacts)`.

### 7c. The decider

```python
# dialectical_chess/decide.py  (chunk P2.7)
from __future__ import annotations

from dataclasses import dataclass

from doxa import Opinion
from doxa.argumentation import evaluate
from argumentation.dung import grounded_extension

from dialectical_chess.arguments import MoveProbe
from dialectical_chess.opinion_graph import (
    MoveArgumentationArtifacts,
    build_argumentation_artifacts,
)
from dialectical_chess.skeptical_filter import skeptical_survivors


@dataclass(frozen=True)
class ArgumentationDecision:
    """The decider result. `selected` is one of the input probes."""
    selected: MoveProbe
    empty_survivors: bool                 # §5d — True iff the filter excluded
                                          # every move and the pool fell back
    move_opinion: dict[str, Opinion]      # uci -> resolved move-node Opinion
                                          # (full belief/uncertainty, for
                                          # explanation)


def choose_move_argumentation(probes: list[MoveProbe]) -> ArgumentationDecision:
    """The opinion-valued decider (§6). Returns an ArgumentationDecision
    whose `selected` is one of the input probes."""
    if not probes:
        raise ValueError("position has no legal moves")
    artifacts: MoveArgumentationArtifacts = build_argumentation_artifacts(probes)
    opinions = evaluate(artifacts.graph.graph)
    survivors = skeptical_survivors(artifacts)
    empty = not survivors
    pool = survivors if survivors else {p.uci for p in probes}
    move_arg = artifacts.move_arg
    selected = max(
        (p for p in probes if p.uci in pool),
        key=lambda p: (opinions[move_arg[p.uci]].expectation(), p.uci),
    )
    return ArgumentationDecision(
        selected=selected,
        empty_survivors=empty,
        move_opinion={uci: opinions[arg] for uci, arg in move_arg.items()},
    )
```

### 7d. How it wires into the existing engine

`arguments.py:choose_move` becomes the thin wrapper the plan's P2.3 specifies:
it calls `choose_move_argumentation(probes)` and returns `decision.selected`
(a `MoveProbe`, as the existing `choose_move` contract requires). The caller
`engine.py:analyze` reads only `selected.uci` / `selected` from the returned
`MoveProbe`, so no signature change propagates beyond `choose_move`. The new
`empty_survivors` / `move_opinion` fields are available on the
`ArgumentationDecision` for the explanation surface and for tests; the legacy
`choose_move` path discards them, which is acceptable for P2.3's thin wrapper.

---

## 8. The `static_prior.py` contract — LOCKED (Codex M3 fix — full spec)

P2.5 consumes `tau`; `tau` decides every unargued move and interacts directly
with high-uncertainty conflict (§5a shows high `tau` keeps an into-mate `E` high).
The original note declared `static_prior` "out of scope." **That is corrected
here: the full contract is locked now**, so P2.4 builds to a fixed spec and P2.5
cannot be blocked.

### 8a. Function signatures

```python
# dialectical_chess/static_prior.py  (chunk P2.4)
from __future__ import annotations

from dialectical_chess.arguments import MoveProbe

TAU_SCALE: float = 400.0     # centipawn scale of the squash (-> tuning.py)
TAU_CLAMP: tuple[float, float] = (0.01, 0.99)   # open-interval-safe clamp


def static_prior(probe: MoveProbe) -> float:
    """A centipawn-scale positional/material prior for `probe`'s move,
    computed DISJOINTLY from the typed ArgumentEvidence that becomes graph
    nodes. Sign convention: positive favours the move. Range: unbounded
    signed centipawns (a residual HCE-like scalar)."""


def squash(prior: float) -> float:
    """Map an unbounded centipawn prior to a base rate in [0.01, 0.99].

        squash(prior) = clamp( 0.5 + 0.5 * tanh(prior / TAU_SCALE),
                               *TAU_CLAMP )

    squash(0.0) == 0.5 exactly; monotone increasing; never reaches the
    open-interval endpoints 0.0 / 1.0 (so Opinion.vacuous(squash(...))
    never raises). Computed: prior=+400 -> 0.880797; prior=-400 ->
    0.119203; prior=0 -> 0.500000."""
```

The move node's intrinsic is `Opinion.vacuous(squash(static_prior(probe)))` (§2).

### 8b. Included terms — positional and material only

`static_prior` is a *disjoint* re-evaluation of the position after the move. It
**includes** exactly the standard HCE-style positional and material terms,
evaluated on the post-move board:

- **Material balance** — summed piece values of the side to move minus the
  opponent, on the post-move board (`OWNED_PIECE_VALUE` scale: P=100, N=300,
  B=300, R=500, Q=900).
- **Piece-square / activity** — a small positional term: central pawn presence,
  minor-piece development off the back rank, rook/queen on an open file, knight
  on a supported outpost, passed pawns. These mirror the *geometry* the
  positional-reason detectors look at, but `static_prior` reads the **board
  state**, never the reason labels.
- **King-safety geometry** — castled-vs-uncastled, pawn-shield integrity — again
  read from board state.

The output is a signed centipawn scalar; it is *not* clamped before `squash`
(squash does the clamping).

### 8c. Excluded `ArgumentEvidence` label families — the no-double-count rule

`probe.score` is an accumulated mix of the very facts that become evidence nodes
(`reports/argdriven-research-engine.md §5`). `static_prior` **must not** be a
function of `probe.score`, and it **must not** re-count anything the typed
evidence already carries into the graph. The following label families are the
graph's job and are **excluded** from `static_prior`:

| excluded family | why — it is already a graph node |
|---|---|
| `terminal:checkmate`, `procedural:mate_in_one` | terminal facts → support leaf / decisive |
| `material:capture:*`, `material:promotion:*` | the capture/promotion is a **support leaf** (§1a); counting captured material in `tau` *and* as a support leaf double-counts the same gain |
| `tactical:*`, `smt:*`, `search:*`, `search_support:*`, `search_refutes:*` | tactical / search evidence → support or objection leaves, or filter refutations |
| every `objection:*` / `safety:*` / `opening:*` / `strategy:*` / `king_safety:*` objection family | objections → objection leaves or filter refutations |
| `reply_*` | reply-attacks → objection leaves / filter refutations |
| `defeater:*` | consumed by residual suppression (§1e) |

`static_prior` may read the **board** for the same *kind* of positional fact a
`development:` or `center_control:` reason detects (both look at piece geometry),
but it must compute it from board state, independently — it must never read
`probe.reasons`, `probe.objections`, `probe.reply_attacks`, `probe.reason_evidence`,
`probe.objection_evidence`, `probe.reply_attack_evidence`, or `probe.score`. The
disjointness is structural: `static_prior(probe)` touches only `probe`'s board
position (reconstructible from the move applied to the root) and the move itself.

The one acceptable overlap is *category*, not *quantity*: positional geometry
appears in both `static_prior` (as a centipawn term) and as a `development:` /
`center_control:` support leaf. This is a deliberate, bounded overlap — the
positional reasons are weak (`support_strength = 1`, §1b row `k=1`), and the
prior's positional term is small relative to material. The no-double-count tests
(§8e) bound it: a position differing only in a positional feature must move `tau`
*or* add a support leaf, and the test asserts the two channels are not both
counting the *same* centipawns. Material and tactical/search/objection facts have
**zero** overlap — they are exclusively the graph's.

### 8d. Calibration corpus and `TAU_SCALE`

`TAU_SCALE` sets how fast the centipawn prior saturates into a base rate. It is
calibrated, not guessed:

- **Corpus:** the existing bench suite (`dialectical_chess/bench.py` harness) —
  the same positions Phase 2's regression gate uses. No new corpus is created.
- **Procedure:** for every bench position, compute `static_prior` for the
  engine-best move and for a clearly-worse move; choose `TAU_SCALE` so that a
  roughly-equal position maps near `tau = 0.5` and a clearly-winning residual
  (on the order of a minor piece, ~300 cp) saturates toward `tau ≈ 0.8`. A
  one-pawn (~100 cp) edge should land near `tau ≈ 0.62`. The starting value
  `TAU_SCALE = 400` satisfies this — computed against `math.tanh`:
  `squash(100) = 0.622459`, `squash(300) = 0.817574`, `squash(400) = 0.880797`,
  `squash(0) = 0.500000`, and the clamp holds (`squash(±1e6) = 0.99 / 0.01`).
  The exact value is picked by sweeping `TAU_SCALE` over the bench suite and
  taking the value that maximizes bench accuracy.
- `TAU_SCALE` is a **tuning knob, not a correctness property** — it lives in
  `tuning.py` with `EV` and `A_ROLE`.

### 8e. No-double-count tests (P2.4)

P2.4 cannot be marked complete without these:

- `test_static_prior_ignores_probe_score` — two probes with the same board
  position after the move but different `probe.score` (e.g. constructed so
  `score` differs by a sentinel term) produce the **same** `static_prior`.
- `test_static_prior_ignores_evidence_labels` — `static_prior` is invariant under
  mutating `probe.reasons` / `probe.objections` / `probe.reply_attacks` /
  `probe.*_evidence` while holding the board fixed.
- `test_material_counted_once` — a position where the move captures a piece: the
  captured material appears as a `material:capture:*` support leaf in the
  artifact, and `static_prior` for that move equals `static_prior` of the *same
  board reached without the capture being a labelled reason* — i.e. the capture's
  material gain is in the graph, not double-booked into `tau`. (Concretely: the
  prior reads post-move board material, which already reflects the capture; the
  test asserts the support leaf is the *only* place the *strength* of the capture
  is calibrated — the prior's material term and the support leaf's `k` are not
  both tuned against the same gain.)
- `test_squash_open_interval` — `squash` output is strictly inside `(0.01, 0.99)`
  for inputs across `[-1e6, +1e6]`; `Opinion.vacuous(squash(x))` never raises.
- `test_squash_monotone_and_centered` — `squash` is strictly increasing and
  `squash(0.0) == 0.5` exactly.

---

## 9. Checklist triage — the 26 properties of `reviews/01-theory-and-intent.md` §3

Of the 26 testable design properties, the opinion-valued Phase 2 **satisfies 15**,
**partially satisfies 1**, and **defers 10**.

### Satisfied by Phase 2 (15)

| # | Property | Delivered by |
|---|---|---|
| 1 | Explicit argument-graph data structure, built per position | `BipolarOpinionGraph` (§7, P2.5) |
| 2 | `bestmove` read off a semantics computation over the graph | §6 decision rule; `evaluate` |
| 3 | Objections/supports are generator-produced graph edges, not `if`s | §1 (typed-evidence → leaves/edges) |
| 6 | Attack edges typed (rebut vs. undercut) | §1 — objection leaves are rebuts; defeaters/defenses are undercuts handled by residual suppression (§1e) |
| 7 | Support edges exist, distinct from attack (bipolar) | §1a — `supports` is a distinct `frozenset` from `attacks` |
| 8 | A Dung extension is computed; non-members excluded | §5 — `grounded_extension` of the filter graph |
| 9 | Grounded = least-fixpoint iteration of `F_AF` | §5 — `argumentation.dung.grounded_extension` is the least-fixpoint computation |
| 14 | Survivors ranked by a named gradual semantics (total preorder) | §6 — `doxa.argumentation` opinion-valued CCF; `expectation()` is a total order |
| 15 | Vacuity: unargued move's strength == base score `tau` | §2 — computed exact, `E == tau` for all `tau` |
| 16 | Balance: strength < `tau` iff attacked, > `tau` iff supported | §1b / §1d — computed monotone tables either side of `tau` |
| 18 | `tau` a separate input, not identically the computed strength | §2 / §8 — `tau` is `intrinsic[move].a`; `expectation()` is the output; structurally distinct |
| 19 | All strengths bounded `[0,1]` by construction | `doxa.Opinion` enforces `b,d,u ∈ [0,1]`, `expectation() ∈ [0,1]` |
| 21 | Multiple reasons aggregated as a set; no per-subset arguments | §1a — per-role strength summation; one leaf per role; the rev-1 copy-multiplication is deleted in P2.3 |
| 23 | Cyclic graphs handled / convergence story | The graph is a strict DAG; `evaluate` is Kahn topological, raises `CyclicGraphError` if a cycle ever appears |
| 25 | AGU/APU layering — semantics layer holds no chess knowledge | §7 — `doxa.argumentation` and `argumentation.dung` are chess-agnostic; all chess knowledge is in `opinion_graph.py` / `skeptical_filter.py` |

### Partially satisfied (1)

| # | Property | Status |
|---|---|---|
| 24 | Acyclic portions handled in topological order | Satisfied for the *whole* graph (it is wholly acyclic), not "portions." `evaluate` is entirely topological (Kahn). Counted partial only because the phrasing presumes a mixed cyclic/acyclic graph; the intent is exceeded. |

### Deferred (10)

| # | Property | Why deferred / where |
|---|---|---|
| 4 | Move args instantiate AS1/AS2 with named slots | Phase 2 gives a move node a `tau` + aggregate leaf edges; full slots are schema enrichment — Phase 3 (P3.1) |
| 5 | Objections map to the closed CQ1–CQ17 catalogue, each tagged | Generator work; objection *kinds* exist but are not CQ-tagged — Phase 3 (P3.2) |
| 10 | Every argument carries a value label | **Dropped** — §4 drops the value layer; `argument_value` is not consumed |
| 11 | `defeat` = `attack` + value check; relations separate | **Dropped** — §4; no value-relative `defeat` in the opinion model |
| 12 | Value ordering configurable; changing it can change the move | **DEFERRED / NOT-SATISFIED** — §4c (Codex M2): the value layer is dropped; Phase 2 delivers no audience/value-reordering mechanism, and no tuning knob is claimed to substitute for one. A Phase-3 addition if wanted. |
| 13 | Forcing/tactical facts representable as a dominating value | **Re-assigned** — §4b/§5: the dominating-`SOUNDNESS` role is taken by the Dung skeptical hard-filter, a binary mechanism. The property's *intent* (a sound forced loss dominates) is met; its value-layer *mechanism* is dropped. |
| 17 | Strict monotonicity | Not satisfied. With per-role aggregation (§1a) adding an *independent* reason of the same role *does* strictly raise `E` (1 vs 2 vs 4 reasons → 0.775 / 0.850 / 0.910). But adding a *duplicate identical* reason does not — the engine emits each reason once, and CCF idempotence still means an exact-duplicate leaf is absorbed. Strict monotonicity in the textbook "any added supporter strictly increases strength" sense is not a Phase-2 gate; documented, accepted. |
| 20 | Full documented axiom profile (Bonzon 16 / Baroni 11) | Headline profile is here (vacuity yes, balance yes, monotone-in-aggregate-strength yes, strict-on-duplicates no); the exhaustive table is `docs/semantics.md` — Phase 2 P2.8 short version, full audit Phase 3 |
| 22 | Losing reasons retained and inspectable for explanation | The `evidence_trace` (§7a) retains every `ArgumentEvidence`; the full `Opinion` per move is on `ArgumentationDecision.move_opinion`. A dedicated inspection UI is Phase 3 |
| 26 | Two-stage categorise-then-accumulate (Besnard) | `doxa.argumentation` is single-stage (one CCF fixpoint per node). A future refinement — deferred, flagged. |

---

## 10. Plan assumptions corrected

1. **The value layer is dropped, not "built if kept."** Chunk P2.6's framing
   implied the value layer was a live possibility. §4 settles it: dropped. P2.6
   still builds the skeptical filter — read P2.6 as "build the skeptical filter;
   do not build a value layer."

2. **The skeptical filter is mandatory, not optional.** §5 shows — computed,
   under the *corrected* aggregation encoding (§5a worked example B revised) —
   that opinion `expectation()` alone ranks a queen-grab-into-forced-mate above
   sound alternatives whenever its prior is high. The filter is kept.

3. **The original worked example B's numbers are superseded.** The original note's
   `nf3 = 0.775` used the broken one-leaf-per-reason encoding; under aggregation
   `nf3`'s two reasons → `k=2 → E=0.850`. The K2 *failure* is unchanged; §5a
   gives a corrected worked example (`qxq E=0.820` beats unargued `alt E=0.500`,
   and beats a one-reason quiet move `E=0.775`).

4. **`probe.score` normalization is not used.** P2.4's `static_prior` is a
   *disjoint* re-evaluation, never a normalization of `probe.score` (§8). The
   engine-research report's §5 sigmoid-of-`probe.score` recipe is superseded by
   the §8 disjoint-prior contract.

5. **Strict monotonicity (checklist 17) is partially recovered.** The rev-1
   plan deferred it entirely. With per-role aggregation (§1a) it now holds for
   *independent* reasons (computed: 1/2/4 reasons → strictly increasing `E`); it
   fails only for *exact-duplicate* leaves under CCF idempotence. The deferral
   stands for the textbook form; the §9 entry states the corrected, narrower
   cause.

6. **`static_prior` is fully specified, not deferred.** Codex M3: the original
   note declared it out of scope. §8 of this note is the complete P2.4 contract —
   signatures, included terms, excluded label families, clamp, calibration
   corpus, no-double-count tests.

---

## 11. Locked-decision summary

| Item | Locked decision (v2) |
|---|---|
| 1. Leaf encoding | Aggregate per move per role *before* opinion construction (C1): `agg_k = Σ strengths`, one support leaf + one objection leaf per move, `leaf_intrinsic(k) = Opinion.from_evidence(k·EV, 0, 0.5)`, `EV=2.0`. `k=0` items omitted entirely — no leaf, no edge (C2). |
| 1e. Defeaters/defenses | Residual suppression (M1): `effective_objection_strength = max(0, obj_k − Σ defeater_strength)`; defeaters are not graph nodes. A fully-defeated objection → `k=0` → omitted → move restored to baseline, never boosted above it. Computed proof §1e. |
| 2. Move node | `intrinsic = Opinion.vacuous(tau)`, `tau = squash(static_prior(probe))` ∈ `[0.01,0.99]`. Unargued move `E == tau` exactly (computed). |
| 3. Edge model | Every edge `= Opinion.dogmatic_true(0.5)` (fully trusted). All grading lives in the aggregate leaf strength. |
| 4. Value layer | **Dropped.** No `values.py`. Checklist item 12 is honestly marked deferred/not-satisfied (M2) — no tuning knob is claimed to substitute. |
| 5. Skeptical filter | **Kept and mandatory.** Dung `grounded_extension` over a pure-attack filter graph of `is_forced_mate_refutation` refutations. `move ← refutation` only; **no counterdefeater edges** (C4 — forced-mate refutations are never counter-defeated in Phase 2; tested). Empty survivors → `empty_survivors=True`, fall back to all moves (Minor 3 — testable). |
| 6. Decision rule | `bestmove = argmax_{p.uci ∈ pool} (evaluate(graph)[move(p)].expectation(), p.uci)`. Tie-break: largest UCI wins, stated explicitly (Minor 4). No `probe.score`. |
| 7. Module API | **One builder** (C3): `build_argumentation_artifacts(probes) -> MoveArgumentationArtifacts` (opinion graph + `move_arg` + filter `ArgumentationFramework` + `evidence_trace`). `skeptical_survivors(artifacts)` and `choose_move_argumentation(probes)` consume the artifact; neither re-parses probes. `ArgumentationFramework(arguments=..., defeats=filter_defeats)` — `defeats=` required (M4). |
| 8. `static_prior` | Full contract locked (M3): `static_prior` / `squash` signatures, included positional+material terms, excluded `ArgumentEvidence` label families, `[0.01,0.99]` clamp, `TAU_SCALE` calibrated on the bench corpus, five no-double-count tests. |
| 9. Checklist triage | 15 satisfied, 1 partial, 10 deferred — item 12 honestly deferred, item 17 partially recovered. |

---

## 12. Codex finding → resolution map

| Finding | Resolution in this note |
|---|---|
| **C1** same-band reason collapse | §1a — aggregate evidence per move per role *before* opinion construction; one support leaf + one objection leaf per move, strength = Σ of the items' strengths. Computed: 1/2/4 `k=1` reasons → `E` = 0.775 / 0.850 / 0.910 — distinct, sane, strictly monotone. The original one-leaf-per-reason encoding (and the CCF-idempotence collapse to a flat 0.775) is removed. |
| **C2** `k=0` evidence | §1d — `build_argumentation_artifacts` omits every zero-strength item entirely (no leaf, no edge), including aggregates that sum to 0. `leaf_intrinsic` requires `strength > 0` and raises on 0; the public vacuous-leaf branch is deleted. Computed corruption (`0.725 → 0.8825` at `tau=0.55`) cannot occur because the corrupting leaf is never built. |
| **C3** one artifact builder | §7a — single `build_argumentation_artifacts(probes) -> MoveArgumentationArtifacts` carrying the `BipolarMoveGraph`, `move_arg`, the filter `ArgumentationFramework`, and an `evidence_trace`. `skeptical_survivors` and `choose_move_argumentation` consume the artifact; the separate `build_opinion_graph` / `build_skeptical_filter` / `skeptical_survivors(probes)` are removed. |
| **C4** counterdefeater policy | §5c — examined `objection_defeater_evidence` (`arguments.py:187-216`): synthesized defeaters fire only for `QUEEN_BLUNDER`, `MOVED_PIECE_EN_PRIS`, `OPENING_PREMATURE_MINOR_CHECK`, `FLANK_PAWN_WEAKENING/LUNGE` — never for `REPLY_MATE_IN_ONE`, `REPLY_FORCED_MATE`, or the `search_refutes:` mate sentinel. A forced-mate refutation is a terminal fact and is **never counter-defeatable in Phase 2**. Locked: filter graph is `move ← refutation` only, no counterdefeater edges; P2.6 includes `test_filter_graph_has_no_counterdefeater_edges`. |
| **M1** attack-on-attack becomes positive belief | §1e — defeating an objection is encoded as residual strength suppression, *not* an attack-on-attack edge. Computed: `[objection fully defeated]` lands *exactly* on the no-objection baseline across `tau ∈ {0.3,0.5,0.7}`; a partially-defeated objection sits strictly between full-objection and baseline; monotone in defeater strength; **never above baseline**. The original encoding (move with `k=1` objection + `k=17` defeater → `E=0.722 > 0.5` baseline) is removed. |
| **M2** value-layer drop vs checklist 12 | §4c / §9 — the claim that `EV`/`TAU_SCALE` satisfy audience/value reordering is withdrawn as false. Checklist item 12 is marked **deferred / not-satisfied**, consistent with the value-layer drop. The P2.8 exit gate records it as deferred. No false claim remains. |
| **M3** `static_prior` contract | §8 — full `static_prior.py` contract: `static_prior` / `squash` signatures, included positional+material terms, the excluded `ArgumentEvidence` label families (material/tactical/search/objection/reply/defeater — the no-double-count rule), the `[0.01,0.99]` clamp, `TAU_SCALE` calibrated on the bench corpus, and five no-double-count tests. P2.5 is no longer blocked. |
| **M4** Dung constructor | §5b / §7 — `ArgumentationFramework(arguments=..., defeats=filter_defeats)`; `defeats=` is required (computed: `attacks=`-only raises `TypeError: missing 1 required positional argument: 'defeats'`). `grounded_extension` reads `defeats`. `attacks=` is optional metadata only. |
| **Minor 1** stale defeater numbers | §1b — corrected: `k=33 → E=0.985294`, `k=97 → E=0.994898` (computed against real `doxa`). The "≈0.970 / ≈0.990" text is gone. |
| **Minor 2** P2.7 sketch vs API | §7b — reconciled: both the plan sketch's `skeptical_survivors(bmg)` and the original `skeptical_survivors(probes)` are replaced by the artifact-based `skeptical_survivors(artifacts)`. |
| **Minor 3** empty-survivor fallback | §5d — `ArgumentationDecision.empty_survivors: bool` (a structured field, not a log note); two required tests (`test_empty_survivors_when_every_move_hard_refuted`, `test_over_filtered_position_detected`). |
| **Minor 4** tie-break rule | §6 — stated explicitly and consistently: `max(..., key=(expectation, p.uci))` selects the **lexicographically largest UCI** on an exact tie. The incorrect comparison to `doxa`'s smallest-name-first traversal is removed. |

### Codex "Keep" items — retained unchanged

- Positive one-sided leaf encoding for objections (the objection leaf believes
  its own claim; `evaluate` negates attackers at the target) — kept (§1).
- Fully-trusted edges for Phase 2 — kept (§3).
- The skeptical filter — kept and shown mandatory under the corrected encoding (§5).
- The no-`probe.score` decision rule; `tau` from a disjoint `static_prior` — kept
  and fully specified (§6, §8).

---

Every numeric claim in this note was produced by throwaway scripts run against
the real `doxa` package at HEAD `f076502` and the real `argumentation` package.
The scripts were deleted, not committed. No production code was written; no file
under `dialectical_chess/` or `tests/` was created or modified; no commit was
made.
