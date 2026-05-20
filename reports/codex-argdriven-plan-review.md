# Codex adversarial review of `reviews/PLAN-argumentation-driven.md`

Overall verdict: the plan is directionally right but **not ready to execute as written**. DF-QuAD is a defensible replacement for quadratic energy if the new graph preserves the proved DAG layering, and moving `probe.score` into `tau` can be legitimate. The current plan still has two structural defects: P2.4 computes DF-QuAD over the raw attack graph after P2.3 computes value-filtered `defeats`, contradicting the design record that both grounded and gradual stages run on the `defeats` graph; and P2.2 derives `tau` from the aggregate HCE `probe.score`, which already includes the same tactical/material/positional facts that P2 also turns into graph nodes. Fix those before execution; otherwise the rewrite risks becoming a cleaner HCE-with-argumentation pipeline rather than an argumentation-driven decision.

## Critical

### C1. P2.4 ranks on raw attacks, not the value-filtered defeat graph

Location: `reviews/PLAN-argumentation-driven.md:481-493`, `reviews/PLAN-argumentation-driven.md:511-516`; contradicted by `reports/argdriven-research-theory.md:187-194` and `reports/argdriven-research-theory.md:235-237`.

P2.3 correctly defines `defeats_of(...)` by value-filtering `bmg.graph.attacks`, and uses those defeats for `grounded_extension`. P2.4 then calls `dfquad_strengths(bmg.graph)`, so the gradual score still sees **raw attacks**, including value-dispreferred attacks that Bench-Capon says failed to defeat. That means a low-priority `TEMPO` objection can be barred from filtering a `MATERIAL` move but still lower its `sigma`, splitting the Dung and gradual layers.

Concrete fix: after `defeats = defeats_of(...)`, build the strength graph as a `WeightedBipolarGraph` with the same arguments, initial weights, and supports, but `attacks=defeats`. Run both `grounded_extension` and `dfquad_strengths` on that defeat graph. Add a test where a value-dispreferred attack neither removes the move nor lowers its `sigma`; then flip the audience order and prove both can change.

### C2. `tau` is defined from a double-counted, non-componentized HCE aggregate

Location: `reviews/PLAN-argumentation-driven.md:420-423`, `reviews/PLAN-argumentation-driven.md:526-528`; score provenance at `reports/argdriven-research-engine.md:370-403` and normalization warning at `reports/argdriven-research-engine.md:433-453`.

The plan says `probe.score` appears once as `tau`, with mate sentinels stripped. But `probe.score` is not a clean static prior: it is a single accumulated integer containing checkmate, captures, promotion, moved-piece safety, tactical threats, opening penalties, reply-mate penalties, positional reason counts, SMT witnesses, search result, repetition penalty, and forced-reply-mate scan results. Many of those same facts become support or objection nodes in P2.2. Only stripping mate sentinels leaves captures, queen hangs, flank-pawn penalties, search refutations, and positional reasons counted once in `tau` and again through graph structure.

It is also not reliably strippable after aggregation. Once the sentinels and ordinary scores are summed into one `int`, the plan cannot prove which part of `probe.score` was a terminal fact and which was residual prior.

Concrete fix: change the plan to create a componentized score surface before P2.2, e.g. `MoveProbe.static_prior_score` or `ScoreBreakdown`, where terminal/tactical facts that become graph nodes are excluded or explicitly marked. `squash_score` must consume that residual prior, not raw `probe.score`. Add tests proving a forced mate affects `sigma` through graph nodes, not through `tau`, and that removing a support/objection component from the graph removes its only decision effect.

### C3. The Phase 2 hard gate is subjective and under-specified

Location: `reviews/PLAN-argumentation-driven.md:580-583`, `reviews/PLAN-argumentation-driven.md:606-611`.

The plan allows every changed ablation move to be "justified as correct/better" but does not define the evidence artifact, evaluator, or minimum proof. It also says "no Stockfish-2000 strength regression" without a game count, confidence threshold, fixed seed/openings, or acceptable draw/loss-on-time/error policy. This is not a hard gate yet; it is reviewer judgment plus a noisy match headline.

Concrete fix: require a checked-in triage report for P2.5/P2.6 with old move, new move, old/new `sigma`, grounded-survivor status, value-order result, tactical oracle result where applicable, and a signed verdict per changed test. For the match gate, specify the runner, openings, time control, seed, number of games or SPRT/equivalence bound, and fail-closed treatment of crashes/time losses. Require the tactical EPD gate in open question 7.

## Major

### M1. P2.5 is too large to be a single executable chunk

Location: `reviews/PLAN-argumentation-driven.md:538-579`, `reviews/PLAN-argumentation-driven.md:664-672`.

P2.5 deletes the decider, selector modes, old graph builder, copy-multiplication, old value framework glue, `EngineSettings.selector_mode`, bench axes, UCI/probe/match/summary plumbing, and possibly rewrites or deletes `optimizer.py`. That is more than an atomic cutover; it is several independent API deletions plus the riskiest behavior change. If it fails, the failure surface will be too wide to diagnose cleanly.

Concrete fix: keep the "no two production deciders" invariant, but make the checklist mechanically smaller: delete `selector_mode` fan-out as an earlier production-surface simplification while preserving the current single default behavior, land optimizer deletion/rewire as its own explicitly gated chunk, then make P2.5 only "old default decision path out, new argumentation decision path in." If the protocol requires one commit per chunk, redefine the chunks rather than packing all of this into one chunk.

### M2. The plan schedules thesis tests after most of the implementation

Location: `reviews/PLAN-argumentation-driven.md:589-603`.

P2.6 creates the tests that prove "argumentation decides" after P2.1-P2.5 build and cut over the new pipeline. That leaves the early implementation slices without the red tests that define the target behavior.

Concrete fix: move the core `tests/test_argumentation_thesis.py` skeleton before P2.1 as failing tests: graph edge construction from real `probe_moves`, value-dispreferred attack behavior, grounded survivor filtering, `sigma` ranking on the defeat graph, and a position where argumentation diverges from plain `probe.score`. P2.6 can keep documentation and final principle-table rerun.

### M3. P0.2 depends on a future test file

Location: `reviews/PLAN-argumentation-driven.md:168-183`.

P0.2 says its test is added to `tests/test_board_differential.py`, which is not created until P1.1. A Phase 0 chunk cannot depend on a future Phase 1 file for its own proof.

Concrete fix: either create the minimal differential test file in P0.2, or add the exact `1.b3 h5` FEN serialization test to an existing test file in P0.2 and let P1.1 expand it into the full oracle harness.

### M4. Value mapping remains underspecified where it matters most

Location: `reviews/PLAN-argumentation-driven.md:392-408`; current parsing concern at `reviews/07-directional.md:14-15`.

P2.1 says `positional` maps to `KING_SAFETY`/`SPACE`/etc. "by prefix." That is still a string-label contract unless the plan also moves the producer to emit typed value labels. The plan does not enumerate every current `argument_value`/label prefix and the exact `Value` it maps to.

Concrete fix: require an exhaustive mapping table covering every label family emitted by `probe.py`, with unknown labels failing closed. If Phase 2 still consumes string labels, say that plainly and add a P3 deletion target; do not imply labels are display-only until `probe.py` emits typed values directly.

### M5. Some review findings are not owned by any chunk

Locations: `reviews/04-substrate.md:153-168`, `reviews/06-tests.md:177-188`, `reviews/03-decision-pipeline.md:305-314`.

The plan covers the headline K/J findings, but it does not clearly own `OwnedBoard.apply`'s unsafe public behavior, malformed parser/error-path tests, or the unguarded `has_search_refutation_at_most` parse path. These are not central to the argumentation rewrite, but they are substrate risks in the same reviewed surface.

Concrete fix: add them explicitly to P3.6 or a separate Phase 0/P1 cleanup chunk: document or add `apply_checked`, add malformed FEN/UCI parser tests, and harden `search_refutes:` score parsing.

## Minor

### m1. P2.4 should assert the DF-QuAD mode it relies on

Location: `reviews/PLAN-argumentation-driven.md:524-528`; library behavior at `reports/argdriven-research-library.md:189-197`.

The plan has an acyclicity assertion in P2.2, but P2.4 should still assert `result.converged` and `result.integration_method == "dfquad_topological"` in tests. If the graph ever silently reaches `dfquad_fixed_point`, the engine should fail a correctness test before a match gate.

### m2. `tanh` is acceptable, but the scale must be calibrated before it becomes a gate

Location: `reviews/PLAN-argumentation-driven.md:420-423`.

`0.5 + 0.5*tanh(score/S)` is equivalent in shape to a logistic sigmoid and is defensible. The risk is not `tanh`; it is choosing `S` without distribution data. Add a Phase 1 measurement that records the residual-prior distribution after sentinel/evidence stripping, then set `TAU_SCALE` from that distribution.

### m3. The report should stop saying "no pin bump" only because the modules exist

Location: `reports/argdriven-research-library.md:51-78`.

The library report correctly proves the pinned modules and APIs exist. The plan should additionally pin the exact import names in P2.2/P2.4 tests, because `argumentation.__init__` re-exports modules, not symbols (`reports/argdriven-research-library.md:82-106`).

## Direct answers to the 7 open questions

1. **P0.3 — Z3 divergence:** no meaningful position class was found where the SAT layer can add a correct result the direct Python checks would miss. In `smt_mate_in_one_moves`, candidate moves are already produced by `owned_is_checkmate(board.apply(move))`, then post-verified by `verifies_mate_in_one`; the solver can only return sat/unsat/unknown over that candidate set. In `smt_fork_witnesses`, `fork_witness_after` already computes the candidates and the solver returns all candidates unchanged. Divergence risk is only degradation: `ImportError` or `unknown` can drop true witnesses. Delete the SAT round trip and keep direct tests.

2. **P0.4 — draw claiming:** keep Phase 0 evaluation-only. UCI has no clean standard "claim draw now" command for the engine to emit in place of a legal move, and custom claim signaling would broaden the protocol scope. The engine should evaluate fifty-move/threefold terminal states correctly and choose drawing moves when appropriate; match/GUI layers can adjudicate or claim.

3. **P1.3 — time-budget formula:** use `movetime` when present: `budget_ms = max(10, movetime - overhead_ms)`. Otherwise use `remaining`, side increment, and `movestogo`: `moves = movestogo or 30`; `base = remaining / moves`; `budget = base + 0.75 * increment - overhead_ms`; clamp to `[10, min(remaining - reserve_ms, remaining * 0.20)]`, with `reserve_ms = max(100, min(1000, 0.03 * remaining))`. If the clamp leaves no positive time, return the best currently available move immediately. This is conservative and avoids spending a huge fraction of the clock on one move.

4. **P2.1/P2.4 — 6-value VAF vs. 2-tier:** keep the 6-value VAF in Phase 2. A 2-tier `SOUNDNESS` vs. rest model would satisfy the hard safety gate but would not prove checklist item 12: that soft value ordering is load-bearing. The fix is to make the mapping exhaustive and fail-closed, not to defer soft values.

5. **P2.5 — `optimizer.py`:** keep it only as a separate research CLI that consumes the same `BipolarMoveGraph` and cannot affect production `choose_move`. Remove it from `EngineSettings.selector_mode` and benchmark selector axes. If rewriting it to the new graph is nontrivial, delete it in P2.5 and reintroduce a new optimizer later.

6. **DF-QuAD vs. quadratic energy:** DF-QuAD is correct for the current and proposed layered graph **if** P2.2 enforces acyclicity and P2.4 asserts topological DF-QuAD execution. Prefer quadratic energy only after an intentional feature introduces cycles. The plan must also fix C1: DF-QuAD should run over the value-filtered defeat graph, not raw attacks.

7. **Phase 2 gate:** no, "no Stockfish-2000 regression" is not enough. Require it, but also require a focused tactical/argumentation EPD gate with measured improvement or at least strict non-regression on the K2 classes: forced mate, queen hang, reply-mate refutation, material safety, and value-order discrimination. The argumentation rewrite should prove it fixed the failure class it targets, not merely survive a noisy match.

## Keep as-is

- Phase 0 before Phase 1 before Phase 2 is justified: FEN corruption and fail-open tooling would contaminate measurement.
- DF-QuAD is the right default once acyclicity is enforced; the library provides the needed APIs at the pinned SHA.
- Deleting selector-mode fan-out and ending with one production decider is the right final state.
- The plan is right to make fail-closed match/EPD tooling a prerequisite for strength gates.
- Bringing `tests/test_argumentation_thesis.py` into Phase 2 is necessary; it just needs to move earlier inside Phase 2.
