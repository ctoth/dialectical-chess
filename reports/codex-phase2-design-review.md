# Codex Phase 2 design review

Verdict: the load-bearing `doxa` computations in `reports/argdriven-phase2-design.md` hold, including the decisive queen-grab example, so the skeptical filter is genuinely needed. The design is not structurally doomed, but it is not sound to implement unchanged: the one-leaf-per-reason encoding collapses identical evidence under CCF, `k=0` leaves are not safely "dropped" once connected by edges, the filter consistency promise is stronger than the API enforces, hard-filter counterdefeat/false-positive behavior is underspecified, the value-layer drop changes a stated theory property, and the static-prior contract remains a load-bearing blank delegated to P2.4. Fix those before P2.2/P2.5 coding.

## Computation check

Re-run against the real `doxa` checkout at `C:\Users\Q\code\doxa` with `uv run python` and `doxa.argumentation.evaluate`:

- Unargued move: `Opinion.vacuous(tau)` resolves to `E == tau` for `tau = 0.1, 0.3, 0.5, 0.7, 0.9`. This matches the design claim at `reports/argdriven-phase2-design.md:239-257` and follows from `evaluate`/`_accrue` returning `Opinion.vacuous(tau_x)` on an empty pool (`C:\Users\Q\code\doxa\src\doxa\argumentation.py:242-246`).
- Intrinsic family with `EV=2.0`, `s=0`, `a=0.5`: `k=1 -> E=0.750000`, `k=9 -> E=0.950000`, `k=17 -> E=0.972222`. The monotone table at `reports/argdriven-phase2-design.md:98-117` is correct. The constructor path is `Opinion.from_evidence` and `expectation` (`C:\Users\Q\code\doxa\src\doxa\opinion.py:104-121`).
- Dogmatic mate objection: alone gives `E=0.000000`; with one `k=9` supporter at `tau=0.75`, the move resolves to `b=0.000000, d=0.100000, u=0.900000, E=0.675000`. This confirms `reports/argdriven-phase2-design.md:384-397`.
- Queen-grab example: `move:qxq` with `tau=0.85`, `k=9` support, `k=6` mate objection resolves to `b=0.128571, d=0.085714, u=0.785714, E=0.796429`; `move:nf3` with `tau=0.55` and two `k=1` reasons resolves to `E=0.775000`. The queen grab beats the quiet move, confirming `reports/argdriven-phase2-design.md:399-409`.
- Weak-objection value example: `k=9` support plus `k=1` objection at `tau=0.6` gives `E=0.750000`; dropping the objection gives `E=0.960000`. This confirms `reports/argdriven-phase2-design.md:315-332`.
- Dung filter behavior with the actual `formal-argumentation` API: `defeats={obj:mate -> move:qxq}` gives grounded extension `{obj:mate, move:nf3}`; adding `def:mate -> obj:mate` gives `{def:mate, move:qxq, move:nf3}`.
- Identical reason collapse: at `tau=0.55`, one `k=1` support, two `k=1` supports, and four `k=1` supports all resolve to `E=0.775000`. The design already notes CCF idempotence at `reports/argdriven-phase2-design.md:62-64`, but it does not apply that fact to the one-leaf-per-reason encoding.
- `k=0` connected leaves are not generally harmless. In support-only cases they were neutral in my checks, but in mixed conflict they changed the result: `supports=[9], attacks=[1]` gave `E=0.725000`, while `supports=[9,0], attacks=[1]` gave `E=0.882500`. So the builder must skip zero-strength evidence rather than emit vacuous leaves.

## Critical

1. The one-leaf-per-reason encoding cannot count same-band reasons.
   Location: `reports/argdriven-phase2-design.md:72-96`, `reports/argdriven-phase2-design.md:201-202`, `reports/argdriven-phase2-design.md:640`; CCF implementation at `C:\Users\Q\code\doxa\src\doxa\opinion.py:508-533`, `C:\Users\Q\code\doxa\src\doxa\opinion.py:587-714`.
   The design maps each typed evidence item to its own leaf, then relies on CCF accrual. CCF is idempotent on identical opinions, and many engine reasons share exactly the same `k` and `a`. Re-run result: one, two, or four `k=1` positional supports all produce `E=0.775000` at `tau=0.55`. That means the Phase 2 graph cannot distinguish a move with one weak positional reason from a move with several independent weak positional reasons. Concrete fix: aggregate evidence counts before opinion construction, e.g. one support leaf per move/role with `r = sum(k_i * EV)`, or a documented evidence-bag node that sums Beta evidence for explainability while presenting one opinion source to CCF.

2. `k=0` evidence must be skipped, not encoded as a connected vacuous leaf.
   Location: `reports/argdriven-phase2-design.md:116-121`, `reports/argdriven-phase2-design.md:530-535`; doxa source-pool construction at `C:\Users\Q\code\doxa\src\doxa\argumentation.py:220-240`.
   Model C drops a node's own vacuous intrinsic only when evaluating that node. It does not drop a vacuous child opinion arriving over a support or attack edge; edge sources are appended before CCF. Re-run result: adding a `k=0` support source to a mixed `k=9` support plus `k=1` attack changed `E` from `0.725000` to `0.882500`. Concrete fix: make `build_opinion_graph` omit every zero-strength evidence item entirely: no leaf, no edge. Delete the `leaf_intrinsic(0) -> Opinion.vacuous(A_ROLE)` API branch or mark it private/testing-only.

3. Filter/opinion consistency is promised but not enforced by the locked API.
   Location: `reports/argdriven-phase2-design.md:444-453`, `reports/argdriven-phase2-design.md:538-600`.
   The design says the filter graph and opinion graph are built "in one pass from the same `MoveProbe` list, by the same builder", but the API actually exposes separate `build_opinion_graph(probes)` and `skeptical_survivors(probes)` calls. That lets P2.5 and P2.6 classify evidence independently and drift. Concrete fix: replace the two independent builders with one artifact builder, e.g. `build_argumentation_artifacts(probes) -> MoveArgumentationArtifacts`, containing the `BipolarOpinionGraph`, `move_arg`, filter AF, and a trace table from each `ArgumentEvidence` to opinion/filter nodes. `skeptical_survivors` should consume that artifact, not re-parse probes.

4. The hard filter has no specified counterdefeater policy.
   Location: `reports/argdriven-phase2-design.md:422-441`; prompt asks this explicitly. The current text includes hard refuters but not any hard-filter edges that can defeat a refuter. In real Dung behavior, adding `def -> obj:mate` makes `move:qxq` survive; omitting that edge permanently filters the move before the opinion graph can consider the defense. Concrete fix: either prove and state that `is_forced_mate_refutation` arguments are never counterdefeatable in Phase 2, then assert that in tests, or include a closed list of filter-level counterdefeaters and their `defeats` edges. Do not leave it to coder judgment.

## Major

1. Attack-on-attack does more than "blunt" an objection; it becomes positive support for the move.
   Location: `reports/argdriven-phase2-design.md:150-159`; `C:\Users\Q\code\doxa\src\doxa\argumentation.py:227-231`.
   Re-run result: a move with only a `k=1` objection and a `k=17` defeater of that objection resolves to `E=0.722222`; with `k=97` defeater it resolves to `E=0.744898`. That is not neutralization. The defeated objection is negated by its attack edge into belief for the move. Concrete fix: either document "defeating an objection is affirmative support" and add thesis tests for it, or change the encoding so defeaters suppress/dampen objection evidence rather than converting it into support.

2. Dropping the value layer does not satisfy the stated value-ordering property.
   Location: `reports/argdriven-phase2-design.md:348-359`, `reports/argdriven-phase2-design.md:656-659`; theory requirement at `reviews/01-theory-and-intent.md:104`.
   Replacing Bench-Capon value order with global tuning parameters is not equivalent: a value ordering is audience-relative per argument value, while `EV` and `TAU_SCALE` are global calibration knobs. Concrete fix: either keep a minimal value/audience layer for the exact checklist-12 property, or explicitly revise the checklist and P2.8 gates so Phase 2 no longer claims that audience/value reordering is satisfied.

3. The static prior is still a load-bearing design gap.
   Location: `reports/argdriven-phase2-design.md:229-237`, `reports/argdriven-phase2-design.md:551-552`; plan requirement at `reviews/PLAN-argumentation-driven.md:482-507`.
   The design-lock says `tau = squash(static_prior(probe))` but then declares `static_prior` out of scope. Because `tau` decides unargued moves and interacts directly with high-uncertainty conflict, P2.5 cannot be implemented safely without the exact no-double-count vocabulary and term list. Concrete fix: add a `static_prior.py` contract section now: function signatures, included terms, excluded `ArgumentEvidence` labels, clamp behavior, calibration corpus, and no-double-count tests.

4. The skeptical-filter API uses the wrong Dung constructor vocabulary unless corrected.
   Location: `reports/argdriven-phase2-design.md:580-600`; actual package at `C:\Users\Q\code\argumentation\src\argumentation\dung.py:19-38`, `C:\Users\Q\code\argumentation\src\argumentation\dung.py:207-218`.
   `ArgumentationFramework` requires `defeats`, with optional `attacks`; passing only `attacks=` raises `TypeError`. `grounded_extension` ignores attack metadata and uses `framework.defeats`. Concrete fix: specify `ArgumentationFramework(arguments=..., defeats=filter_defeats)` and, if metadata is wanted, `attacks=filter_defeats`.

## Minor

1. Defeater numeric examples are understated.
   Location: `reports/argdriven-phase2-design.md:150-154`.
   Re-run values are `k=33 -> E=0.985294` and `k=97 -> E=0.994898`, not approximately `0.970` and `0.990`. Fix the text so implementers do not calibrate from stale numbers.

2. The P2.7 sketch and the design API disagree on the filter input.
   Location: `reviews/PLAN-argumentation-driven.md:557-568` versus `reports/argdriven-phase2-design.md:566-577`.
   The plan sketch passes `skeptical_survivors(bmg)`; the design passes `skeptical_survivors(probes)`. Concrete fix: make this one artifact-based call as in Critical issue 1.

3. The empty-survivor fallback needs a testable policy, not just a log note.
   Location: `reports/argdriven-phase2-design.md:455-463`.
   Falling back to all moves is reasonable in lost positions, but it can also mask an over-broad filter. Concrete fix: require the decider result to expose `empty_survivors=True`, and add a test where every move is hard-refuted plus a test where an over-filtered constructed position fails.

4. The deterministic tie-break contradicts the stated "smallest-name-first" rationale.
   Location: `reports/argdriven-phase2-design.md:478-487`.
   `max(..., key=(expectation, p.uci))` chooses the largest UCI string on equal expectation, while the text compares it to doxa's smallest-name-first traversal. Exact ties are common because identical same-band reasons collapse under CCF. Concrete fix: either state "largest UCI wins" as the intended deterministic rule, or use a smallest-UCI tiebreak explicitly.

## Keep

- Keep the positive one-sided leaf encoding for objections. The polarity is correct: the objection leaf believes its own claim, and `evaluate` negates attackers at the target (`C:\Users\Q\code\doxa\src\doxa\argumentation.py:227-231`).
- Keep fully trusted edges for Phase 2. The doxa computations confirm leaf strength alone gives the intended monotone family, and avoiding a second strength knob is a good constraint.
- Keep the skeptical filter. The queen-grab computation confirms expectation-only ranking reintroduces the K2 failure.
- Keep the no-`probe.score` decision rule. The plan correctly requires a disjoint prior because `MoveProbe.score` is an accumulated mix of the same facts that become evidence nodes (`reports/argdriven-research-engine.md:21-23`, `reviews/PLAN-argumentation-driven.md:482-487`).
