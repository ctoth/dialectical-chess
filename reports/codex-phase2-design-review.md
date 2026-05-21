# Codex Phase 2 design review

Verdict: the load-bearing `doxa` computations in `reports/argdriven-phase2-design.md` hold, including the decisive queen-grab example, so the skeptical filter is genuinely needed. The design is not structurally doomed, but it is not sound to implement unchanged: the filter consistency promise is stronger than the API actually enforces, hard-filter counterdefeat/false-positive behavior is underspecified, the value-layer drop silently changes a stated theory property, and the static-prior contract remains a load-bearing blank delegated to P2.4. Fix those before P2.2/P2.5 coding.

## Computation check

Re-run against the real `doxa` checkout at `C:\Users\Q\code\doxa` with `uv run python` and `doxa.argumentation.evaluate`:

- Unargued move: `Opinion.vacuous(tau)` resolves to `E == tau` for `tau = 0.1, 0.3, 0.5, 0.7, 0.9`. This matches the design claim at `reports/argdriven-phase2-design.md:239-257` and follows from `evaluate`/`_accrue` returning `Opinion.vacuous(tau_x)` on an empty pool (`C:\Users\Q\code\doxa\src\doxa\argumentation.py:242-246`).
- Intrinsic family with `EV=2.0`, `s=0`, `a=0.5`: `k=1 -> E=0.750000`, `k=9 -> E=0.950000`, `k=17 -> E=0.972222`. The monotone table at `reports/argdriven-phase2-design.md:98-117` is correct. The constructor path is `Opinion.from_evidence` and `expectation` (`C:\Users\Q\code\doxa\src\doxa\opinion.py:104-121`).
- Dogmatic mate objection: alone gives `E=0.000000`; with one `k=9` supporter at `tau=0.75`, the move resolves to `b=0.000000, d=0.100000, u=0.900000, E=0.675000`. This confirms `reports/argdriven-phase2-design.md:384-397`.
- Queen-grab example: `move:qxq` with `tau=0.85`, `k=9` support, `k=6` mate objection resolves to `b=0.128571, d=0.085714, u=0.785714, E=0.796429`; `move:nf3` with `tau=0.55` and two `k=1` reasons resolves to `E=0.775000`. The queen grab beats the quiet move, confirming `reports/argdriven-phase2-design.md:399-409`.
- Weak-objection value example: `k=9` support plus `k=1` objection at `tau=0.6` gives `E=0.750000`; dropping the objection gives `E=0.960000`. This confirms `reports/argdriven-phase2-design.md:315-332`.
- Dung filter behavior with the actual `formal-argumentation` API: `defeats={obj:mate -> move:qxq}` gives grounded extension `{obj:mate, move:nf3}`; adding `def:mate -> obj:mate` gives `{def:mate, move:qxq, move:nf3}`.

## Critical

1. Filter/opinion consistency is promised but not enforced by the locked API.
   Location: `reports/argdriven-phase2-design.md:444-453`, `reports/argdriven-phase2-design.md:538-600`.
   The design says the filter graph and opinion graph are built "in one pass from the same `MoveProbe` list, by the same builder", but the API actually exposes separate `build_opinion_graph(probes)` and `skeptical_survivors(probes)` calls. That lets P2.5 and P2.6 classify evidence independently and drift. Concrete fix: replace the two independent builders with one artifact builder, e.g. `build_argumentation_artifacts(probes) -> MoveArgumentationArtifacts`, containing the `BipolarOpinionGraph`, `move_arg`, filter AF, and a trace table from each `ArgumentEvidence` to opinion/filter nodes. `skeptical_survivors` should consume that artifact, not re-parse probes.

2. The hard filter has no specified counterdefeater policy.
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

## Keep

- Keep the positive one-sided leaf encoding for objections. The polarity is correct: the objection leaf believes its own claim, and `evaluate` negates attackers at the target (`C:\Users\Q\code\doxa\src\doxa\argumentation.py:227-231`).
- Keep fully trusted edges for Phase 2. The doxa computations confirm leaf strength alone gives the intended monotone family, and avoiding a second strength knob is a good constraint.
- Keep the skeptical filter. The queen-grab computation confirms expectation-only ranking reintroduces the K2 failure.
- Keep the no-`probe.score` decision rule. The plan correctly requires a disjoint prior because `MoveProbe.score` is an accumulated mix of the same facts that become evidence nodes (`reports/argdriven-research-engine.md:21-23`, `reviews/PLAN-argumentation-driven.md:482-487`).
