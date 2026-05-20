# Typed Argumentation Evidence Workstream

## Requested Outcome

Make dialectical chess choose through typed argumentation evidence instead of string-prefix graph semantics, then continue the Stockfish 2000 improvement loop from the cleaned engine.

## Final State

- `MoveProbe` exposes typed evidence tuples for reasons, objections, and reply attacks at the probe boundary.
- `build_root_argument_graph()` derives support, objections, defeaters, values, and pressure from `ArgumentEvidence` fields.
- String labels remain only as stable display/serialization identifiers and graph node IDs.
- `arguments.py` does not parse objection labels to decide objection severity, defeaters, argument values, reply-attack strength, or defense strength.
- The old objection escape-hatch helpers are deleted from `arguments.py`: `severe_objection_weight`, `is_moved_minor_or_major_en_pris`, `has_search_support`, `has_advanced_flank_pawn_response`, `objection_defeaters` as string-prefix logic, `extra_support_copies`, `extra_defeater_copies`, `extra_objection_copies`, `extra_defense_copies`, and `extra_reply_attack_copies`.
- The selector uses categoriser scores as the primary argument-mode decider.
- Missing `argumentation` imports remain hard failures.

## Owner Boundaries

- Owned source files: `dialectical_chess/evidence.py`, `dialectical_chess/arguments.py`.
- Owned tests: focused argument/evidence tests under `tests/test_dialectical_chess_evidence_ablation.py`, plus any narrow regression added for typed objection graph behavior.
- Diagnostic outputs under `scratch/` and generated `config.json` are not committed.
- Untracked folders `notes/`, `papers/`, `prompts/`, and `pyghidra_mcp_projects/` are outside this workstream.

## Ordered Phases

1. Define typed objection and defeater classification in `ArgumentEvidence`.
2. Attach typed evidence tuples to `MoveProbe` during construction.
3. Rewrite graph construction to consume typed evidence fields and generate only opaque node IDs from labels.
4. Delete string-prefix objection and defeater logic from `arguments.py`.
5. Add or update tests proving objection defeat is represented as graph edges from typed evidence.
6. Run focused type and test gates, then commit the source slice.
7. Run a current Stockfish 2000 fast-chess gate and mine losses to choose the next chess-strength slice.

## Old-Path Search Gates

- `rg -n -F "objection.startswith" dialectical_chess/arguments.py` returns no matches.
- `rg -n -F "reason.startswith(\"search_support" dialectical_chess/arguments.py` returns no matches.
- `rg -n -F "extra_defeater_copies" dialectical_chess/arguments.py` returns no matches.
- `rg -n -F "extra_objection_copies" dialectical_chess/arguments.py` returns no matches.
- `rg -n -F "severe_objection_weight" dialectical_chess/arguments.py` returns no matches.

## Verification Gates

- `uv run pyright`
- `uv run pytest --timeout=90 tests/test_dialectical_chess_evidence_ablation.py -k "evidence or objection or compensating or premature_minor_check or flank_pawn or search_refutes"`
- `uv run pytest --timeout=90 tests/test_dialectical_chess_engine_api.py -k "clock or lower_depth or depth_zero"`
- After the commit: one 2-game Stockfish 2000 UCI match at `30+0.2`, no timeouts, crashes, or losses on time.
