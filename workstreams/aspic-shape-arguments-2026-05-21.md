# ASPIC+ Shape Arguments Workstream

## Requested Outcome

Make the castled-king pawn shell a structured ASPIC+ argument spliced directly
into the root argument graph, so its subchunks (the individual shield pawns and
the king square) are first-class subarguments that the grounded extension and
categoriser rank independently. The structured reasoning replaces the
hand-coded flank-pawn objection heuristics and is allowed to change move
selection where it reasons better than the old prefixes.

## Final State

- A new `dialectical_chess/shapes.py` builds a per-move ASPIC+ theory for the
  castled-king pawn shell from `(parent_board, move, child_board)` and solves it
  via `argumentation.aspic.build_abstract_framework`.
- The shell is a defeasible rule `R_shell` whose antecedents are the shield-pawn
  and king-square premises; applying it yields the shape argument, and its
  premise subarguments are its subchunks.
- A shield-pawn push contributes a premise that is a contrary of the matching
  `pawn_home(sq)` subchunk, so the weakening is an ASPIC+ undermine; a two-square
  lunge additionally rebuts `shell_intact`; an opponent flank-pawn threat and a
  response move argue over the same subchunk.
- `MoveProbe` carries a `shape_projection` field; `build_root_argument_graph()`
  splices the projection's arguments and defeats into the root framework under
  per-move namespaced IDs, with the surviving undermining argument attacking
  `move:<uci>` and a fully defended `king_safe` attacking `doubt:<uci>`.
- Shape graph nodes receive `ArgumentEvidence` built directly from the ASPIC+
  conclusion, not from label-prefix parsing, and map to the `positional` value.
- The hand-coded flank-pawn helpers are deleted from `probe.py`:
  `flank_pawn_weakening_objections`, `advanced_flank_pawn_response_labels`, and
  `advanced_flank_pawn_threats`.
- The flank-pawn `ObjectionKind` members (`FLANK_PAWN_WEAKENING`,
  `CASTLED_FLANK_PAWN_WEAKENING`, `FLANK_PAWN_LUNGE`,
  `UNANSWERED_ADVANCED_FLANK_PAWN`) and `DefeaterKind.ADVANCED_FLANK_PAWN_RESPONSE`
  are removed once the structured path supersedes them.
- No new dependency: `argumentation.aspic` ships in the already-pinned
  `formal-argumentation` revision. Missing `argumentation` imports stay hard
  failures.

## Owner Boundaries

- Owned source files: `dialectical_chess/shapes.py` (new),
  `dialectical_chess/probe.py`, `dialectical_chess/arguments.py`,
  `dialectical_chess/evidence.py`.
- Owned tests: `tests/test_shape_arguments.py` (new), plus narrow regressions in
  `tests/test_dialectical_chess_evidence_ablation.py`.
- `queen_flank_invasion_objections` and non-king shapes (outposts, IQP) are out
  of scope.
- ADF-based non-monotone shell conditions are out of scope and noted as future
  work.
- Diagnostic outputs under `scratch/` and generated `config.json` are not
  committed.

## Ordered Phases

1. P2.1: commit this workstream document.
2. P2.2: add `shapes.py` with the kingside-castle shell theory builder and
   solver; unit-test the projection in isolation (subchunk count, undermine on a
   push, `king_safe` IN/OUT).
3. P2.3: generalize the theory to queenside and Black via square mirroring,
   reusing `king_is_castled` and `king_flank_pawn_squares`.
4. P2.4: add the `shape_projection` field to `MoveProbe`, populate it in
   `probe_moves`, and delete the absorbed flank-pawn helpers from `probe.py`.
5. P2.5: splice the projection into `build_root_argument_graph`; add the shape
   `ArgumentEvidence` factory and value mapping in `evidence.py`.
6. P2.6: add tests proving subchunks are individually IN/OUT in the grounded
   extension and that threat versus response resolves at the subchunk.
7. P2.7: run the verification gates, run a Stockfish 2000 match, and mine any
   move-selection changes to confirm they are improvements.

## Old-Path Search Gates

- `rg -n -F "def flank_pawn_weakening_objections" dialectical_chess/` returns no matches.
- `rg -n -F "def advanced_flank_pawn_response_labels" dialectical_chess/` returns no matches.
- `rg -n -F "def advanced_flank_pawn_threats" dialectical_chess/` returns no matches.
- `rg -n -F "FLANK_PAWN_WEAKENING" dialectical_chess/` returns no matches.
- `rg -n -F "ADVANCED_FLANK_PAWN_RESPONSE" dialectical_chess/` returns no matches.

## Verification Gates

- `uv run pyright`
- `uv run pytest --timeout=90 tests/test_shape_arguments.py`
- `uv run pytest --timeout=90 tests/test_dialectical_chess_evidence_ablation.py -k "evidence or objection or flank or shell or shape"`
- `uv run pytest --timeout=90 tests/test_dialectical_chess_engine_api.py -k "clock or lower_depth or depth_zero"`
- After the commit: one 2-game Stockfish 2000 UCI match at `30+0.2`, no
  timeouts, crashes, losses on time, or strength regression.
