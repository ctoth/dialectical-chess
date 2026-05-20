# Phase 1 board.py - independent Analyst review

Review workflow actually used: I read the controlling prompt, then reviewed
`C:\Users\Q\code\dialectical-checkers\dialectical_checkers\board.py`,
`tests\test_board.py`, `notes\checkers-design.md` section 2, and
`notes\checkers-port-plan.md` section 5.1. I ran the existing test suite and
used temporary read-only oracle probes outside the repository under review. I
did not modify any code or test under review.

## Findings

### CRITICAL - king jump loops back to the origin are generated incorrectly

File and line:
`dialectical_checkers/board.py:272-277`

`_expand_jumps()` tests the landing square against `self.cells[land]`, but
`self.cells` is the original board. The moving king's origin therefore remains
occupied for the entire recursive expansion. In a legal circular king jump, the
origin is empty after the first hop and can be a later landing square; the
implementation incorrectly blocks that continuation and emits incomplete jump
sequences.

Concrete oracle check:

```text
FEN: B:W17,18,25,26:BK14
engine: ['14x21x30x23', '14x23x30x21']
pydraughts english oracle: ['14x21x30x23x14', '14x23x30x21x14']
```

Why it matters: WCDF 1.19/1.20 require a started multi-jump to continue until
no further capture remains. The engine returns moves that stop one capture too
early, so the legal move set is wrong and the returned moves are illegal
in positions with circular king captures.

### MAJOR - the captured-piece-at-end tests do not exercise the claimed rule

File and line:
`tests/test_board.py:190-204`, `tests/test_board.py:208-225`

The tests named for captured-piece blocking do not actually construct the
subtle condition they claim to cover.

`test_captured_square_blocks_landing()` uses a Red man and then explains that
the possible continuation is backward, so the position cannot test whether a
captured square remains occupied during a legal continuation.

`test_captured_pieces_block_landing_in_chain()` says it constructs a king with
White men on 18 and 11, but the FEN is only `B:W18:BK15`. There is no ring, no
second enemy piece, and no geometric continuation. The final assertion only
checks that generated moves do not list the same captured square twice.

Why it matters: this is one of the prompt's structurally weak gates. The code
does include `land in captured` handling, but the test suite does not prove the
WCDF 1.19 end-of-sequence removal rule on a real multi-jump continuation.

### MAJOR - deterministic edge cases do not cover king multi-jumps or king backward capture

File and line:
`tests/test_board.py:69-83`, `tests/test_board.py:476-481`,
`tests/test_board.py:484-516`

The curated `EDGE_CASES` include king quiet movement and a one-square king
capture, but no king backward capture and no king multi-jump. My coverage probe
over the 13 edge cases found:

```text
has_king: 3 positions
king_move: 3 positions
king_jump: 1 position
king_backward_jump: 0 positions
king_multijump: 0 positions
```

The seeded random walk did reach some king cases:

```text
positions_checked: 304
has_king: 173 positions
king_move: 106 positions
king_jump: 11 positions
king_backward_jump: 9 positions
king_multijump: 2 positions
```

That means kings are not completely untested. However, the only deterministic
oracle edge cases do not cover the exact king-chain surface where the
implementation is wrong, and the random walk follows the engine's own generated
moves. An omitted legal continuation can therefore stay invisible unless the
walk happens to reach and compare that exact position.

Why it matters: the current suite passes while the legal king-loop position
above disagrees with the oracle. The board substrate needs explicit curated
oracle positions for king backward capture, king multi-jump, and king loop
captures that revisit the origin square.

### MINOR - malformed PDN-FEN with empty square tokens is silently accepted

File and line:
`dialectical_checkers/board.py:447-450`

`from_fen()` strips each comma-separated token and silently continues on empty
tokens. For example, `B:W1,,2:B` parses as if it were `B:W1,2:B`.

Why it matters: the prompt asks for malformed and edge-case PDN-FEN review.
The parser rejects many malformed forms, including bad field counts, bad turn
tokens, wrong tags, out-of-range squares, and duplicate squares, but it accepts
empty list elements created by doubled or trailing commas. There is also no
invalid-FEN test coverage in `tests/test_board.py`; the FEN tests at
`tests/test_board.py:349-370` cover valid round-trips only.

### MINOR - threefold repetition uses Python's randomized hash as position identity

File and line:
`dialectical_checkers/board.py:183-185`, `dialectical_checkers/board.py:387-390`

The semantic choice `(cells, turn)` is correct for repetition identity: it
includes the side to move and excludes counters. The implementation then stores
`hash((self.cells, self.turn))`, which is process-randomized for strings and
not collision-free.

Why it matters: this is not causing the current test failure, and in-process
repetition usually works. But the under-specified "position hash" resolution is
weaker than using the tuple identity or a stable serialization directly. A hash
collision could produce a false draw, and persisted/debugged histories are not
stable across runs.

## Verified Sound Surfaces

- `uv run pytest -q` currently passes: `41 passed in 1.88s`.
- The start-position perft claim is covered by `tests/test_board.py:467-473`
  and passed with `[7, 49, 302, 1469, 7361, 36768]`.
- The differential side mapping is not vacuous in the current tests:
  `tests/test_board.py:48-63` compares actual pydraughts English PDN move
  strings, and `CheckersBoard.initial().to_fen()` serializes Red as PDN-FEN
  `B` at `board.py:395-416`. The start edge case and asymmetric curated
  positions would fail if pydraughts `BLACK` were not the engine's Red.
- Crowning-ends-turn is implemented at `board.py:281-289`, unit-tested at
  `tests/test_board.py:228-244`, included in the differential edge cases at
  `tests/test_board.py:80` and `tests/test_board.py:476-481`, and matched the
  oracle in an additional probe: `B:W25,26:B21` produced `['21x30']` for both.
- Terminal no-move-is-loss is implemented at `board.py:355-376` and tested at
  `tests/test_board.py:255-263`; there is no stalemate-draw branch.
- The 80-ply no-progress draw threshold is implemented at `board.py:385-386`
  and tested at `tests/test_board.py:308-325`.
- Threefold repetition is exercised by a back-and-forth king sequence at
  `tests/test_board.py:328-346`.
- Empty-side FEN fields are supported by `board.py:445-446`; an oracle probe
  for `B:W:B` produced no legal moves in both the engine and pydraughts.
- Perft and the seeded random walk are deterministic in the current code path:
  `legal_moves()` returns sorted tuples at `board.py:196-199`, and the random
  walk uses `random.Random(20260520)` at `tests/test_board.py:492`.
