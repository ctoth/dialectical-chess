"""Named tuning constants with provenance for engine time management."""

from __future__ import annotations

# Phase 2 opinion-valued argumentation constants from
# reports/argdriven-phase2-design-v2.md section 1f.
OPINION_EVIDENCE_UNITS_PER_STRENGTH = 2.0
OPINION_LEAF_BASE_RATE = 0.5

# P1.3 adopts the Codex review time-budget formula from
# reviews/PLAN-argumentation-driven.md. These constants are the named terms in
# that formula.
TIME_OVERHEAD_MS = 50
TIME_DEFAULT_MOVES_TO_GO = 30
TIME_INCREMENT_FRACTION = 0.75
TIME_RESERVE_FRACTION = 0.03
TIME_MAX_MOVE_FRACTION = 0.20
TIME_MIN_BUDGET_MS = 10
TIME_MIN_RESERVE_MS = 100
TIME_MAX_RESERVE_MS = 1000

# Feature profile thresholds keyed off the computed per-move budget.
CRITICAL_BUDGET_MS = 100
LOW_BUDGET_MS = 300
REPLY_MATE_SCAN_MIN_BUDGET_MS = 2_000

# Shared chess material table. P3.3 collapses duplicate search/SMT piece-value
# dictionaries to this single source.
PIECE_VALUE = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900, "k": 0}

# Probe/evidence scoring constants migrated from the Phase-2 probe heuristics.
COMPENSATING_TACTICAL_THREAT_THRESHOLD = 700
LARGE_SEARCH_REFUTATION_THRESHOLD = -500
CHECKMATE_SCORE = 1_000_000
CHECK_SCORE = 1_000
POSITIONAL_REASON_SCORE = 25
REPLY_MATE_REFUTATION_SCORE = -100_000
SEARCH_REPLY_MATE_TRIGGER_SCORE = -700
UNSUPPORTED_MAJOR_DRIFT_PENALTY = -300
KING_ESCAPE_SCORE = 300
MOVED_PIECE_DEFENDED_SCORE = 15
QUEEN_BLUNDER_EXCHANGE_THRESHOLD = -300
MINOR_PIECE_VALUE = 300
MAJOR_PIECE_VALUE = 500
QUEEN_VALUE = 900
