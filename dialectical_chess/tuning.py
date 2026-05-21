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
