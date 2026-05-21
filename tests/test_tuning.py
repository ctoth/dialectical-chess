from __future__ import annotations

from dialectical_chess import tuning


def test_time_tuning_constants_are_in_documented_ranges() -> None:
    assert 0 <= tuning.TIME_OVERHEAD_MS <= 250
    assert 1 <= tuning.TIME_DEFAULT_MOVES_TO_GO <= 60
    assert 0.0 <= tuning.TIME_INCREMENT_FRACTION <= 1.0
    assert 0.0 < tuning.TIME_RESERVE_FRACTION <= 0.10
    assert 0.0 < tuning.TIME_MAX_MOVE_FRACTION <= 0.50
    assert 1 <= tuning.TIME_MIN_BUDGET_MS <= tuning.CRITICAL_BUDGET_MS
    assert 0 < tuning.TIME_MIN_RESERVE_MS <= tuning.TIME_MAX_RESERVE_MS
    assert tuning.CRITICAL_BUDGET_MS <= tuning.LOW_BUDGET_MS
    assert tuning.LOW_BUDGET_MS <= tuning.REPLY_MATE_SCAN_MIN_BUDGET_MS
