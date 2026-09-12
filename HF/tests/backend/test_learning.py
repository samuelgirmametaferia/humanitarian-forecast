from __future__ import annotations

from datetime import UTC, datetime

from api.hf_api.learning import reconciliation_status, should_reconcile


def test_reconcile_is_deterministic_third_day_window() -> None:
    expected = [
        (datetime(2026, 9, day, tzinfo=UTC).toordinal() % 3 == 0)
        for day in (11, 12, 13)
    ]
    actual = [should_reconcile(datetime(2026, 9, day, tzinfo=UTC)) for day in (11, 12, 13)]
    assert actual == expected
    assert actual.count(True) == 1


def test_reconciliation_runs_on_the_due_window_and_reports_the_live_loop() -> None:
    status = reconciliation_status(datetime(2026, 9, 13, tzinfo=UTC))

    assert status["due"] is True
    assert status["ran"] is True
    assert "labeled" in str(status["reason"])
