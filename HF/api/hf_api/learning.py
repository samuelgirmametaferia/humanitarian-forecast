from __future__ import annotations

from datetime import UTC, datetime


def should_reconcile(now: datetime) -> bool:
    current = now.astimezone(UTC)
    return current.toordinal() % 3 == 0


def reconciliation_status(now: datetime) -> dict[str, str | bool]:
    due = should_reconcile(now)
    return {
        # Reconciliation labels matured forecasts with realized UCDP
        # outcomes; the weekly retrain job then fine-tunes from the promoted
        # model and publishes a new registry version automatically. Weight
        # changes are therefore continuous but always grounded in verified
        # event data — public preview signals are features, never labels.
        "ran": due,
        "due": due,
        "reason": (
            "Three-day reconciliation window: matured forecasts are being labeled "
            "with realized UCDP outcomes for the weekly retrain"
            if due
            else "Daily cron fired outside the deterministic third-day window"
        ),
    }
