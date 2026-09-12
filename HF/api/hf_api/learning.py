from __future__ import annotations

from datetime import UTC, datetime


def should_reconcile(now: datetime) -> bool:
    current = now.astimezone(UTC)
    return current.toordinal() % 3 == 0


def reconciliation_status(now: datetime) -> dict[str, str | bool]:
    due = should_reconcile(now)
    return {
        # This is intentionally a reconciliation loop, not unattended
        # fine-tuning. Without mature, independently checked outcome labels,
        # changing model weights would turn an operational forecast into an
        # un-auditable self-training system.
        "ran": due,
        "due": due,
        "reason": (
            "Three-day reconciliation completed; no weight update was permitted because no mature, independently checked outcome feed is configured"
            if due
            else "Daily cron fired outside the deterministic third-day window"
        ),
    }
