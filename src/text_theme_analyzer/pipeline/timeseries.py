"""Weekly (or monthly) bucketing of cluster activity + spike + stale detection."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta

from text_theme_analyzer.pipeline.model import Spike, StaleIdea, ThemeTimeseries


def _bucket_start(d: date, bucket: str) -> date:
    if bucket == "month":
        return d.replace(day=1)
    if bucket == "week":
        # ISO week: Monday is day 1.
        return d - timedelta(days=d.weekday())
    raise ValueError(f"Unknown bucket: {bucket}")


def build_timeseries(
    note_to_cluster: dict[str, int],
    note_dates: dict[str, date | None],
    *,
    bucket: str = "week",
    spike_window: int = 8,
    stale_window: int = 8,
) -> ThemeTimeseries:
    """Build per-cluster time series from note→cluster and note→date maps.

    Notes with `date is None` are excluded.
    """
    # Group (cluster_id, bucket_date) counts.
    counts: dict[int, Counter[date]] = {}
    for note_id, cid in note_to_cluster.items():
        d = note_dates.get(note_id)
        if d is None:
            continue
        b = _bucket_start(d, bucket)
        counts.setdefault(cid, Counter())[b] += 1

    series: dict[int, dict[date, int]] = {
        cid: dict(c) for cid, c in counts.items()
    }

    spikes: list[Spike] = []
    stale: list[StaleIdea] = []

    if not series:
        return ThemeTimeseries(bucket=bucket, series=series, spikes=spikes, stale=stale)

    # Determine the global date range.
    all_buckets = sorted({b for c in counts.values() for b in c})
    if not all_buckets:
        return ThemeTimeseries(bucket=bucket, series=series, spikes=spikes, stale=stale)
    first, last = all_buckets[0], all_buckets[-1]

    # Build the full bucket axis (so rolling windows have enough history).
    full_axis: list[date] = []
    cursor = first
    step = timedelta(days=7) if bucket == "week" else timedelta(days=31)
    while cursor <= last:
        full_axis.append(cursor)
        cursor = _step_bucket(cursor, bucket)

    # For each cluster, scan the full axis and emit spikes / stale.
    for cid, c_counts in counts.items():
        per_bucket = [c_counts.get(b, 0) for b in full_axis]
        # Spike detection: count > rolling mean + 2*std (window = spike_window).
        for i, b in enumerate(full_axis):
            if per_bucket[i] == 0:
                continue
            start = max(0, i - spike_window)
            prior = per_bucket[start:i]
            if len(prior) < 2:
                continue
            mean = sum(prior) / len(prior)
            std = (sum((x - mean) ** 2 for x in prior) / len(prior)) ** 0.5
            threshold = mean + 2 * std
            if per_bucket[i] > threshold and per_bucket[i] > mean + 1:
                spikes.append(Spike(
                    cluster_id=cid,
                    bucket=b,
                    count=per_bucket[i],
                    rolling_mean=mean,
                    delta=per_bucket[i] - mean,
                ))

        # Stale detection with a 3-rung severity ladder.
        # - weak:   ≥2 total notes, silent for ≥2×stale_window buckets.
        # - medium: ≥3 total notes, silent for the standard stale_window.
        # - strong: ≥5 total notes, last activity in the first half of
        #           the observed range, silent for the standard window.
        #           This is the "I used to write about this a LOT, then
        #           suddenly stopped" pattern — most actionable.
        total = sum(per_bucket)
        if total < 2:
            continue
        # Compute the trailing quiet streak (consecutive zeros at the tail).
        quiet_streak = 0
        for c in reversed(per_bucket):
            if c == 0:
                quiet_streak += 1
            else:
                break
        seen_dates = [b for b, c in zip(full_axis, per_bucket) if c > 0]
        if not seen_dates:
            continue
        last_seen = seen_dates[-1]
        last_seen_idx = full_axis.index(last_seen)
        # Is "last seen" in the first half of the data range? Used for strong.
        first_half_threshold = full_axis[0] + (full_axis[-1] - full_axis[0]) / 2
        in_first_half = last_seen <= first_half_threshold
        # Standard stale: standard window of silence at the end.
        recent = per_bucket[-stale_window:] if len(per_bucket) >= stale_window else per_bucket
        if sum(recent) != 0:
            continue
        severity = "medium"
        if total >= 3 and quiet_streak >= stale_window:
            severity = "medium"
        if total >= 2 and quiet_streak >= 2 * stale_window:
            # weak ladder: take the *most* severe of medium/weak the data
            # supports. 3 notes + long silence is still medium; 2 notes +
            # long silence is weak.
            if total < 3:
                severity = "weak"
        if (
            total >= 5
            and quiet_streak >= stale_window
            and in_first_half
        ):
            severity = "strong"
        stale.append(StaleIdea(
            cluster_id=cid,
            first_seen=seen_dates[0],
            last_seen=last_seen,
            frequency=total,
            severity=severity,
            quiet_streak_buckets=quiet_streak,
        ))

    return ThemeTimeseries(bucket=bucket, series=series, spikes=spikes, stale=stale)


def _step_bucket(b: date, bucket: str) -> date:
    if bucket == "week":
        return b + timedelta(days=7)
    if bucket == "month":
        # Advance to next month, day=1.
        if b.month == 12:
            return b.replace(year=b.year + 1, month=1)
        return b.replace(month=b.month + 1)
    raise ValueError(bucket)
