# Design notes

This document explains the arithmetic, the event-time model and the storage
layout. The README is the user-facing summary; this is the reference for why
the implementation looks the way it does.

## Event time is the only clock

Every record has a key `(series, ts, seq)`:

- `ts` is the event timestamp, an integer. The library uses nanoseconds by
  convention but does not interpret the unit.
- `seq` breaks ties between records with the same `ts`. When the caller omits
  it, the store assigns a per-series arrival counter, so two observations with
  the same timestamp remain distinct.
- Records are ordered by `(ts, seq)` and nothing else. Wall-clock time, file
  order and arrival order never affect a value.

A feature value at key `k` is a function of the accepted records with key
`<= k`. That is the point-in-time contract. It holds by construction, because
the state machine only ever sees records in key order and never consults the
future. The tests check it against a batch computation over every prefix.

## Watermarks

Each series remembers `max_ts`, the greatest timestamp accepted so far. The
watermark is

```
watermark = max_ts - allowed_lateness
```

A record is **late** when `ts < watermark`. The comparison is strict: a record
exactly at the watermark is still accepted, so with `allowed_lateness = 0` any
record strictly older than the newest timestamp is late, while a record with
the same timestamp is ordered by `seq`.

`allowed_lateness` is in the same integer units as `ts`. Configuring it is how
the caller states how much disorder the stream is allowed to contain.

## Feature recurrences

All windows are **count windows**: the last `W` observations that carried the
required input. A record whose input column is missing (NaN) does not advance
the window and produces a NaN output row.

The recurrences below are the definition used by the implementation. They are
executed identically by the incremental path and by the batch replay.

### Rolling mean

State: `n`, `mean`, and a ring buffer of the last `W` values. Adding `x`:

```
n'    = n + 1
delta = x - mean
mean' = mean + delta / n'
```

Removing `x` (the eviction step) is the inverse:

```
n'    = n - 1
delta = x - mean
mean' = mean - delta / n'
```

### Rolling standard deviation

State: `n`, `mean`, `M2 = sum((x - mean)^2)` and the ring buffer. This is
Welford's recurrence with the removal step described by Chan, Golub and
LeVeque (1979):

```
add:    n' = n + 1
        delta = x - mean
        mean' = mean + delta / n'
        M2'   = M2 + delta * (x - mean')

remove: n' = n - 1
        delta = x - mean
        mean' = mean - delta / n'
        M2'   = M2 - delta * (x - mean')
```

The value is `sqrt(max(M2 / n, 0))`, the population standard deviation
(`ddof = 0`). Keeping `M2` instead of a sum of squares avoids subtracting two
large, nearly equal numbers, which is what makes the naive formula lose all
significance when the values are constant with a large offset.

### Rolling z-score

`(x - mean) / std` with the moments above. Undefined and returned as NaN when
`n < 2` or `std == 0`.

### Cumulative return

`x / oldest - 1`, where `oldest` is the first value of the window (the oldest
surviving observation, or the first observation if the window is not full).
Undefined and returned as NaN when `n < 2` or `oldest == 0`.

### Rolling Pearson correlation

State for a pair of columns: `n`, `mean_x`, `mean_y`, `M2x`, `M2y` and the
co-moment `C = sum((x - mean_x) * (y - mean_y))`.

```
add:    n' = n + 1
        dx = x - mean_x
        dy = y - mean_y
        mean_x' = mean_x + dx / n'
        mean_y' = mean_y + dy / n'
        M2x' = M2x + dx * (x - mean_x')
        M2y' = M2y + dy * (y - mean_y')
        C'   = C + dx * (y - mean_y')

remove: n' = n - 1
        dx = x - mean_x
        dy = y - mean_y
        mean_x' = mean_x - dx / n'
        mean_y' = mean_y - dy / n'
        M2x' = M2x - dx * (x - mean_x')
        M2y' = M2y - dy * (y - mean_y')
        C'   = C - dx * (y - mean_y')
```

`r = C / sqrt(M2x * M2y)`, clamped to `[-1, 1]`. With two observations the
correlation is exactly ±1 by definition and returned as such. Undefined and
returned as NaN when `n < 2` or either `M2` is non-positive.

### EWMA

`alpha = 1 - 0.5**(1 / halflife)` and

```
s_0 = x_0
s_t = alpha * x_t + (1 - alpha) * s_{t-1}
```

The first observation seeds the state. No warm-up beyond that: an EWMA is
defined from the first value, unlike a window statistic.

## Why the bit-for-bit test is not circular

The batch path builds a fresh state machine per feature and series and replays
the records in `(ts, seq)` order. The incremental path carries the same state
machine across calls and, after a late insertion, rebuilds it by replaying the
same records in the same order. Both paths execute the same Python function
with the same arguments in the same order, so the results are identical down to
the last bit of the float.

The test therefore proves that state management is exact: the right values are
evicted, the replay starts at the right position, and out-of-order arrivals
converge to the sorted result. It does **not** prove that the recurrence equals
the textbook formula in real arithmetic; no floating-point recurrence does.
`tests/test_features.py` checks that separately against NumPy's two-pass
implementations with a relative tolerance of `1e-8`, which is the precision
limit of the sliding removal step.

Two consequences worth knowing:

- A late insertion replays the tail and re-versions every value whose bits
  changed. Values whose window contents did not change can still differ in the
  last bit, because the recurrence sees the same numbers in a different order.
  That is the cost of the bit-for-bit contract.
- The correlation of a two-point window is exact, but in general the sliding
  Welford state accumulates rounding. The drift is slow: after 500 000 window
  updates on a random walk, `std` differed from a NumPy window by 3e-11
  relative and the mean by 7e-15. A periodic refresh from the ring buffer would
  give a hard bound; it is not needed for the tolerances this library promises,
  and the bit-for-bit contract is unaffected either way.

## Late-data policies

`LatePolicy.DROP` (default) discards a late record and writes a row to
`dropped/` with the record key, the watermark and the policy. The feature state
is untouched. This is lossy and deliberate: the caller configured a watermark
and the store respects it.

`LatePolicy.RECOMPUTE` inserts the record at its event-time position and
replays the series from the first record. Every output from the insertion
position onwards is compared bit by bit with the stored value; a changed value
gets a new version with reason `late_recompute`. The final state is identical
to having received the record in order.

The replay is O(N) in the series length. That is the honest cost of keeping the
audit exact without periodic state snapshots. The common path, an in-order
tick, is O(1) and writes one row per feature.

## Point-in-time queries

`point_in_time(feature, series, ts)` reads every stored version for the
partition, keeps the highest version per key, sorts by `(ts, seq)` and
bisects on `ts`. Only keys with `ts <= query` can be returned. The result is
the value of the newest observation at or before the query time, which is the
usual feature-store lookup semantics.

The timeline is cached per `(feature, series)` and invalidated on any write.

## Storage layout

```
root/
  catalog.json
  features/feature=<name>/series=<series>/part-<ns>-<id>.parquet
  raw/series=<series>/part-<ns>-<id>.parquet
  dropped/series=<series>/part-<ns>-<id>.parquet
```

`catalog.json` holds the column names, the late-data configuration and the
feature definitions, so reopening a store restores the definitions without the
caller repeating them.

The feature table is append-only. A row is one version of one feature at one
key:

| column        | type        | meaning                                   |
|---------------|-------------|-------------------------------------------|
| `feature`     | string      | feature name                              |
| `series`      | string      | entity                                    |
| `ts`          | int64       | event time                                |
| `seq`         | int64       | tie-breaker                               |
| `value`       | float64     | feature value                             |
| `version`     | int32       | 1 for the first write, then 2, 3, ...     |
| `reason`      | string      | `append`, `backfill`, `late_recompute`, `out_of_order` |
| `ingested_ns` | int64       | wall-clock time of the write              |

Every write goes to a new file named after the write instant and a random
suffix, written to a temporary name and moved with `os.replace`. A reader
never sees a half-written file, and a crash loses at most the in-memory buffer
(the default flush threshold is 50 000 rows).

`raw/` keeps the accepted input records. On `open`, the store reads them,
sorts by `(ts, seq)` and rebuilds every state machine by replaying the series.
The feature table is never used to reconstruct state, which keeps the raw log
as the single source of truth.

Partition names are percent-encoded, so a series called `BTC / USD` becomes
`series=BTC%20%2F%20USD` and stays a single directory component on Windows and
on POSIX.

## Complexity

| Operation                          | Time                  | Memory            |
|------------------------------------|-----------------------|-------------------|
| In-order tick, F features          | O(F)                  | O(W) per feature  |
| Out-of-order or late recompute     | O(N · F)              | O(N · F)          |
| Point-in-time query (cached)       | O(log N)              | O(N) timeline     |
| Point-in-time query (cold)         | O(N)                  | O(N) timeline     |
| Reopen                             | O(N · F)              | O(N · F)          |

`N` is the number of records in the series and `W` the window size.

## Limitations

- Single writer. There is no locking and no cross-process coordination.
- State lives in memory. Reopening replays the raw log, which is linear in the
  number of records.
- Count windows only. Time-based windows would make eviction depend on `ts`
  deltas rather than on a fixed ring.
- Queries load the whole partition. Predicate pushdown on `ts` is possible but
  not implemented.
- The audit table grows without compaction. Correction rows are rare, but a
  long-lived store needs a retention policy eventually.
- The sliding moments are not exact in real arithmetic. See the discussion
  above.
