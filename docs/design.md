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

Windows are either **count windows** (the last `W` observations) or **time
windows** (every observation with `ts` in `[t - D, t]`). A record whose input
column is missing (NaN) does not advance either window and produces a NaN
output row.

Count windows keep a ring buffer of values. Time windows keep a deque of
`(ts, value)` pairs and, after appending the new record, pop from the left while
`oldest.ts < t - D`. The comparison is strict, so an observation exactly `D`
time units old is still inside the window. Because records are processed in
non-decreasing `ts`, evictions are monotone.

The recurrences below are the definition used by the implementation. They are
executed identically by the incremental path and by the batch replay, which is
why time windows preserve the bit-for-bit guarantee without special cases.

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
  state/LATEST
  state/checkpoint-<id>/manifest.json
  state/checkpoint-<id>/state.parquet
  audit/pruned-<ns>.json
```

`catalog.json` holds the column names, the late-data configuration, the
checkpoint interval, the audit policy and the feature definitions, so reopening
a store restores the definitions without the caller repeating them. Catalogues
written by 0.1.0 without the new keys are read with their defaults; the schema
version stays 1 because every added field is optional.

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

`raw/` keeps the accepted input records. Without a checkpoint, `open` replays
them; with one, it restores the state and reads only the files that are not in
the checkpoint manifest. The feature table is never used to reconstruct state,
which keeps the raw log as the single source of truth.

Partition names are percent-encoded, so a series called `BTC / USD` becomes
`series=BTC%20%2F%20USD` and stays a single directory component on Windows and
on POSIX.

## Checkpoints

A checkpoint is one immutable directory plus a pointer:

- `state.parquet` has one row per `(series, feature)` with the exact fields of
  the state machine: `n`, `mean`, `M2`, the EWMA value, the correlation
  co-moment, and the window contents as `times`, `xs` and `ys` lists. Float64
  values are stored as float64, so no precision is lost.
- `manifest.json` stores the catalogue hash, the last key and record count of
  every series, and the list of raw files already reflected in the state.
- `state/LATEST` names the newest checkpoint. It is replaced atomically after
  the new directory is complete, so a crash leaves either the old checkpoint or
  the new one, never a partial mix.

`checkpoint()` flushes the storage, writes the state of every series, prunes
older checkpoint directories (`checkpoint_keep`, default 2) and, when
`retain_records=False`, drops the in-memory history. `close()` writes a
checkpoint when the interval is enabled.

On open, the manifest must match the current columns and feature definitions
(sha256 of the canonical catalogue). If it does, the state machines are
restored and only the raw files absent from the manifest are replayed. If it
does not, the store falls back to a full replay, which is also the behaviour
of a store with no checkpoint.

A restored engine keeps `processed` (how many records it has seen) and
`last_key`, so ordering, watermarks and auto-`seq` continue exactly where the
checkpoint stopped. The test suite feeds the same 80 further ticks to a
restored store and to one that never closed and requires bit-identical values.

## Streaming replay

`FeatureStore.open` with `retain_records=False` and no checkpoint reads the raw
log file by file and feeds the state machines in order, so memory stays O(1) in
`N` instead of O(N · F). The raw log is written in arrival order; if a batch is
not sorted by `(ts, seq)` (because a late record was accepted), the store
discards the partial state and rebuilds from a fully sorted read. That is the
documented O(N) memory fallback, and it only triggers for logs with late
corrections.

## Pruning

`prune(keep_versions=N)` groups each partition by `(ts, seq)`, sorts by
`version` and keeps the last N rows of each group. It rewrites the partition
with the surviving rows, deletes the old immutable files and writes
`audit/pruned-<ns>.json` with the policy and per-partition counts. The latest
version of a key is always kept, so every query returns the same value before
and after pruning.

The Parquet schema is unchanged, which is why no migration is needed. The
operation is not a transaction across partitions: a crash between the compacted
write and the deletion of the old files can leave two identical rows with the
same version. `history` would show them twice, but `get` and `point_in_time`
keep the highest version per key, so results are unaffected.

## Complexity

| Operation                          | Time                  | Memory            |
|------------------------------------|-----------------------|-------------------|
| In-order tick, F features          | O(F)                  | O(W) per feature  |
| Out-of-order or late recompute     | O(N · F)              | O(N · F)          |
| Point-in-time query (cached)       | O(log N)              | O(N) timeline     |
| Point-in-time query (cold)         | O(N)                  | O(N) timeline     |
| Reopen, no checkpoint              | O(N · F)              | O(N · F) retained, O(F · W) streaming |
| Reopen, checkpoint                 | O(tail · F)           | O(F · W) + tail   |
| Checkpoint write                   | O(F · W)              | O(F · W)          |
| Prune, keep N versions             | O(rows)               | O(rows of partition) |

`N` is the number of records in the series, `W` the window size (in records for
count windows, in records inside the time span for time windows) and `tail` the
records written after the checkpoint.

## Limitations

- Single writer. There is no locking and no cross-process coordination.
- A late recompute older than the retained history rebuilds from the raw log,
  which is O(N) in time and memory. Checkpoints make reopening cheap, not
  arbitrarily old corrections.
- Queries load the whole partition. Predicate pushdown on `ts` is possible but
  not implemented.
- Pruning rewrites a partition in place and is not a cross-partition
  transaction; see the duplicate-row note above.
- Time windows store one entry per record inside the span, so a wide window
  over a dense stream costs memory proportional to the number of records in it.
- The sliding moments have bounded floating-point drift; they are not exact in
  real arithmetic. See the discussion above.
