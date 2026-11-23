# feature-store

[![CI](https://github.com/IkerAbad/feature-store/actions/workflows/ci.yml/badge.svg)](https://github.com/IkerAbad/feature-store/actions/workflows/ci.yml)

Incremental financial features with point-in-time correctness, count and time
windows, watermarks for late data, restart checkpoints and a versioned Parquet
backend.

The PyPI distribution is `incremental-feature-store`; the import package is
`feature_store`.

Author: Iker Abad (GitHub: [IkerAbad](https://github.com/IkerAbad)).
License: MIT, Copyright (c) 2026 Iker Abad.

## What it does

A feature definition is declarative. Windows can be counted in observations or
measured in event time:

```python
from feature_store import FeatureSpec

FeatureSpec.mean("sma_20", window=20)  # last 20 ticks
FeatureSpec.mean("sma_1s", time_window=1_000_000_000)  # last second
FeatureSpec.std("std_20", window=20)
FeatureSpec.zscore("z_20", window=20)
FeatureSpec.cum_return("ret_20", window=20)
FeatureSpec.corr("corr_20", window=20, x="x", y="y")
FeatureSpec.ewma("ewma_5", halflife=5.0)
```

A tick updates only the state it touches:

```python
from feature_store import FeatureStore

store = FeatureStore.open("data", columns=("x", "y"), late_policy="recompute")
store.register(FeatureSpec.mean("sma_20", window=20))
store.push("BTC", ts=1_700_000_000_000_000_000, value=101.5)

store.get("sma_20", "BTC")  # latest event-time value
store.point_in_time("sma_20", "BTC", ts)  # value using records with ts <= t
store.history("sma_20", "BTC")  # every retained version
store.checkpoint()  # make the next open O(tail)
store.prune(keep_versions=2)  # compact the audit table
```

The four properties the library exists for:

- **Incremental**: an in-order tick costs O(features) and O(window) memory. The
  full history is never recomputed on the normal path.
- **Point-in-time**: a value at `t` only uses records with timestamp at or
  before `t`. The event timestamp is the only clock; arrival order never leaks
  into a feature.
- **Restartable**: periodic checkpoints store the exact state machines, so
  reopening restores them and replays only the records written after the
  checkpoint. The restored state is verified bit for bit against a full replay.
- **Auditable**: every correction is a new row with a higher `version`. The old
  value is never overwritten, and pruning keeps the last N versions
  configurable per store.

## Guarantees

1. **No future leakage.** Features are functions of the accepted records sorted
   by `(ts, seq)` up to the query key. `point_in_time` can only return rows with
   `ts <= t`. Tested against a fresh batch computation for every prefix of a
   series.
2. **Bit-for-bit incremental/batch equivalence.** The batch path builds fresh
   state machines and replays the sorted history; the incremental path carries
   the same machines. Both execute the same arithmetic in the same order, so the
   float bits are identical. The test suite compares 20 synthetic series for all
   six features and requires exact equality, not a tolerance
   (`tests/test_equivalence.py`).
3. **Late data is explicit.** A watermark (`max_ts - allowed_lateness`) decides
   what is late. The configured policy either drops the record with an audit
   entry or inserts it in event-time order and recomputes the affected tail.
4. **Versions are append-only.** A corrected value keeps the previous versions.
   `history` returns them in `(ts, seq, version)` order until the audit policy
   prunes the oldest ones.
5. **Checkpoints are exact.** `checkpoint()` serialises every state machine,
   including the window contents. Reopening restores it and replays only the raw
   files written afterwards. A test continues both the reopened store and a
   never-closed one for 80 ticks and requires bit-identical values.
6. **Pruning cannot change a query.** `prune(keep_versions=N)` rewrites each
   partition without the older versions. The latest version of every key is
   always kept, the Parquet schema is unchanged, and the operation writes a
   manifest under `audit/`.

The exactness in guarantee 2 is exactness with respect to the recurrence that
defines each feature, not to real-number arithmetic. `tests/test_features.py`
also compares every feature against an independent NumPy implementation with a
relative tolerance of `1e-8`; the difference is the rounding of the sliding
Welford update. See [docs/design.md](docs/design.md) for the argument in full.

## Features

A window is either a count of observations (`window=20`) or a span of event
time (`time_window=60_000_000_000`). A record with a missing input (NaN) does
not advance either kind of window and produces a NaN output.

| Kind         | Value                                                        | Warm-up |
|--------------|--------------------------------------------------------------|---------|
| `mean`       | arithmetic mean of the window                                 | first observation |
| `std`        | population standard deviation (`ddof=0`)                      | first observation |
| `zscore`     | `(x - mean) / std` of the incoming value                      | 2 observations, `std > 0` |
| `cum_return` | `x / oldest - 1` over the window                              | 2 observations, `oldest != 0` |
| `corr`       | Pearson correlation of two columns over the window            | 2 observations, both variances `> 0` |
| `ewma`       | `alpha = 1 - 0.5**(1/halflife)`, `s = alpha*x + (1-alpha)*s`   | first observation |

Time windows are inclusive: the value at `t` uses every observation with
`ts` in `[t - time_window, t]`. Count windows keep a ring buffer; time windows
keep a deque of `(ts, value)` pairs. Both are O(1) amortised per record.

Undefined values are `NaN`, not zero. `std` and `corr` use Welford's recurrence
with an exact removal step, which avoids subtracting two large sums of squares.

## Late data

```python
store = FeatureStore.open(
    "data",
    columns=("x", "y"),
    late_policy="drop",  # or "recompute"
    allowed_lateness=60_000_000_000,  # 60 s in nanoseconds
)
```

The watermark is `max_ts - allowed_lateness` **per series**. A record is late
when `ts < watermark`; a record exactly at the watermark is still accepted.

- `drop`: the record is discarded and written to `dropped/` with the watermark
  and the policy. Nothing else changes. This is lossy by construction.
- `recompute`: the record is inserted at its event-time position and the series
  is replayed. Every output whose bits changed from the insertion point onwards
  gets a new version with reason `late_recompute`. The final state is identical
  to receiving the record in order.

Out-of-order records that are **not** late are always accepted, even under
`drop`: the watermark says what is too old, not what is out of order. The replay
is O(N) in the series length when the engine no longer holds the history, and
O(tail) when it does; the in-order path is O(1).

## Storage

```
data/
  catalog.json
  features/feature=sma_20/series=BTC/part-<ns>-<id>.parquet
  raw/series=BTC/part-<ns>-<id>.parquet
  dropped/series=BTC/part-<ns>-<id>.parquet
  state/LATEST
  state/checkpoint-<id>/{manifest.json,state.parquet}
  audit/pruned-<ns>.json
```

- Partitioned by feature and series, percent-encoded so any name stays a single
  path component.
- Append-only feature files, written to a temporary name and moved atomically.
- One row per version of each `(feature, series, ts, seq)` key, with `version`
  and `reason` (`append`, `backfill`, `late_recompute`, `out_of_order`).
- The raw log is the source of truth. Without a checkpoint, `FeatureStore.open`
  replays it; with one, it restores the exact state and reads only the raw files
  written after the checkpoint.
- `catalog.json` stores the columns, the feature definitions, the late-data
  policy, the checkpoint interval and the audit policy.
- `flush()` and `close()` persist the pending buffer; the default threshold is
  50 000 rows. `close()` also writes a checkpoint when checkpoints are enabled,
  so the next open is fast.

Checkpoints are enabled by default (`checkpoint_every=1_000_000` records per
series). The state file is Parquet with one row per `(series, feature)` and
list columns for the window contents; float64 values keep their bits. The
checkpoint format is versioned separately (`format: 1`) and does not touch the
feature-table schema.

Pruning rewrites one partition at a time with the surviving rows and deletes
the old files. The schema is unchanged, so no migration is needed. The manifest
under `audit/` records the policy and the rows removed.

Queries read the partition, keep the highest version per key and bisect on
`ts`. The resulting timeline is cached until the next write.

## Benchmark

Machine: AMD Ryzen 9 5900X, 32 GB DDR4-3600, Windows 11 Pro, Python 3.12.10,
NumPy 2.5.5, single-threaded. Measured on 2026-11-23. Each number is one
measured run, not a best-of.

Command:

```text
python benchmarks/bench.py --ticks 10000000 --recompute-ticks 10000000 \
    --storage-ticks 200000 --scaling-ticks 30000
```

### Ten million ticks

| Path | Features | Time (s) | ns/tick | Ticks/s |
|---|---:|---:|---:|---:|
| `incremental` (streaming) | 5 | 46.517 | 4,652 | 214,976 |
| `recompute_window` | 5 | 76.341 | 7,634 | 130,990 |
| `incremental_retained` | 5 | 89.765 | 8,977 | 111,402 |
| `incremental + EWMA` (streaming) | 6 | 49.581 | 4,958 | 201,691 |
| `batch_once` | 6 | 3.757 | 376 | 2,661,719 |

Streaming incremental vs per-tick window recompute: **1.64x faster** for the
same five features.

Window scaling, 30,000 ticks per row, same seed, same feature set, streaming
mode:

| Window | Incremental (ns/tick) | Recompute (ns/tick) | Speedup |
|---:|---:|---:|---:|
| 20 | 4,725 | 7,592 | 1.61x |
| 100 | 4,711 | 30,759 | 6.53x |
| 500 | 4,872 | 144,906 | 29.74x |

Parquet backend, 200 000 ticks and five features (1 000 000 feature rows plus
200 000 raw rows):

- End-to-end ingestion, engine included: 26,885 ticks/s (writes are amortised
  every 50 000 rows).
- Reopen: 0.01 s, because `close()` wrote a checkpoint and the state is restored
  instead of replaying the raw log.
- 100 point-in-time queries: 0.72 s (the first builds the timeline, the rest
  hit the cache).

### What is measured

- `incremental`: 10 million in-order ticks through `FeatureStore.null` in
  streaming mode (`retain_records=False`), with the five windowed features
  (`mean`, `std`, `zscore`, `cum_return`, `corr`). No rows are materialised, no
  disk is touched and the engine does not keep the history, so memory is
  O(window). Converting each NumPy chunk to Python floats is part of the feed
  loop and is included.
- `recompute_window`: the non-incremental baseline. On every tick it rescans the
  trailing window with plain Python and computes the same five statistics. No
  library shortcuts, no NumPy.
- `incremental_retained`: the same stream with the default retained history,
  which is what feeds in-memory corrections. It keeps every record and output
  and is the mode the earlier release always used.
- `incremental + EWMA`: streaming mode with a sixth feature, which is O(1) and
  has no window.
- `batch_once`: one vectorized NumPy pass over the whole history, the offline
  equivalent. It is fast, but it materialises every array, cannot answer a
  query until the pass ends and has no concept of late data or versions.
- `storage`: 200 000 ticks through the Parquet backend, then close (which
  flushes and writes a checkpoint), reopen (which restores it) and 100
  point-in-time queries. Reported separately because it measures disk I/O, not
  feature arithmetic.

### Reopening: 1, 10 and 100 million ticks

The second benchmark isolates `FeatureStore.open` and uses a raw log generated
directly with pyarrow, because the restart path does not read the feature
table. Every size is measured twice on the same store: first with no
checkpoint (the state machines consume every raw record) and then after a
checkpoint plus 100 000 further raw records (only the tail is replayed). The
full replay uses `retain_records=False`, which keeps memory bounded and makes
the 100-million case possible at all.

Command:

```text
python benchmarks/reopen.py --ticks 1000000 10000000 100000000
```

Measured on 2026-11-23 with Python 3.12.10, NumPy 2.5.5.

| Ticks | Full replay (s) | Retained replay (s) | Restore + 100k tail (s) | Speedup |
|---:|---:|---:|---:|---:|
| 1,000,000 | 9.81 | 11.80 | 0.94 | 10.5x |
| 10,000,000 | 95.34 | 89.29 | 0.97 | 98.3x |
| 100,000,000 | 753.56 | not run | 0.72 | 1043.9x |

The checkpoint write takes 13-21 ms. The state file depends on the number of features and the window, not on the number of ticks: it stores the running statistics and the live window, not the history. The checkpoint directory also stores the list of consumed raw files, so it grows slowly with the number of files (7 KB at 1M ticks, 11 KB at 100M in this configuration). The 100 000-record tail accounts for roughly 0.75-0.98 s at the measured replay rate, so the checkpoint restore itself is the remainder of the restore column.

The checkpoint state is independent of the number of ticks: it stores the
feature states, not the history. The 1M and 10M runs also measured the retained
replay that the previous release performed (every record and every output kept
in memory); the 100M case cannot be measured that way on a 32 GB machine, which
is the limitation checkpoints remove.

### Honest reading

At window 20 the streaming path is 1.64x faster: both sides are Python loops,
but the incremental path touches only the new observation and the one being
evicted, while the baseline rescans twenty values. The margin is not larger
because per-tick interpreter overhead dominates.

Retaining the full history roughly doubles the cost (89.8 s against 46.5 s for
5 features) and uses O(N) memory, because every record and every output is
appended to ten-million-element lists. That mode exists for in-memory
corrections on small and medium series; for long streams the streaming mode
plus checkpoints is the intended configuration.

The algorithmic difference is the scaling: the incremental cost does not grow
with the window, and at window 500 the same machine measures 29.7 times fewer
nanoseconds per tick. A vectorized batch pass is still faster per tick than any
per-tick Python loop. That is expected and not the comparison this library is
for: the streaming path wins on update latency, bounded memory, late-data
corrections, auditable versions and restart cost, not on bulk offline
throughput.

The restart measurement is the other half of the story: reopening 100 million
ticks replays in 753 s without a checkpoint and in 0.72 s with one, a 1,044x
difference, because only the 100 000-record tail is replayed.

## Install

From PyPI:

```bash
python -m pip install incremental-feature-store
```

From source:

```bash
python -m pip install -e ".[dev]"
```

Python 3.12+, NumPy, pyarrow. No other runtime dependencies.

## Tests

```bash
python -m pytest -q --cov=feature_store
```

| Acceptance criterion | Test |
|---|---|
| Incremental equals batch bit for bit on 20 synthetic series, count and time windows | `test_equivalence.py::test_twenty_series_incremental_equals_batch`, `test_time_windows_incremental_equals_batch` |
| Out-of-order data follows the configured policy | `test_late_data.py`, `test_properties.py::test_drop_policy_equals_batch_without_the_dropped_records` |
| Point-in-time has no future leakage | `test_point_in_time.py`, `test_properties.py::test_point_in_time_matches_the_prefix` |
| Checkpoints restore the exact state and replay only the tail | `test_checkpoints.py` |
| Pruning keeps the last N versions and never changes a query | `test_pruning.py` |
| Parquet partitions and row versions | `test_storage.py` |
| Register, query, audit | `test_api.py`, `test_storage.py` |
| Features match the textbook definition | `test_features.py` (NumPy reference, `rtol=1e-8`) |

The suite also checks that any arrival order of the same records ends in the
same bit-exact state (`test_properties.py`), which is the strongest form of the
point-in-time property.

CI runs lint (`ruff check`, `ruff format --check`), types (`mypy`), the whole
test suite with a 90% coverage floor and scaled-down benchmarks on Linux and
Windows.

## Project layout

```text
feature-store/
├── benchmarks/
│   ├── bench.py       # 10M ticks: incremental vs recompute
│   ├── reopen.py      # 1M/10M/100M ticks: replay vs checkpoint restore
│   └── results/
├── docs/
│   └── design.md
├── src/feature_store/
│   ├── batch.py       # from-scratch batch oracle
│   ├── checkpoint.py  # exact state snapshots
│   ├── engine.py      # state machines and per-series engine
│   ├── errors.py
│   ├── spec.py        # FeatureSpec, FeatureKind, LatePolicy
│   ├── storage.py     # Parquet, memory and null backends
│   └── store.py       # FeatureStore public API
├── tests/
├── pyproject.toml
└── README.md
```

## Limitations

- Single writer, single process. No locking, no distributed coordination.
- A late recompute is O(N) when the engine no longer holds the history. With
  retained history it is O(tail); with checkpoints it rebuilds from the raw
  log, which is the documented cost of an arbitrarily old correction.
- Queries load the entire partition. Predicate pushdown on `ts` is future work.
- Pruning rewrites a partition in place. A crash between the compacted write
  and the deletion of the old files can leave identical duplicate rows; queries
  keep working because the highest version wins.
- Time windows keep one deque entry per record inside the window, so a very
  wide window over a dense stream uses memory proportional to that span.
- The sliding moments have bounded floating-point drift; they are not exact in
  real arithmetic. See [docs/design.md](docs/design.md).

## License

MIT. See [LICENSE](LICENSE).
