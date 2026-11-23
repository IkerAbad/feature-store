# feature-store

[![CI](https://github.com/IkerAbad/feature-store/actions/workflows/ci.yml/badge.svg)](https://github.com/IkerAbad/feature-store/actions/workflows/ci.yml)

Incremental financial features with point-in-time correctness, watermarks for
late data and a versioned Parquet backend.

Author: Iker Abad (GitHub: [IkerAbad](https://github.com/IkerAbad)).
License: MIT, Copyright (c) 2026 Iker Abad.

## What it does

A feature definition is declarative:

```python
from feature_store import FeatureSpec

FeatureSpec.mean("sma_20", window=20)
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
store.history("sma_20", "BTC")  # every version ever written
```

The three properties the library exists for:

- **Incremental**: an in-order tick costs O(features) and O(window) memory. The
  full history is never recomputed on the normal path.
- **Point-in-time**: a value at `t` only uses records with timestamp at or
  before `t`. The event timestamp is the only clock; arrival order never leaks
  into a feature.
- **Auditable**: every correction is a new row with a higher `version`. The old
  value is never overwritten, so the history explains itself.

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
   `history` returns all of them in `(ts, seq, version)` order.

The exactness in guarantee 2 is exactness with respect to the recurrence that
defines each feature, not to real-number arithmetic. `tests/test_features.py`
also compares every feature against an independent NumPy implementation with a
relative tolerance of `1e-8`; the difference is the rounding of the sliding
Welford update. See [docs/design.md](docs/design.md) for the argument in full.

## Features

All windows are count windows: the last `W` observations that carried the
required input. A record with a missing input (NaN) does not advance the window
and produces a NaN output.

| Kind         | Value                                                        | Warm-up |
|--------------|--------------------------------------------------------------|---------|
| `mean`       | arithmetic mean of the window                                 | first observation |
| `std`        | population standard deviation (`ddof=0`)                      | first observation |
| `zscore`     | `(x - mean) / std` of the incoming value                      | 2 observations, `std > 0` |
| `cum_return` | `x / oldest - 1` over the window                              | 2 observations, `oldest != 0` |
| `corr`       | Pearson correlation of two columns over the window            | 2 observations, both variances `> 0` |
| `ewma`       | `alpha = 1 - 0.5**(1/halflife)`, `s = alpha*x + (1-alpha)*s`   | first observation |

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
is O(N) in the series length; the in-order path is O(1).

## Storage

```
data/
  catalog.json
  features/feature=sma_20/series=BTC/part-<ns>-<id>.parquet
  raw/series=BTC/part-<ns>-<id>.parquet
  dropped/series=BTC/part-<ns>-<id>.parquet
```

- Partitioned by feature and series, percent-encoded so any name stays a single
  path component.
- Append-only files, written to a temporary name and moved atomically.
- One row per version of each `(feature, series, ts, seq)` key, with `version`
  and `reason` (`append`, `backfill`, `late_recompute`, `out_of_order`).
- The raw log is the source of truth: `FeatureStore.open` rebuilds the state by
  replaying it, and `catalog.json` restores the feature definitions.
- `flush()` and `close()` persist the pending buffer; the default threshold is
  50 000 rows.

Queries read the partition, keep the highest version per key and bisect on
`ts`. The resulting timeline is cached until the next write.

## Benchmark

Machine: AMD Ryzen 9 5900X, 32 GB DDR4-3600, Windows 11 Pro, Python 3.12.10,
NumPy 2.5.5, single-threaded. Each number is one measured run, not a best-of.

Command:

```text
python benchmarks/bench.py --ticks 10000000 --recompute-ticks 10000000 \
    --storage-ticks 200000 --scaling-ticks 30000
```

### Ten million ticks

| Path | Features | Time (s) | ns/tick | Ticks/s |
|---|---:|---:|---:|---:|
| `incremental` | 5 | 101.235 | 10,123 | 98,780 |
| `recompute_window` | 5 | 129.469 | 12,947 | 77,239 |
| `incremental + EWMA` | 6 | 130.592 | 13,059 | 76,575 |
| `batch_once` | 6 | 7.643 | 764 | 1,308,418 |

Incremental vs per-tick window recompute: **1.28x faster** for the same five features.

Window scaling, 30,000 ticks per row, same seed and same feature set:

| Window | Incremental (ns/tick) | Recompute (ns/tick) | Speedup |
|---:|---:|---:|---:|
| 20 | 11,650 | 14,708 | 1.26x |
| 100 | 11,688 | 64,545 | 5.52x |
| 500 | 10,236 | 264,314 | 25.82x |

Parquet backend, 200 000 ticks and five features (1 000 000 feature rows plus 200 000 raw rows):

- End-to-end ingestion, engine included: 24,097 ticks/s (writes are amortised every 50 000 rows).
- Reopen, which replays the raw log: 2.04 s.
- 100 point-in-time queries: 0.85 s (the first builds the timeline, the rest hit the cache).
- Remaining buffer flushed at close: 0.00 s.

### What is measured

- `incremental`: 10 million in-order ticks through `FeatureStore.null`, with the
  five windowed features (`mean`, `std`, `zscore`, `cum_return`, `corr`). No rows
  are materialised and no disk is touched; converting each NumPy chunk to Python
  floats is part of the feed loop and is included.
- `recompute_window`: the non-incremental baseline. On every tick it rescans the
  trailing window with plain Python and computes the same five statistics. No
  library shortcuts, no NumPy.
- `incremental + EWMA`: the same stream with a sixth feature, which is O(1) but
  has no window.
- `batch_once`: one vectorized NumPy pass over the whole history, the offline
  equivalent. It is fast, but it materialises every array, cannot answer a
  query until the pass ends and has no concept of late data or versions.
- `storage`: 200 000 ticks through the Parquet backend, then flush, reopen and
  100 point-in-time queries. Reported separately because it measures disk I/O,
  not feature arithmetic.

### Honest reading

The per-tick factor is modest at window 20 because both paths are dominated by
Python interpreter overhead: the incremental path does five small state updates,
the baseline does five short scans. The algorithmic difference is that the
incremental cost does not grow with the window, while the baseline does. The
scaling table above shows that: at window 500 the same machine measures 25.8
times fewer nanoseconds per tick for the incremental path.

A vectorized batch pass is faster per tick than any per-tick Python loop. That is
expected and not the comparison this library is for: the streaming path wins on
update latency, bounded memory, late-data corrections and auditable versions,
not on bulk offline throughput. If the job is "compute this array once", use
NumPy; if the job is "keep this array correct while ticks arrive", use this.

## Install

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
| Incremental equals batch bit for bit on 20 synthetic series | `test_equivalence.py::test_twenty_series_incremental_equals_batch` |
| Out-of-order data follows the configured policy | `test_late_data.py`, `test_properties.py::test_drop_policy_equals_batch_without_the_dropped_records` |
| Point-in-time has no future leakage | `test_point_in_time.py`, `test_properties.py::test_point_in_time_matches_the_prefix` |
| Parquet partitions and row versions | `test_storage.py` |
| Register, query, audit | `test_api.py`, `test_storage.py` |
| Features match the textbook definition | `test_features.py` (NumPy reference, `rtol=1e-8`) |

The suite also checks that any arrival order of the same records ends in the
same bit-exact state (`test_properties.py`), which is the strongest form of the
point-in-time property.

CI runs lint (`ruff check`, `ruff format --check`), types (`mypy`), the whole
test suite with a 90% coverage floor and a scaled-down benchmark on Linux and
Windows.

## Project layout

```text
feature-store/
├── benchmarks/
│   ├── bench.py
│   └── results/
├── docs/
│   └── design.md
├── src/feature_store/
│   ├── batch.py       # from-scratch batch oracle
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
- State is in memory. Reopening replays the raw log, which is O(N).
- Count windows only. Time-based windows are not implemented.
- A late recompute is O(N) in the series length. Periodic state checkpoints
  would make it O(tail), at the cost of more state on disk.
- Queries load the entire partition. Predicate pushdown on `ts` is future work.
- The audit table grows without compaction.
- The sliding moments have bounded floating-point drift; they are not exact in
  real arithmetic. See [docs/design.md](docs/design.md).

## License

MIT. See [LICENSE](LICENSE).
