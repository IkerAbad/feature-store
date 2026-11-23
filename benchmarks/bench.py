"""Reproducible benchmark: incremental updates versus recomputation.

The primary comparison is the one that matters for a feature store: keep the
state and update one tick at a time versus recompute the trailing window from
scratch on every tick. A single vectorized batch pass over the whole history is
also measured, because it is the honest floor for offline work: it is fast, but
it materialises everything, cannot answer a query before the pass ends and has
no notion of late data or versions.

Run the published configuration::

    python benchmarks/bench.py --ticks 10000000 --recompute-ticks 10000000

Run a quick smoke configuration::

    python benchmarks/bench.py --ticks 50000 --recompute-ticks 50000 --storage-ticks 5000
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import tempfile
import time
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from feature_store import FeatureSpec, FeatureStore

COLUMNS = ("x", "y")


def windowed_specs(window: int) -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec.mean(f"mean_{window}", window),
        FeatureSpec.std(f"std_{window}", window),
        FeatureSpec.zscore(f"zscore_{window}", window),
        FeatureSpec.cum_return(f"ret_{window}", window),
        FeatureSpec.corr(f"corr_{window}", window),
    )


@dataclass(frozen=True)
class Timing:
    name: str
    ticks: int
    seconds: float
    features: int

    @property
    def ns_per_tick(self) -> float:
        return self.seconds / self.ticks * 1e9

    @property
    def ticks_per_second(self) -> float:
        return self.ticks / self.seconds


def chunk_stream(
    ticks: int, chunk_size: int, seed: int
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Deterministic random walk yielded in chunks, so memory stays bounded."""
    rng = np.random.default_rng(seed)
    price = 100.0
    produced = 0
    while produced < ticks:
        size = min(chunk_size, ticks - produced)
        steps = rng.standard_normal(size)
        path = price * np.exp(np.cumsum(steps * 0.01))
        price = float(path[-1])
        noise = rng.standard_normal(size)
        other = path * 0.5 + noise * 0.05
        yield produced, path, other
        produced += size


def run_incremental(
    ticks: int,
    chunk_size: int,
    seed: int,
    specs: Sequence[FeatureSpec],
    *,
    retain_records: bool = False,
) -> Timing:
    store = FeatureStore.null(columns=COLUMNS, retain_records=retain_records)
    for spec in specs:
        store.register(spec)
    start = time.perf_counter()
    for offset, xs, ys in chunk_stream(ticks, chunk_size, seed):
        for index, (x, y) in enumerate(zip(xs.tolist(), ys.tolist(), strict=True)):
            store.push_values("SYM", offset + index, (x, y))
    seconds = time.perf_counter() - start
    name = "incremental" if not retain_records else "incremental_retained"
    return Timing(name, ticks, seconds, len(specs))


def run_recompute_window(ticks: int, chunk_size: int, seed: int, window: int) -> Timing:
    """Naive baseline: recompute every windowed feature from the window itself."""
    x_window: deque[float] = deque(maxlen=window)
    y_window: deque[float] = deque(maxlen=window)
    start = time.perf_counter()
    for _offset, xs, ys in chunk_stream(ticks, chunk_size, seed):
        for x, y in zip(xs.tolist(), ys.tolist(), strict=True):
            x_window.append(x)
            y_window.append(y)
            n = len(x_window)
            if n < 2:
                continue
            mean_x = sum(x_window) / n
            mean_y = sum(y_window) / n
            var_x = sum((value - mean_x) ** 2 for value in x_window) / n
            var_y = sum((value - mean_y) ** 2 for value in y_window) / n
            std_x = math.sqrt(var_x)
            _zscore = (x - mean_x) / std_x if std_x else math.nan
            _return = x / x_window[0] - 1.0
            if var_x and var_y:
                covariance = (
                    sum(
                        (left - mean_x) * (right - mean_y)
                        for left, right in zip(x_window, y_window, strict=True)
                    )
                    / n
                )
                _corr = covariance / math.sqrt(var_x * var_y)
            else:
                _corr = math.nan
    seconds = time.perf_counter() - start
    return Timing("recompute_window", ticks, seconds, len(windowed_specs(window)))


def _rolling(values: np.ndarray, window: int, reducer: str) -> np.ndarray:
    """Vectorized rolling statistic, used only as the offline batch baseline."""
    padded = np.concatenate([np.full(window - 1, np.nan), values])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    result = np.empty(len(values))
    result[: window - 1] = np.nan
    if reducer == "mean":
        result[window - 1 :] = windows[window - 1 :].mean(axis=1)
    elif reducer == "std":
        result[window - 1 :] = windows[window - 1 :].std(axis=1)
    elif reducer == "max":
        result[window - 1 :] = windows[window - 1 :].max(axis=1)
    else:  # pragma: no cover
        raise ValueError(reducer)
    return result


def run_batch_once(ticks: int, chunk_size: int, seed: int, window: int) -> Timing:
    """One vectorized pass over the entire history (offline equivalent)."""
    xs_parts: list[np.ndarray] = []
    ys_parts: list[np.ndarray] = []
    for _, xs, ys in chunk_stream(ticks, chunk_size, seed):
        xs_parts.append(xs)
        ys_parts.append(ys)
    xs = np.concatenate(xs_parts)
    ys = np.concatenate(ys_parts)
    start = time.perf_counter()
    means = _rolling(xs, window, "mean")
    stds = _rolling(xs, window, "std")
    _zscore = (xs - means) / stds
    oldest = np.concatenate([np.full(window - 1, np.nan), xs[: ticks - window + 1]])
    _return = xs / oldest - 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        covariance = _rolling(xs * ys, window, "mean") - means * _rolling(ys, window, "mean")
        _corr = covariance / (stds * _rolling(ys, window, "std"))
    # EWMA is recursive, so the offline pass still needs a sequential loop.
    alpha = 1.0 - 0.5 ** (1.0 / 5.0)
    beta = 1.0 - alpha
    ewma = np.empty(ticks)
    value = float(xs[0])
    for index in range(ticks):
        value = alpha * float(xs[index]) + beta * value
        ewma[index] = value
    seconds = time.perf_counter() - start
    return Timing("batch_once", ticks, seconds, len(windowed_specs(window)) + 1)


def run_storage(
    ticks: int, chunk_size: int, seed: int, root: Path, window: int
) -> dict[str, float]:
    """End-to-end Parquet cost: ingest, flush, reopen and query.

    Writes are buffered and flushed every 50 000 rows (the default threshold),
    so most of the disk work happens during ingestion and the final flush only
    writes the remainder. ``flush_seconds`` reports that remainder.
    """
    store = FeatureStore.open(root, columns=COLUMNS)
    for spec in windowed_specs(window):
        store.register(spec)
    start = time.perf_counter()
    for offset, xs, ys in chunk_stream(ticks, chunk_size, seed):
        for index, (x, y) in enumerate(zip(xs.tolist(), ys.tolist(), strict=True)):
            store.push_values("SYM", offset + index, (x, y))
    ingest = time.perf_counter() - start
    start = time.perf_counter()
    store.close()
    write = time.perf_counter() - start
    start = time.perf_counter()
    reopened = FeatureStore.open(root)
    load = time.perf_counter() - start
    start = time.perf_counter()
    for step in range(100):
        ts = int(step * (ticks - 1) / 99)
        reopened.point_in_time(f"mean_{window}", "SYM", ts)
    query = time.perf_counter() - start
    reopened.close()
    return {
        "ticks": ticks,
        "ingest_seconds": ingest,
        "flush_seconds": write,
        "reopen_seconds": load,
        "queries": 100,
        "query_seconds": query,
        "rows_per_second_ingest": ticks / ingest,
    }


def measure_all(args: argparse.Namespace) -> dict[str, object]:
    results: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "ticks": args.ticks,
            "recompute_ticks": args.recompute_ticks,
            "chunk": args.chunk,
            "seed": args.seed,
            "window": args.window,
            "storage_ticks": args.storage_ticks,
            "scaling_ticks": args.scaling_ticks,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        },
    }
    timings: list[Timing] = []
    windowed = windowed_specs(args.window)
    timings.append(run_incremental(args.ticks, args.chunk, args.seed, windowed))
    timings.append(run_recompute_window(args.recompute_ticks, args.chunk, args.seed, args.window))
    timings.append(
        run_incremental(args.ticks, args.chunk, args.seed, windowed, retain_records=True)
    )
    with_ewma = (*windowed, FeatureSpec.ewma("ewma_5", 5.0))
    timings.append(run_incremental(args.ticks, args.chunk, args.seed, with_ewma))
    timings.append(run_batch_once(args.ticks, args.chunk, args.seed, args.window))
    results["timings"] = [
        asdict(timing) | {"ns_per_tick": timing.ns_per_tick} for timing in timings
    ]

    incremental = timings[0]
    recompute = timings[1]
    results["speedup_vs_recompute"] = recompute.seconds / incremental.seconds

    if args.scaling_ticks:
        scaling = []
        for window in (20, 100, 500):
            specs = windowed_specs(window)
            fast = run_incremental(args.scaling_ticks, args.chunk, args.seed, specs)
            slow = run_recompute_window(args.scaling_ticks, args.chunk, args.seed, window)
            scaling.append(
                {
                    "window": window,
                    "incremental_ns_per_tick": fast.ns_per_tick,
                    "recompute_ns_per_tick": slow.ns_per_tick,
                    "speedup": slow.seconds / fast.seconds,
                }
            )
        results["scaling"] = scaling

    if args.storage_ticks:
        storage_root = args.storage_root or Path(tempfile.mkdtemp(prefix="feature-store-bench-"))
        results["storage"] = run_storage(
            args.storage_ticks, args.chunk, args.seed, Path(storage_root), args.window
        )
    return results


def print_report(results: dict[str, object]) -> None:
    config = results["config"]
    timings = results["timings"]
    environment = results["environment"]
    assert isinstance(config, dict) and isinstance(timings, list) and isinstance(environment, dict)
    print(
        f"ticks={config['ticks']:,}  window={config['window']}  seed={config['seed']}  "
        f"python={environment['python']}  numpy={environment['numpy']}"
    )
    print()
    print(f"{'path':<18}{'features':>9}{'seconds':>12}{'ns/tick':>12}{'ticks/s':>14}")
    for item in timings:
        assert isinstance(item, dict)
        print(
            f"{item['name']:<18}{item['features']:>9}{item['seconds']:>12.3f}"
            f"{item['ns_per_tick']:>12.1f}{item['ticks'] / item['seconds']:>14,.0f}"
        )
    speedup = float(results["speedup_vs_recompute"])  # type: ignore[arg-type]
    print()
    print(f"incremental vs per-tick window recompute: {speedup:.2f}x faster")

    scaling = results.get("scaling")
    if isinstance(scaling, list):
        print()
        print(f"{'window':>8}{'incremental ns/tick':>22}{'recompute ns/tick':>21}{'speedup':>10}")
        for row in scaling:
            print(
                f"{row['window']:>8}{row['incremental_ns_per_tick']:>22.1f}"
                f"{row['recompute_ns_per_tick']:>21.1f}{row['speedup']:>9.2f}x"
            )

    storage = results.get("storage")
    if isinstance(storage, dict):
        print()
        print(
            f"parquet ingest {storage['rows_per_second_ingest']:,.0f} rows/s, "
            f"flush {storage['flush_seconds']:.2f}s, reopen {storage['reopen_seconds']:.2f}s, "
            f"{storage['queries']} point-in-time queries in {storage['query_seconds']:.3f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=1_000_000)
    parser.add_argument("--recompute-ticks", type=int, default=None)
    parser.add_argument("--storage-ticks", type=int, default=200_000)
    parser.add_argument("--scaling-ticks", type=int, default=30_000)
    parser.add_argument("--chunk", type=int, default=1_000_000)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/out/bench.json"))
    parser.add_argument("--storage-root", type=Path, default=None)
    args = parser.parse_args()
    if args.recompute_ticks is None:
        args.recompute_ticks = args.ticks
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = measure_all(args)
    print_report(results)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
