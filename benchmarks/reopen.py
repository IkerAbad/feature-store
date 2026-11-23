"""Reopen cost: replay the raw log versus restore a checkpoint.

The benchmark isolates ``FeatureStore.open``. It writes the raw log directly
with pyarrow (in event-time order) because the cost being measured is the
restart path, not ingestion. Feature rows are not needed: ``open`` never reads
them, and leaving them out keeps the 100-million-tick run on a normal disk.

For every size it measures:

1. ``replay``: open with no checkpoint. The state machines consume every raw
   record, with ``retain_records=False`` so memory stays bounded.
2. ``replay_retained``: the same open with the default retained history, only
   for sizes where the machine has enough RAM. This is what the previous
   release did.
3. ``restore``: open with the checkpoint written after (1) plus a tail of
   ``--tail`` raw records. The tail is the only part replayed.

Usage::

    python benchmarks/reopen.py --ticks 1000000 10000000 100000000
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from feature_store import FeatureSpec, FeatureStore
from feature_store.checkpoint import read_latest

COLUMNS = ("x", "y")
RAW_SCHEMA = pa.schema(
    [
        pa.field("series", pa.string(), nullable=False),
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("seq", pa.int64(), nullable=False),
        pa.field("x", pa.float64(), nullable=False),
        pa.field("y", pa.float64(), nullable=False),
        pa.field("ingested_ns", pa.int64(), nullable=False),
    ]
)


def specs(window: int) -> list[FeatureSpec]:
    return [
        FeatureSpec.mean("mean", window),
        FeatureSpec.std("std", window),
        FeatureSpec.zscore("zscore", window),
        FeatureSpec.cum_return("ret", window),
        FeatureSpec.corr("corr", window),
        FeatureSpec.mean("tmean", time_window=2 * window),
        FeatureSpec.corr("tcorr", time_window=2 * window),
        FeatureSpec.ewma("ewma", 5.0),
    ]


def write_raw(
    root: Path,
    ticks: int,
    *,
    chunk: int,
    seed: int,
    start_ts: int = 0,
    price: float = 100.0,
    prefix: str = "part",
) -> tuple[int, int, float]:
    """Write sorted raw rows; returns (rows, files, last_price)."""
    rng = np.random.default_rng(seed)
    directory = root / "raw" / "series=S"
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    files = 0
    now = time.time_ns()
    while written < ticks:
        size = min(chunk, ticks - written)
        steps = rng.standard_normal(size)
        path_prices = price * np.exp(np.cumsum(steps * 0.01))
        price = float(path_prices[-1])
        other = path_prices * 0.5 + rng.standard_normal(size) * 0.05
        table = pa.Table.from_pydict(
            {
                "series": pa.repeat(pa.scalar("S"), size),
                "ts": np.arange(start_ts + written, start_ts + written + size, dtype=np.int64),
                "seq": np.arange(written, written + size, dtype=np.int64),
                "x": path_prices,
                "y": other,
                "ingested_ns": np.full(size, now, dtype=np.int64),
            },
            schema=RAW_SCHEMA,
        )
        name = f"{prefix}-{files:05d}.parquet"
        pq.write_table(table, directory / name, compression="zstd")
        written += size
        files += 1
    return written, files, price


def tree_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def measure_size(
    ticks: int,
    *,
    tail: int,
    chunk: int,
    seed: int,
    window: int,
    workdir: Path,
    measure_retained: bool,
) -> dict[str, object]:
    root = workdir / f"store-{ticks}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    bootstrap = FeatureStore.open(root, columns=COLUMNS, checkpoint_every=None)
    for spec in specs(window):
        bootstrap.register(spec)
    bootstrap.close()
    shutil.rmtree(root / "state", ignore_errors=True)

    started = time.perf_counter()
    rows, files, price = write_raw(root, ticks, chunk=chunk, seed=seed)
    generation = time.perf_counter() - started
    print(f"  generated {rows:,} rows in {files} files ({generation:.1f}s)", flush=True)

    started = time.perf_counter()
    replay = FeatureStore.open(root, checkpoint_every=None, retain_records=False, checkpoint_keep=1)
    replay_seconds = time.perf_counter() - started
    assert replay.state_summary()["S"]["processed"] == ticks
    print(f"  replay (bounded memory): {replay_seconds:.2f}s", flush=True)

    checkpoint_started = time.perf_counter()
    checkpoint_path = replay.checkpoint()
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    assert checkpoint_path is not None
    replay.close()
    checkpoint = read_latest(root)
    assert checkpoint is not None

    tail_rows, _, _ = write_raw(
        root,
        tail,
        chunk=tail,
        seed=seed + 1,
        start_ts=ticks,
        price=price,
        prefix="tail",
    )
    started = time.perf_counter()
    restored = FeatureStore.open(
        root, checkpoint_every=None, retain_records=False, checkpoint_keep=1
    )
    restore_seconds = time.perf_counter() - started
    summary = restored.state_summary()["S"]
    assert summary["processed"] == ticks + tail_rows, summary
    assert summary["retained"] == 0
    restored.close()
    print(f"  restore + {tail_rows:,} tail: {restore_seconds:.2f}s", flush=True)

    checkpoint_size = tree_bytes(checkpoint.path)
    raw_size = tree_bytes(root / "raw")

    retained_seconds: float | None = None
    if measure_retained:
        # Reopen from the same raw log but keeping every record and output.
        # This is the pre-checkpoint behaviour; it is only attempted when the
        # machine has enough RAM for the series.
        shutil.rmtree(root / "state", ignore_errors=True)
        started = time.perf_counter()
        retained = FeatureStore.open(root, checkpoint_every=None, retain_records=True)
        retained_seconds = time.perf_counter() - started
        assert retained.state_summary()["S"]["processed"] == ticks + tail_rows
        retained.close()
        print(f"  replay (retained history): {retained_seconds:.2f}s", flush=True)

    shutil.rmtree(root, ignore_errors=True)
    result: dict[str, object] = {
        "ticks": ticks,
        "tail": tail_rows,
        "replay_seconds": replay_seconds,
        "replay_ns_per_tick": replay_seconds / ticks * 1e9,
        "restore_seconds": restore_seconds,
        "restore_speedup": replay_seconds / restore_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "checkpoint_bytes": checkpoint_size,
        "raw_bytes": raw_size,
        "raw_files": files,
    }
    if retained_seconds is not None:
        result["replay_retained_seconds"] = retained_seconds
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ticks", type=int, nargs="+", default=[1_000_000, 10_000_000, 100_000_000]
    )
    parser.add_argument("--tail", type=int, default=100_000)
    parser.add_argument("--chunk", type=int, default=5_000_000)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--retained-max",
        type=int,
        default=10_000_000,
        help="measure the retained-history replay up to this size (0 disables)",
    )
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/reopen.json"))
    args = parser.parse_args()

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="feature-store-reopen-"))
    workdir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        },
        "config": {
            "tail": args.tail,
            "chunk": args.chunk,
            "window": args.window,
            "seed": args.seed,
            "features": len(specs(args.window)),
        },
        "results": [],
    }
    results: list[dict[str, object]] = []
    print(
        f"reopen benchmark | python {platform.python_version()} | "
        f"{platform.platform()} | tail={args.tail:,} | window={args.window}",
        flush=True,
    )
    for ticks in sorted(args.ticks):
        print(f"ticks={ticks:,}", flush=True)
        try:
            result = measure_size(
                ticks,
                tail=args.tail,
                chunk=args.chunk,
                seed=args.seed,
                window=args.window,
                workdir=workdir,
                measure_retained=(args.retained_max > 0 and ticks <= args.retained_max),
            )
        except MemoryError as error:
            result = {"ticks": ticks, "error": f"MemoryError: {error}"}
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    payload["results"] = results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    if args.workdir is None:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
