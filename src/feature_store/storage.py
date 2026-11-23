"""Row storage backends: append-only Parquet partitions, memory and a null sink.

Layout on disk::

    root/
    ├── catalog.json
    ├── features/
    │   └── feature=<name>/series=<series>/part-<ns>-<id>.parquet
    ├── raw/
    │   └── series=<series>/part-<ns>-<id>.parquet
    └── dropped/
        └── series=<series>/part-<ns>-<id>.parquet

Every write creates a new immutable file, so a reader never sees a half-written
partition and two writers cannot overwrite each other's rows. The feature
table is append-only and keeps one row per version of every key, which is what
makes the audit trail exact.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

_RAW_META = frozenset({"series", "ts", "seq", "ingested_ns"})

_FEATURE_SCHEMA = pa.schema(
    [
        pa.field("feature", pa.string(), nullable=False),
        pa.field("series", pa.string(), nullable=False),
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("seq", pa.int64(), nullable=False),
        pa.field("value", pa.float64(), nullable=False),
        pa.field("version", pa.int32(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
        pa.field("ingested_ns", pa.int64(), nullable=False),
    ]
)

_DROPPED_SCHEMA = pa.schema(
    [
        pa.field("series", pa.string(), nullable=False),
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("seq", pa.int64(), nullable=False),
        pa.field("watermark", pa.int64(), nullable=False),
        pa.field("policy", pa.string(), nullable=False),
        pa.field("ingested_ns", pa.int64(), nullable=False),
    ]
)


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """One version of one feature value at one event-time key."""

    feature: str
    series: str
    ts: int
    seq: int
    value: float
    version: int
    reason: str
    ingested_ns: int


@dataclass(frozen=True, slots=True)
class RawRecord:
    """An accepted input record, kept for replay and late-data recomputation."""

    series: str
    ts: int
    seq: int
    values: tuple[float, ...]
    ingested_ns: int


@dataclass(frozen=True, slots=True)
class DroppedRecord:
    """A record rejected by the late-data policy."""

    series: str
    ts: int
    seq: int
    watermark: int
    policy: str
    ingested_ns: int


class Storage(Protocol):
    """Minimal contract the feature store needs from a backend."""

    wants_rows: bool

    def write_rows(self, rows: Sequence[FeatureRow]) -> None: ...

    def write_raw(self, records: Sequence[RawRecord]) -> None: ...

    def write_dropped(self, records: Sequence[DroppedRecord]) -> None: ...

    def rows(self, feature: str, series: str) -> list[FeatureRow]: ...

    def read_raw(self) -> list[RawRecord]: ...

    def read_dropped(self) -> list[DroppedRecord]: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class MemoryStorage:
    """In-memory backend, useful for tests and notebooks."""

    wants_rows = True

    def __init__(self) -> None:
        self._rows: list[FeatureRow] = []
        self._raw: list[RawRecord] = []
        self._dropped: list[DroppedRecord] = []

    def write_rows(self, rows: Sequence[FeatureRow]) -> None:
        self._rows.extend(rows)

    def write_raw(self, records: Sequence[RawRecord]) -> None:
        self._raw.extend(records)

    def write_dropped(self, records: Sequence[DroppedRecord]) -> None:
        self._dropped.extend(records)

    def rows(self, feature: str, series: str) -> list[FeatureRow]:
        return [row for row in self._rows if row.feature == feature and row.series == series]

    def read_raw(self) -> list[RawRecord]:
        return list(self._raw)

    def read_dropped(self) -> list[DroppedRecord]:
        return list(self._dropped)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class NullStorage:
    """Discard everything, skipping row construction in the store.

    The store checks ``wants_rows`` before building any row, which is what makes
    the engine-only benchmark possible.
    """

    wants_rows = False

    def __init__(self) -> None:
        self.rows_written = 0
        self.raw_written = 0
        self.dropped_written = 0

    def write_rows(self, rows: Sequence[FeatureRow]) -> None:
        self.rows_written += len(rows)

    def write_raw(self, records: Sequence[RawRecord]) -> None:
        self.raw_written += len(records)

    def write_dropped(self, records: Sequence[DroppedRecord]) -> None:
        self.dropped_written += len(records)

    def rows(self, feature: str, series: str) -> list[FeatureRow]:
        return []

    def read_raw(self) -> list[RawRecord]:
        return []

    def read_dropped(self) -> list[DroppedRecord]:
        return []

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class ParquetStorage:
    """Append-only Parquet backend, partitioned by feature and series."""

    wants_rows = True

    def __init__(
        self, root: str | Path, columns: Sequence[str], *, flush_every: int = 50_000
    ) -> None:
        self.root = Path(root)
        self.columns = tuple(columns)
        self.flush_every = flush_every
        self._row_buffer: list[FeatureRow] = []
        self._raw_buffer: list[RawRecord] = []
        self._dropped_buffer: list[DroppedRecord] = []
        self._features_dir = self.root / "features"
        self._raw_dir = self.root / "raw"
        self._dropped_dir = self.root / "dropped"
        for directory in (self._features_dir, self._raw_dir, self._dropped_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # -- writes ---------------------------------------------------------
    def write_rows(self, rows: Sequence[FeatureRow]) -> None:
        self._row_buffer.extend(rows)
        if len(self._row_buffer) >= self.flush_every:
            self._flush_rows()

    def write_raw(self, records: Sequence[RawRecord]) -> None:
        self._raw_buffer.extend(records)
        if len(self._raw_buffer) >= self.flush_every:
            self._flush_raw()

    def write_dropped(self, records: Sequence[DroppedRecord]) -> None:
        self._dropped_buffer.extend(records)
        if len(self._dropped_buffer) >= self.flush_every:
            self._flush_dropped()

    def flush(self) -> None:
        self._flush_rows()
        self._flush_raw()
        self._flush_dropped()

    def close(self) -> None:
        self.flush()

    # -- reads ----------------------------------------------------------
    def rows(self, feature: str, series: str) -> list[FeatureRow]:
        rows = [row for row in self._row_buffer if row.feature == feature and row.series == series]
        table = _read_partition(self._feature_partition(feature, series))
        if table is not None:
            rows.extend(
                FeatureRow(
                    feature=item["feature"],
                    series=item["series"],
                    ts=item["ts"],
                    seq=item["seq"],
                    value=item["value"],
                    version=item["version"],
                    reason=item["reason"],
                    ingested_ns=item["ingested_ns"],
                )
                for item in table.to_pylist()
            )
        return rows

    def read_raw(self) -> list[RawRecord]:
        records: list[RawRecord] = list(self._raw_buffer)
        for directory in sorted(self._raw_dir.glob("series=*")):
            table = _read_partition(directory)
            if table is None:
                continue
            names = [name for name in table.schema.names if name not in _RAW_META]
            for item in table.to_pylist():
                records.append(
                    RawRecord(
                        series=item["series"],
                        ts=item["ts"],
                        seq=item["seq"],
                        values=tuple(item[name] for name in names),
                        ingested_ns=item["ingested_ns"],
                    )
                )
        return records

    def read_dropped(self) -> list[DroppedRecord]:
        records: list[DroppedRecord] = list(self._dropped_buffer)
        for directory in sorted(self._dropped_dir.glob("series=*")):
            table = _read_partition(directory)
            if table is None:
                continue
            for item in table.to_pylist():
                records.append(
                    DroppedRecord(
                        series=item["series"],
                        ts=item["ts"],
                        seq=item["seq"],
                        watermark=item["watermark"],
                        policy=item["policy"],
                        ingested_ns=item["ingested_ns"],
                    )
                )
        return records

    # -- internals ------------------------------------------------------
    def _feature_partition(self, feature: str, series: str) -> Path:
        return (
            self._features_dir
            / f"feature={quote(feature, safe='')}"
            / f"series={quote(series, safe='')}"
        )

    def _flush_rows(self) -> None:
        if not self._row_buffer:
            return
        grouped: dict[tuple[str, str], list[FeatureRow]] = {}
        for row in self._row_buffer:
            grouped.setdefault((row.feature, row.series), []).append(row)
        for (feature, series), rows in grouped.items():
            table = pa.Table.from_pylist(
                [
                    {
                        "feature": row.feature,
                        "series": row.series,
                        "ts": row.ts,
                        "seq": row.seq,
                        "value": row.value,
                        "version": row.version,
                        "reason": row.reason,
                        "ingested_ns": row.ingested_ns,
                    }
                    for row in rows
                ],
                schema=_FEATURE_SCHEMA,
            )
            _write_table(table, self._feature_partition(feature, series))
        self._row_buffer.clear()

    def _flush_raw(self) -> None:
        if not self._raw_buffer:
            return
        grouped: dict[str, list[RawRecord]] = {}
        for record in self._raw_buffer:
            grouped.setdefault(record.series, []).append(record)
        schema = pa.schema(
            [
                pa.field("series", pa.string(), nullable=False),
                pa.field("ts", pa.int64(), nullable=False),
                pa.field("seq", pa.int64(), nullable=False),
                *[pa.field(name, pa.float64(), nullable=False) for name in self.columns],
                pa.field("ingested_ns", pa.int64(), nullable=False),
            ]
        )
        for series, records in grouped.items():
            payload = []
            for record in records:
                item: dict[str, object] = {
                    "series": record.series,
                    "ts": record.ts,
                    "seq": record.seq,
                    "ingested_ns": record.ingested_ns,
                }
                for name, value in zip(self.columns, record.values, strict=True):
                    item[name] = value
                payload.append(item)
            table = pa.Table.from_pylist(payload, schema=schema)
            _write_table(table, self._raw_dir / f"series={quote(series, safe='')}")
        self._raw_buffer.clear()

    def _flush_dropped(self) -> None:
        if not self._dropped_buffer:
            return
        grouped: dict[str, list[DroppedRecord]] = {}
        for record in self._dropped_buffer:
            grouped.setdefault(record.series, []).append(record)
        for series, records in grouped.items():
            table = pa.Table.from_pylist(
                [
                    {
                        "series": record.series,
                        "ts": record.ts,
                        "seq": record.seq,
                        "watermark": record.watermark,
                        "policy": record.policy,
                        "ingested_ns": record.ingested_ns,
                    }
                    for record in records
                ],
                schema=_DROPPED_SCHEMA,
            )
            _write_table(table, self._dropped_dir / f"series={quote(series, safe='')}")
        self._dropped_buffer.clear()


def _read_partition(directory: Path) -> pa.Table | None:
    if not directory.exists():
        return None
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return None
    dataset = ds.dataset([str(file) for file in files], format="parquet")
    return dataset.to_table()


def _write_table(table: pa.Table, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    name = f"part-{time.time_ns()}-{uuid.uuid4().hex[:8]}.parquet"
    temporary = directory / (name + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, directory / name)
