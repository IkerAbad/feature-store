"""Public feature store API.

The store owns three things: the catalogue of registered features, the
in-memory incremental state of each series, and a storage backend. It is a
single-writer object: one process, one thread at a time.
"""

from __future__ import annotations

import json
import math
import os
import time
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from .engine import SeriesEngine
from .errors import (
    DuplicateFeatureError,
    DuplicateRecordError,
    FeatureStoreError,
    UnknownColumnError,
    UnknownFeatureError,
)
from .spec import FeatureSpec, LatePolicy
from .storage import (
    DroppedRecord,
    FeatureRow,
    MemoryStorage,
    NullStorage,
    ParquetStorage,
    RawRecord,
    Storage,
)

_CATALOG_NAME = "catalog.json"
_CATALOG_SCHEMA = 1
NaN = float("nan")


class IngestResult(NamedTuple):
    """Outcome of one :meth:`FeatureStore.ingest` call.

    ``status`` is ``applied``, ``out_of_order``, ``late_recomputed`` or
    ``late_dropped``. A tuple, not a dataclass, to keep the per-tick cost low.
    """

    status: str
    series: str
    ts: int
    seq: int
    corrections: int = 0
    watermark: int | None = None


class StoreStats(NamedTuple):
    """Cheap counters describing the current state of the store."""

    features: int
    series: int
    records: int
    corrections: int
    late_dropped: int
    rows_emitted: int


class FeatureStore:
    """Incremental feature store over event-time ordered records.

    Use :meth:`open` for the persistent Parquet backend, :meth:`memory` for an
    ephemeral one and :meth:`null` for benchmarks.
    """

    def __init__(
        self,
        storage: Storage,
        *,
        columns: Sequence[str] = ("x", "y"),
        late_policy: LatePolicy | str = LatePolicy.DROP,
        allowed_lateness: int = 0,
        root: str | Path | None = None,
    ) -> None:
        if not columns:
            raise FeatureStoreError("at least one input column is required")
        if any(not isinstance(column, str) or not column for column in columns):
            raise FeatureStoreError("input column names must be non-empty strings")
        if len(set(columns)) != len(columns):
            raise FeatureStoreError("input column names must be unique")
        if allowed_lateness < 0:
            raise FeatureStoreError("allowed_lateness must be >= 0")
        self._storage = storage
        self._columns = tuple(columns)
        self._late_policy = LatePolicy(late_policy)
        self._allowed_lateness = int(allowed_lateness)
        self._root = Path(root) if root is not None else None
        self._specs: dict[str, FeatureSpec] = {}
        self._series: dict[str, SeriesEngine] = {}
        self._timelines: dict[tuple[str, str], tuple[list[FeatureRow], list[int]]] = {}
        self._versions: dict[tuple[str, str, int, int], int] = {}
        self._corrections = 0
        self._dropped = 0
        self._rows_emitted = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        columns: Sequence[str] | None = None,
        late_policy: LatePolicy | str | None = None,
        allowed_lateness: int | None = None,
        flush_every: int = 50_000,
    ) -> FeatureStore:
        """Open (or create) a persistent store rooted at ``root``.

        On an existing store the catalogue, the late-data configuration and the
        raw record log are read back, and every in-memory state machine is
        rebuilt by replaying the accepted records in event-time order.
        """
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        catalog = _read_catalog(path)
        if catalog is None:
            effective_columns = tuple(columns) if columns is not None else ("x", "y")
            effective_policy = (
                LatePolicy(late_policy) if late_policy is not None else LatePolicy.DROP
            )
            effective_lateness = 0 if allowed_lateness is None else int(allowed_lateness)
            storage = ParquetStorage(path, effective_columns, flush_every=flush_every)
            store = cls(
                storage,
                columns=effective_columns,
                late_policy=effective_policy,
                allowed_lateness=effective_lateness,
                root=path,
            )
            store._write_catalog()
            return store

        stored_columns = tuple(catalog["columns"])
        if columns is not None and tuple(columns) != stored_columns:
            raise FeatureStoreError(
                f"store was created with columns {stored_columns}, got {tuple(columns)}"
            )
        effective_policy = (
            LatePolicy(late_policy)
            if late_policy is not None
            else LatePolicy(catalog["late_policy"])
        )
        effective_lateness = (
            int(allowed_lateness)
            if allowed_lateness is not None
            else int(catalog["allowed_lateness"])
        )
        storage = ParquetStorage(path, stored_columns, flush_every=flush_every)
        store = cls(
            storage,
            columns=stored_columns,
            late_policy=effective_policy,
            allowed_lateness=effective_lateness,
            root=path,
        )
        store._specs = {
            spec.name: spec
            for spec in (FeatureSpec.from_dict(item) for item in catalog.get("features", ()))
        }
        store._replay_raw()
        store._write_catalog()
        return store

    @classmethod
    def memory(
        cls,
        *,
        columns: Sequence[str] = ("x", "y"),
        late_policy: LatePolicy | str = LatePolicy.DROP,
        allowed_lateness: int = 0,
    ) -> FeatureStore:
        """Ephemeral store backed by Python lists."""
        return cls(
            MemoryStorage(),
            columns=columns,
            late_policy=late_policy,
            allowed_lateness=allowed_lateness,
        )

    @classmethod
    def null(
        cls,
        *,
        columns: Sequence[str] = ("x", "y"),
        late_policy: LatePolicy | str = LatePolicy.DROP,
        allowed_lateness: int = 0,
    ) -> FeatureStore:
        """Store that keeps state but writes no rows. Used by the benchmark."""
        return cls(
            NullStorage(),
            columns=columns,
            late_policy=late_policy,
            allowed_lateness=allowed_lateness,
        )

    # ------------------------------------------------------------------
    # Catalogue
    # ------------------------------------------------------------------
    @property
    def columns(self) -> tuple[str, ...]:
        return self._columns

    @property
    def late_policy(self) -> LatePolicy:
        return self._late_policy

    @property
    def allowed_lateness(self) -> int:
        return self._allowed_lateness

    def features(self) -> tuple[FeatureSpec, ...]:
        return tuple(self._specs.values())

    def series(self) -> tuple[str, ...]:
        return tuple(self._series)

    def register(self, spec: FeatureSpec) -> FeatureSpec:
        """Register a feature and backfill it over the records already seen.

        Re-registering an identical spec is a no-op. A different spec under an
        existing name is an error: the audit history of a name must stay
        meaningful.
        """
        if not isinstance(spec, FeatureSpec):
            raise TypeError("register expects a FeatureSpec")
        missing = sorted(set(spec.inputs) - set(self._columns))
        if missing:
            raise UnknownColumnError(
                f"feature {spec.name!r} needs columns {missing}, store has {list(self._columns)}"
            )
        existing = self._specs.get(spec.name)
        if existing is not None:
            if existing == spec:
                return existing
            raise DuplicateFeatureError(f"feature {spec.name!r} already exists with another spec")
        self._specs[spec.name] = spec
        for engine in self._series.values():
            values = engine.add_feature(spec)
            if engine.records and self._storage.wants_rows:
                self._emit_backfill(engine, spec, values)
        self._invalidate()
        self._write_catalog()
        return spec

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest(
        self,
        series: str,
        ts: int,
        *,
        seq: int | None = None,
        **values: float,
    ) -> IngestResult:
        """Feed one record.

        ``ts`` is event time in integer units (nanoseconds by convention) and
        is the only clock used by the features. ``seq`` breaks ties between
        records with the same ``ts``. When omitted it is taken from a
        per-series arrival counter, so two records with the same timestamp stay
        distinct.
        """
        return self._ingest(series, ts, seq, self._encode(values))

    def push(self, series: str, ts: int, value: float, *, seq: int | None = None) -> IngestResult:
        """Single-column shortcut: feed ``value`` as column ``x``."""
        return self._ingest(series, ts, seq, self._encode({"x": value}))

    def push_values(
        self, series: str, ts: int, values: Sequence[float], *, seq: int | None = None
    ) -> IngestResult:
        """Low-level path: ``values`` are already aligned with :attr:`columns`."""
        if len(values) != len(self._columns):
            raise ValueError(f"expected {len(self._columns)} values, got {len(values)}")
        encoded = tuple(float(value) for value in values)
        for value in encoded:
            if math.isinf(value):
                raise ValueError("feature inputs must be finite or NaN")
        return self._ingest(series, ts, seq, encoded)

    def _encode(self, values: Mapping[str, float]) -> tuple[float, ...]:
        if not values:
            raise ValueError("at least one input column is required")
        unknown = sorted(set(values) - set(self._columns))
        if unknown:
            raise UnknownColumnError(f"unknown columns {unknown}, store has {list(self._columns)}")
        encoded = []
        for column in self._columns:
            raw = values.get(column, NaN)
            value = float(raw)
            if math.isinf(value):
                raise ValueError("feature inputs must be finite or NaN")
            encoded.append(value)
        return tuple(encoded)

    def _ingest(
        self, series: str, ts: int, seq: int | None, values: tuple[float, ...]
    ) -> IngestResult:
        engine = self._engine(series)
        if seq is None:
            seq = engine.next_seq
        key = (ts, seq)
        if not engine.keys or key > engine.keys[-1]:
            position = len(engine.keys)
        else:
            position = bisect_right(engine.keys, key)
            if position and engine.keys[position - 1] == key:
                raise DuplicateRecordError(
                    f"record ({ts}, {seq}) already exists for series {series!r}"
                )

        watermark = None if engine.max_ts is None else engine.max_ts - self._allowed_lateness
        if watermark is not None and ts < watermark:
            if self._late_policy is LatePolicy.DROP:
                self._dropped += 1
                self._storage.write_dropped(
                    [
                        DroppedRecord(
                            series=series,
                            ts=ts,
                            seq=seq,
                            watermark=watermark,
                            policy=self._late_policy.value,
                            ingested_ns=time.time_ns(),
                        )
                    ]
                )
                return IngestResult("late_dropped", series, ts, seq, 0, watermark)
            reason = "late_recompute"
            status = "late_recomputed"
        elif position == len(engine.keys):
            reason = "append"
            status = "applied"
        else:
            reason = "out_of_order"
            status = "out_of_order"

        if position == len(engine.keys):
            engine.append(ts, seq, values)
            changes: list[tuple[str, int, float, float]] = []
        else:
            changes = engine.insert(position, ts, seq, values)
        self._invalidate()

        if self._storage.wants_rows:
            self._storage.write_raw(
                [
                    RawRecord(
                        series=series,
                        ts=ts,
                        seq=seq,
                        values=values,
                        ingested_ns=time.time_ns(),
                    )
                ]
            )
            self._emit_new_row(engine, position, reason)
            if changes:
                self._emit_corrections(engine, changes, reason)
        self._corrections += len(changes)
        return IngestResult(status, series, ts, seq, len(changes), watermark)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get(self, feature: str, series: str) -> float | None:
        """Latest event-time value of ``feature`` for ``series``."""
        timeline, _ = self._timeline(feature, series)
        if not timeline:
            return None
        return timeline[-1].value

    def point_in_time(self, feature: str, series: str, ts: int) -> float | None:
        """Value of ``feature`` at the last key with ``key.ts <= ts``.

        This is the point-in-time contract: no record with a greater timestamp
        can influence the result, ever.
        """
        timeline, timestamps = self._timeline(feature, series)
        index = bisect_right(timestamps, ts)
        if index == 0:
            return None
        return timeline[index - 1].value

    def history(self, feature: str, series: str, *, limit: int | None = None) -> list[FeatureRow]:
        """Every stored version of every key, ordered by ``(ts, seq, version)``."""
        self._require_feature(feature)
        rows = self._storage.rows(feature, series)
        rows.sort(key=lambda row: (row.ts, row.seq, row.version))
        if limit is not None:
            rows = rows[-limit:]
        return rows

    def late_drops(self) -> list[DroppedRecord]:
        """Records rejected by the late-data policy, in insertion order."""
        rows = self._storage.read_dropped()
        rows.sort(key=lambda record: (record.series, record.ts, record.seq))
        return rows

    def stats(self) -> StoreStats:
        return StoreStats(
            features=len(self._specs),
            series=len(self._series),
            records=sum(len(engine) for engine in self._series.values()),
            corrections=self._corrections,
            late_dropped=self._dropped,
            rows_emitted=self._rows_emitted,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def flush(self) -> None:
        self._storage.flush()

    def close(self) -> None:
        self._storage.close()

    def __enter__(self) -> FeatureStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _engine(self, series: str) -> SeriesEngine:
        engine = self._series.get(series)
        if engine is None:
            engine = SeriesEngine(series, self._columns)
            for spec in self._specs.values():
                engine.add_feature(spec)
            self._series[series] = engine
        return engine

    def _require_feature(self, feature: str) -> FeatureSpec:
        spec = self._specs.get(feature)
        if spec is None:
            raise UnknownFeatureError(f"feature {feature!r} is not registered")
        return spec

    def _timeline(self, feature: str, series: str) -> tuple[list[FeatureRow], list[int]]:
        self._require_feature(feature)
        cached = self._timelines.get((feature, series))
        if cached is not None:
            return cached
        latest: dict[tuple[int, int], FeatureRow] = {}
        for row in self._storage.rows(feature, series):
            key = (row.ts, row.seq)
            previous = latest.get(key)
            if previous is None or row.version > previous.version:
                latest[key] = row
        timeline = [latest[key] for key in sorted(latest)]
        timestamps = [row.ts for row in timeline]
        cached = (timeline, timestamps)
        self._timelines[(feature, series)] = cached
        return cached

    def _invalidate(self) -> None:
        self._timelines.clear()

    def _current_version(self, feature: str, series: str, ts: int, seq: int) -> int:
        key = (feature, series, ts, seq)
        cached = self._versions.get(key)
        if cached is not None:
            return cached
        rows = self._storage.rows(feature, series)
        version = max(
            (row.version for row in rows if row.ts == ts and row.seq == seq),
            default=0,
        )
        self._versions[key] = version
        return version

    def _emit_new_row(self, engine: SeriesEngine, position: int, reason: str) -> None:
        ts, seq, _ = engine.records[position]
        now = time.time_ns()
        rows = [
            FeatureRow(
                feature=slot.spec.name,
                series=engine.series,
                ts=ts,
                seq=seq,
                value=slot.values[position],
                version=1,
                reason=reason,
                ingested_ns=now,
            )
            for slot in engine.slots
        ]
        for row in rows:
            self._versions[(row.feature, row.series, row.ts, row.seq)] = 1
        self._rows_emitted += len(rows)
        self._storage.write_rows(rows)

    def _emit_corrections(
        self,
        engine: SeriesEngine,
        changes: Sequence[tuple[str, int, float, float]],
        reason: str,
    ) -> None:
        now = time.time_ns()
        rows = []
        for feature, index, _old, value in changes:
            ts, seq, _ = engine.records[index]
            version = self._current_version(feature, engine.series, ts, seq) + 1
            self._versions[(feature, engine.series, ts, seq)] = version
            rows.append(
                FeatureRow(
                    feature=feature,
                    series=engine.series,
                    ts=ts,
                    seq=seq,
                    value=value,
                    version=version,
                    reason=reason,
                    ingested_ns=now,
                )
            )
        self._rows_emitted += len(rows)
        self._storage.write_rows(rows)

    def _emit_backfill(
        self, engine: SeriesEngine, spec: FeatureSpec, values: Sequence[float]
    ) -> None:
        now = time.time_ns()
        rows = [
            FeatureRow(
                feature=spec.name,
                series=engine.series,
                ts=ts,
                seq=seq,
                value=value,
                version=1,
                reason="backfill",
                ingested_ns=now,
            )
            for (ts, seq, _), value in zip(engine.records, values, strict=True)
        ]
        for row in rows:
            self._versions[(row.feature, row.series, row.ts, row.seq)] = 1
        self._rows_emitted += len(rows)
        self._storage.write_rows(rows)

    def _replay_raw(self) -> None:
        grouped: dict[str, list[RawRecord]] = {}
        for record in self._storage.read_raw():
            grouped.setdefault(record.series, []).append(record)
        for series, records in grouped.items():
            engine = self._engine(series)
            for record in sorted(records, key=lambda item: (item.ts, item.seq)):
                engine.append(record.ts, record.seq, record.values)

    def _write_catalog(self) -> None:
        if self._root is None:
            return
        payload = {
            "schema_version": _CATALOG_SCHEMA,
            "columns": list(self._columns),
            "late_policy": self._late_policy.value,
            "allowed_lateness": self._allowed_lateness,
            "features": [spec.to_dict() for spec in self._specs.values()],
        }
        temporary = self._root / (_CATALOG_NAME + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self._root / _CATALOG_NAME)


def _read_catalog(root: Path) -> dict[str, Any] | None:
    path = root / _CATALOG_NAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FeatureStoreError(f"invalid catalogue in {path}")
    if int(payload.get("schema_version", 0)) != _CATALOG_SCHEMA:
        raise FeatureStoreError(f"unsupported catalogue schema in {path}")
    return payload
