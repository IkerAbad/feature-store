"""Public feature store API.

The store owns three things: the catalogue of registered features, the
in-memory incremental state of each series, and a storage backend. It is a
single-writer object: one process, one thread at a time.

Reopening is O(tail) when a checkpoint exists: the state machines are restored
from ``state/LATEST`` and only the raw files written after the checkpoint are
replayed. Without a checkpoint the store replays the raw log from the first
record, which is O(N). Late records that fall before the retained history force
a rebuild from the raw log; that is the O(N) path either way.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

from ._bits import same_bits
from .checkpoint import Checkpoint, prune_checkpoints, read_latest, write_checkpoint
from .engine import SeriesEngine, make_state
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
_UNSET = object()
NaN = float("nan")
DEFAULT_CHECKPOINT_EVERY = 1_000_000


class _UnsortedRawLog(Exception):
    """Internal signal: the raw files are not in event-time order."""


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


@dataclass(frozen=True, slots=True)
class PrunePolicy:
    """How many versions of each key the audit table keeps."""

    keep_versions: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.keep_versions, int) or isinstance(self.keep_versions, bool):
            raise ValueError("keep_versions must be an integer")
        if self.keep_versions < 1:
            raise ValueError("keep_versions must be >= 1")


@dataclass(frozen=True, slots=True)
class PartitionPrune:
    """Rows removed from one feature/series partition."""

    feature: str
    series: str
    before: int
    after: int
    removed: int


@dataclass(frozen=True, slots=True)
class PruneReport:
    """Outcome of :meth:`FeatureStore.prune`."""

    policy: PrunePolicy
    dry_run: bool
    partitions: tuple[PartitionPrune, ...]
    removed_rows: int
    manifest: Path | None


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
        retain_records: bool = True,
        checkpoint_every: int | None = None,
        checkpoint_keep: int = 2,
        audit_keep_versions: int | None = None,
    ) -> None:
        if not columns:
            raise FeatureStoreError("at least one input column is required")
        if any(not isinstance(column, str) or not column for column in columns):
            raise FeatureStoreError("input column names must be non-empty strings")
        if len(set(columns)) != len(columns):
            raise FeatureStoreError("input column names must be unique")
        if allowed_lateness < 0:
            raise FeatureStoreError("allowed_lateness must be >= 0")
        if checkpoint_every is not None and checkpoint_every < 1:
            raise FeatureStoreError("checkpoint_every must be >= 1 or None")
        if audit_keep_versions is not None and audit_keep_versions < 1:
            raise FeatureStoreError("audit_keep_versions must be >= 1 or None")
        self._storage = storage
        self._columns = tuple(columns)
        self._late_policy = LatePolicy(late_policy)
        self._allowed_lateness = int(allowed_lateness)
        self._root = Path(root) if root is not None else None
        self._retain_records = bool(retain_records)
        self._checkpoint_every = checkpoint_every
        self._checkpoint_keep = max(1, int(checkpoint_keep))
        self._audit_keep_versions = audit_keep_versions
        self._specs: dict[str, FeatureSpec] = {}
        self._series: dict[str, SeriesEngine] = {}
        self._timelines: dict[tuple[str, str], tuple[list[FeatureRow], list[int]]] = {}
        self._versions: dict[tuple[str, str, int, int], int] = {}
        self._since_checkpoint: dict[str, int] = {}
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
        retain_records: bool | None = None,
        checkpoint_every: Any = _UNSET,
        checkpoint_keep: int = 2,
        audit_keep_versions: Any = _UNSET,
    ) -> FeatureStore:
        """Open (or create) a persistent store rooted at ``root``.

        On an existing store the catalogue is read first. If a checkpoint
        exists, the state machines are restored and only the raw files written
        after it are replayed; otherwise every accepted record is replayed.
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
            effective_retain = True if retain_records is None else retain_records
            effective_checkpoint = (
                DEFAULT_CHECKPOINT_EVERY if checkpoint_every is _UNSET else checkpoint_every
            )
            effective_audit = None if audit_keep_versions is _UNSET else audit_keep_versions
            storage = ParquetStorage(path, effective_columns, flush_every=flush_every)
            store = cls(
                storage,
                columns=effective_columns,
                late_policy=effective_policy,
                allowed_lateness=effective_lateness,
                root=path,
                retain_records=effective_retain,
                checkpoint_every=effective_checkpoint,
                checkpoint_keep=checkpoint_keep,
                audit_keep_versions=effective_audit,
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
        effective_retain = (
            bool(catalog.get("retain_records", True)) if retain_records is None else retain_records
        )
        if checkpoint_every is _UNSET:
            stored_checkpoint = catalog.get("checkpoint_every", DEFAULT_CHECKPOINT_EVERY)
            effective_checkpoint = None if stored_checkpoint is None else int(stored_checkpoint)
        else:
            effective_checkpoint = checkpoint_every
        if audit_keep_versions is _UNSET:
            stored_audit = catalog.get("audit_keep_versions")
            effective_audit = None if stored_audit is None else int(stored_audit)
        else:
            effective_audit = audit_keep_versions
        storage = ParquetStorage(path, stored_columns, flush_every=flush_every)
        store = cls(
            storage,
            columns=stored_columns,
            late_policy=effective_policy,
            allowed_lateness=effective_lateness,
            root=path,
            retain_records=effective_retain,
            checkpoint_every=effective_checkpoint,
            checkpoint_keep=checkpoint_keep,
            audit_keep_versions=effective_audit,
        )
        store._specs = {
            spec.name: spec
            for spec in (FeatureSpec.from_dict(item) for item in catalog.get("features", ()))
        }
        store._load_state()
        store._write_catalog()
        return store

    @classmethod
    def memory(
        cls,
        *,
        columns: Sequence[str] = ("x", "y"),
        late_policy: LatePolicy | str = LatePolicy.DROP,
        allowed_lateness: int = 0,
        audit_keep_versions: int | None = None,
    ) -> FeatureStore:
        """Ephemeral store backed by Python lists."""
        return cls(
            MemoryStorage(),
            columns=columns,
            late_policy=late_policy,
            allowed_lateness=allowed_lateness,
            audit_keep_versions=audit_keep_versions,
        )

    @classmethod
    def null(
        cls,
        *,
        columns: Sequence[str] = ("x", "y"),
        late_policy: LatePolicy | str = LatePolicy.DROP,
        allowed_lateness: int = 0,
        retain_records: bool = True,
    ) -> FeatureStore:
        """Store that keeps state but writes no rows. Used by the benchmark."""
        return cls(
            NullStorage(),
            columns=columns,
            late_policy=late_policy,
            allowed_lateness=allowed_lateness,
            retain_records=retain_records,
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

    @property
    def retain_records(self) -> bool:
        return self._retain_records

    @property
    def checkpoint_every(self) -> int | None:
        return self._checkpoint_every

    @property
    def audit_keep_versions(self) -> int | None:
        return self._audit_keep_versions

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
        for series, engine in self._series.items():
            self._backfill_feature(series, engine, spec)
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
        elif engine.last_key is None or key > engine.last_key:
            return self._append_record(
                series, engine, ts, seq, values, "append", "applied", watermark
            )
        else:
            reason = "out_of_order"
            status = "out_of_order"

        if engine.has_full_history:
            position = bisect_right(engine.keys, key)
            if position and engine.keys[position - 1] == key:
                raise DuplicateRecordError(
                    f"record ({ts}, {seq}) already exists for series {series!r}"
                )
            changes = engine.insert(position, ts, seq, values)
            self._invalidate()
            if self._storage.wants_raw:
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
            if self._storage.wants_rows:
                outputs = [slot.values[position] for slot in engine.slots]
                self._emit_new_row(engine, ts, seq, outputs, reason)
                if changes:
                    self._emit_corrections(engine, changes, reason)
            self._corrections += len(changes)
            self._after_ingest(series)
            return IngestResult(status, series, ts, seq, len(changes), watermark)

        return self._rebuild_with_insert(series, ts, seq, values, reason, status, watermark)

    def _append_record(
        self,
        series: str,
        engine: SeriesEngine,
        ts: int,
        seq: int,
        values: tuple[float, ...],
        reason: str,
        status: str,
        watermark: int | None,
    ) -> IngestResult:
        outputs = engine.append(ts, seq, values)
        self._invalidate()
        if self._storage.wants_raw:
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
        if self._storage.wants_rows:
            self._emit_new_row(engine, ts, seq, outputs, reason)
        self._after_ingest(series)
        return IngestResult(status, series, ts, seq, 0, watermark)

    def _after_ingest(self, series: str) -> None:
        if self._checkpoint_every is None:
            return
        count = self._since_checkpoint.get(series, 0) + 1
        self._since_checkpoint[series] = count
        if count >= self._checkpoint_every:
            self.checkpoint()

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

    def state_summary(self) -> dict[str, dict[str, Any]]:
        """Diagnostics: how much of each series is in memory and where it ends.

        ``processed`` is every accepted record, ``retained`` the ones still kept
        for in-memory corrections. After a checkpoint taken with
        ``retain_records=False`` the retained count is zero and ``processed``
        keeps growing.
        """
        return {
            series: {
                "processed": engine.processed,
                "retained": len(engine.records),
                "max_ts": engine.max_ts,
                "last_key": engine.last_key,
            }
            for series, engine in self._series.items()
        }

    # ------------------------------------------------------------------
    # Audit pruning
    # ------------------------------------------------------------------
    def prune(self, keep_versions: int | None = None, *, dry_run: bool = False) -> PruneReport:
        """Keep only the last ``keep_versions`` versions of every key.

        The Parquet schema does not change: the partition is rewritten with the
        surviving rows and the old files are deleted. A pruning manifest is
        written under ``audit/`` so the operation itself is traceable. With
        ``dry_run`` the report is produced and nothing is written.

        Pruning drops history, not state: queries keep returning the latest
        value and checkpoints are unaffected.
        """
        keep = self._audit_keep_versions if keep_versions is None else int(keep_versions)
        if keep is None:
            raise FeatureStoreError(
                "no pruning policy configured; pass keep_versions or set audit_keep_versions"
            )
        policy = PrunePolicy(keep)
        self._storage.flush()
        partitions: list[PartitionPrune] = []
        for feature, series in self._storage.partitions():
            rows = self._storage.rows(feature, series)
            grouped: dict[tuple[int, int], list[FeatureRow]] = {}
            for row in rows:
                grouped.setdefault((row.ts, row.seq), []).append(row)
            kept: list[FeatureRow] = []
            for versions in grouped.values():
                versions.sort(key=lambda row: row.version)
                kept.extend(versions[-policy.keep_versions :])
            if len(kept) == len(rows):
                continue
            kept.sort(key=lambda row: (row.ts, row.seq, row.version))
            partitions.append(
                PartitionPrune(
                    feature=feature,
                    series=series,
                    before=len(rows),
                    after=len(kept),
                    removed=len(rows) - len(kept),
                )
            )
            if not dry_run:
                self._storage.rewrite_partition(feature, series, kept)
        report = PruneReport(
            policy=policy,
            dry_run=dry_run,
            partitions=tuple(partitions),
            removed_rows=sum(item.removed for item in partitions),
            manifest=None,
        )
        if not dry_run:
            self._invalidate()
            self._versions.clear()
            if self._root is not None and partitions:
                manifest = self._write_prune_manifest(report)
                return PruneReport(
                    policy=report.policy,
                    dry_run=False,
                    partitions=report.partitions,
                    removed_rows=report.removed_rows,
                    manifest=manifest,
                )
        return report

    def _write_prune_manifest(self, report: PruneReport) -> Path:
        assert self._root is not None
        directory = self._root / "audit"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"pruned-{time.time_ns():019d}.json"
        payload = {
            "created_ns": time.time_ns(),
            "policy": asdict(report.policy),
            "removed_rows": report.removed_rows,
            "partitions": [asdict(item) for item in report.partitions],
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return path

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def checkpoint(self) -> Path | None:
        """Persist the exact state of every series and make it the restart point.

        The storage is flushed first, so the manifest can list every raw file
        whose records are already reflected in the state. Engines that do not
        retain records are trimmed after the write; retained engines keep their
        history untouched.
        """
        if self._root is None:
            return None
        self._storage.flush()
        rows: list[dict[str, Any]] = []
        series_meta: dict[str, dict[str, Any]] = {}
        for series, engine in self._series.items():
            for name, snapshot in engine.snapshot():
                rows.append({"series": series, "feature": name, **snapshot})
            series_meta[series] = {
                "last_key": list(engine.last_key) if engine.last_key is not None else None,
                "max_ts": engine.max_ts,
                "next_seq": engine.next_seq,
                "processed": engine.processed,
            }
        manifest = {
            "columns": list(self._columns),
            "features": [spec.to_dict() for spec in self._specs.values()],
            "spec_hash": self._spec_hash(),
            "raw_files": self._storage.raw_files(),
            "series": series_meta,
            "created_ns": time.time_ns(),
        }
        checkpoint = write_checkpoint(self._root, manifest, rows)
        prune_checkpoints(self._root, self._checkpoint_keep)
        if not self._retain_records:
            for engine in self._series.values():
                engine.trim()
        self._since_checkpoint.clear()
        return checkpoint.path

    def _load_state(self) -> None:
        checkpoint = read_latest(self._root) if self._root is not None else None
        if checkpoint is not None and self._restore_checkpoint(checkpoint):
            self._append_raw(self._storage.read_raw(exclude_files=checkpoint.raw_files))
        elif self._retain_records:
            self._append_raw(self._storage.read_raw())
        else:
            self._stream_raw()

    def _stream_raw(self) -> None:
        """Replay the raw log file by file, with O(1) memory.

        Raw files are written in arrival order. When the log is sorted by
        ``(ts, seq)`` the replay is streaming. If an out-of-order batch shows
        up (a late record was accepted), the state is discarded and rebuilt
        from a fully sorted read, which needs O(N) memory.
        """
        try:
            for series, batch in self._storage.iter_raw():
                batch.sort(key=lambda record: (record.ts, record.seq))
                engine = self._engine(series)
                last = engine.last_key
                for record in batch:
                    key = (record.ts, record.seq)
                    if last is not None and key <= last:
                        raise _UnsortedRawLog
                    engine.append(record.ts, record.seq, record.values)
                    last = key
        except _UnsortedRawLog:
            self._series.clear()
            self._append_raw(self._storage.read_raw())

    def _restore_checkpoint(self, checkpoint: Checkpoint) -> bool:
        if self._root is None:
            return False
        manifest = checkpoint.manifest
        if tuple(manifest.get("columns", ())) != self._columns:
            return False
        if manifest.get("spec_hash") != self._spec_hash():
            return False
        expected = {(series, name) for series in checkpoint.series for name in self._specs}
        present = {(row["series"], row["feature"]) for row in checkpoint.rows}
        if expected - present:
            return False
        states: dict[str, dict[str, Any]] = {}
        for row in checkpoint.rows:
            states.setdefault(row["series"], {})[row["feature"]] = row
        for series, meta in checkpoint.series.items():
            engine = SeriesEngine(series, self._columns, retain_records=self._retain_records)
            for name, spec in self._specs.items():
                state = make_state(spec, self._columns)
                state.load(states[series][name])
                engine.attach(spec, state)
            last_key = meta.get("last_key")
            engine.last_key = (int(last_key[0]), int(last_key[1])) if last_key else None
            engine.max_ts = None if meta.get("max_ts") is None else int(meta["max_ts"])
            engine.next_seq = int(meta.get("next_seq", 0))
            engine.processed = int(meta.get("processed", 0))
            self._series[series] = engine
        return True

    def _append_raw(self, records: Sequence[RawRecord]) -> None:
        grouped: dict[str, list[RawRecord]] = {}
        for record in records:
            grouped.setdefault(record.series, []).append(record)
        for series, series_records in grouped.items():
            engine = self._engine(series)
            for record in sorted(series_records, key=lambda item: (item.ts, item.seq)):
                engine.append(record.ts, record.seq, record.values)

    def _spec_hash(self) -> str:
        payload = json.dumps(
            [spec.to_dict() for spec in self._specs.values()], sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    # ------------------------------------------------------------------
    # Late rebuild and backfill
    # ------------------------------------------------------------------
    def _rebuild_with_insert(
        self,
        series: str,
        ts: int,
        seq: int,
        values: tuple[float, ...],
        reason: str,
        status: str,
        watermark: int | None,
    ) -> IngestResult:
        records = self._storage.read_raw(series=series)
        keys = {(record.ts, record.seq) for record in records}
        if (ts, seq) in keys:
            raise DuplicateRecordError(f"record ({ts}, {seq}) already exists for series {series!r}")
        items = [(record.ts, record.seq, record.values) for record in records]
        items.append((ts, seq, values))
        items.sort(key=lambda item: (item[0], item[1]))
        position = next(
            index for index, item in enumerate(items) if item[0] == ts and item[1] == seq
        )

        rebuilt = SeriesEngine(series, self._columns, retain_records=True)
        for spec in self._specs.values():
            rebuilt.add_feature(spec)
        for item_ts, item_seq, item_values in items:
            rebuilt.append(item_ts, item_seq, item_values)

        self._storage.write_raw(
            [RawRecord(series=series, ts=ts, seq=seq, values=values, ingested_ns=time.time_ns())]
        )
        outputs = [slot.values[position] for slot in rebuilt.slots]
        if self._storage.wants_rows:
            self._emit_new_row(rebuilt, ts, seq, outputs, reason)
            changes = self._emit_rebuild_corrections(rebuilt, items, position, reason)
        else:
            changes = 0
        self._corrections += changes
        if not self._retain_records:
            rebuilt.trim()
        self._series[series] = rebuilt
        self._versions.clear()
        self._invalidate()
        self._after_ingest(series)
        return IngestResult(status, series, ts, seq, changes, watermark)

    def _emit_rebuild_corrections(
        self,
        engine: SeriesEngine,
        items: Sequence[tuple[int, int, tuple[float, ...]]],
        position: int,
        reason: str,
    ) -> int:
        now = time.time_ns()
        emitted = 0
        for slot in engine.slots:
            rows = self._storage.rows(slot.spec.name, engine.series)
            latest: dict[tuple[int, int], tuple[float, int]] = {}
            for row in rows:
                key = (row.ts, row.seq)
                previous = latest.get(key)
                if previous is None or row.version > previous[1]:
                    latest[key] = (row.value, row.version)
            for index in range(position + 1, len(items)):
                item_ts, item_seq, _ = items[index]
                new_value = slot.values[index]
                previous = latest.get((item_ts, item_seq))
                if previous is not None and same_bits(new_value, previous[0]):
                    continue
                version = 1 if previous is None else previous[1] + 1
                self._storage.write_rows(
                    [
                        FeatureRow(
                            feature=slot.spec.name,
                            series=engine.series,
                            ts=item_ts,
                            seq=item_seq,
                            value=new_value,
                            version=version,
                            reason=reason,
                            ingested_ns=now,
                        )
                    ]
                )
                self._rows_emitted += 1
                emitted += 1
        return emitted

    def _backfill_feature(self, series: str, engine: SeriesEngine, spec: FeatureSpec) -> None:
        if engine.has_full_history:
            values = engine.add_feature(spec)
            order = [(ts, seq) for ts, seq, _ in engine.records]
        else:
            records = sorted(
                self._storage.read_raw(series=series), key=lambda item: (item.ts, item.seq)
            )
            state = make_state(spec, self._columns)
            values = [state.update(record.ts, record.values) for record in records]
            engine.attach(spec, state)
            order = [(record.ts, record.seq) for record in records]
        if not self._storage.wants_rows or not order:
            return
        now = time.time_ns()
        rows = [
            FeatureRow(
                feature=spec.name,
                series=series,
                ts=ts,
                seq=seq,
                value=value,
                version=1,
                reason="backfill",
                ingested_ns=now,
            )
            for (ts, seq), value in zip(order, values, strict=True)
        ]
        for row in rows:
            self._versions[(row.feature, row.series, row.ts, row.seq)] = 1
        self._rows_emitted += len(rows)
        self._storage.write_rows(rows)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def flush(self) -> None:
        self._storage.flush()

    def close(self) -> None:
        if self._root is not None and self._checkpoint_every is not None:
            self.checkpoint()
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
            engine = SeriesEngine(series, self._columns, retain_records=self._retain_records)
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

    def _emit_new_row(
        self,
        engine: SeriesEngine,
        ts: int,
        seq: int,
        outputs: Sequence[float],
        reason: str,
    ) -> None:
        now = time.time_ns()
        rows = [
            FeatureRow(
                feature=slot.spec.name,
                series=engine.series,
                ts=ts,
                seq=seq,
                value=value,
                version=1,
                reason=reason,
                ingested_ns=now,
            )
            for slot, value in zip(engine.slots, outputs, strict=True)
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

    def _write_catalog(self) -> None:
        if self._root is None:
            return
        payload = {
            "schema_version": _CATALOG_SCHEMA,
            "columns": list(self._columns),
            "late_policy": self._late_policy.value,
            "allowed_lateness": self._allowed_lateness,
            "retain_records": self._retain_records,
            "checkpoint_every": self._checkpoint_every,
            "audit_keep_versions": self._audit_keep_versions,
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
