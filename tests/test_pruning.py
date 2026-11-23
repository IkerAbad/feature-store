"""Audit pruning: keep the last N versions and nothing else."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from feature_store import (
    FeatureSpec,
    FeatureStore,
    FeatureStoreError,
    PrunePolicy,
    Tick,
    compute_batch,
)
from feature_store._bits import same_bits


def _store_with_versions() -> FeatureStore:
    store = FeatureStore.memory(columns=("x",), late_policy="recompute")
    store.register(FeatureSpec.mean("m2", 2))
    for index in range(5):
        store.push("S", index * 10, float(index))
    for ts, value, seq in ((5, 100.0, 10), (6, 200.0, 11), (7, 300.0, 12)):
        store.push("S", ts, value, seq=seq)
    return store


def _versions(store: FeatureStore, feature: str, ts: int, seq: int) -> list[int]:
    return [row.version for row in store.history(feature, "S") if row.ts == ts and row.seq == seq]


def test_prune_keeps_the_last_n_versions() -> None:
    store = _store_with_versions()
    assert _versions(store, "m2", 10, 1) == [1, 2, 3, 4]
    latest = store.get("m2", "S")
    report = store.prune(keep_versions=2)
    assert not report.dry_run
    assert report.removed_rows > 0
    assert _versions(store, "m2", 10, 1) == [3, 4]
    assert len(store.history("m2", "S")) == 9
    assert same_bits(store.get("m2", "S"), latest)  # type: ignore[arg-type]


def test_prune_keeps_one_version_when_asked() -> None:
    store = _store_with_versions()
    store.prune(keep_versions=1)
    rows = store.history("m2", "S")
    assert len(rows) == len({(row.ts, row.seq) for row in rows})
    assert _versions(store, "m2", 10, 1) == [4]
    assert store.get("m2", "S") == 3.5


def test_prune_dry_run_changes_nothing() -> None:
    store = _store_with_versions()
    before = store.history("m2", "S")
    report = store.prune(keep_versions=1, dry_run=True)
    assert report.dry_run
    assert report.removed_rows > 0
    assert store.history("m2", "S") == before


def test_prune_is_idempotent() -> None:
    store = _store_with_versions()
    store.prune(keep_versions=2)
    first = store.history("m2", "S")
    second_report = store.prune(keep_versions=2)
    assert second_report.removed_rows == 0
    assert store.history("m2", "S") == first


def test_prune_uses_the_configured_policy() -> None:
    store = FeatureStore.memory(columns=("x",), late_policy="recompute", audit_keep_versions=1)
    store.register(FeatureSpec.mean("m2", 2))
    for index in range(5):
        store.push("S", index * 10, float(index))
    store.push("S", 5, 100.0, seq=10)
    assert store.audit_keep_versions == 1
    report = store.prune()
    assert report.policy == PrunePolicy(1)
    assert _versions(store, "m2", 10, 1) == [2]


def test_prune_without_a_policy_is_an_error() -> None:
    store = FeatureStore.memory(columns=("x",))
    with pytest.raises(FeatureStoreError):
        store.prune()
    with pytest.raises(ValueError):
        PrunePolicy(0)


def test_prune_writes_a_manifest(tmp_path: Path) -> None:
    store = FeatureStore.open(tmp_path, columns=("x",), late_policy="recompute")
    store.register(FeatureSpec.mean("m2", 2))
    for index in range(4):
        store.push("S", index * 10, float(index))
    store.push("S", 5, 50.0, seq=10)
    store.push("S", 6, 60.0, seq=11)
    report = store.prune(keep_versions=1)
    assert report.manifest is not None and report.manifest.exists()
    payload = json.loads(report.manifest.read_text(encoding="utf-8"))
    assert payload["policy"]["keep_versions"] == 1
    assert payload["removed_rows"] == report.removed_rows
    assert payload["partitions"][0]["feature"] == "m2"
    store.close()


def test_prune_preserves_the_latest_values_after_reopen(tmp_path: Path) -> None:
    ticks = [Tick("S", index * 10, {"x": float(index)}, index) for index in range(10)]
    reference = compute_batch(
        [FeatureSpec.mean("m2", 2), FeatureSpec.std("s2", 2)], ticks, columns=("x",)
    )
    store = FeatureStore.open(tmp_path, columns=("x",), late_policy="recompute")
    store.register(FeatureSpec.mean("m2", 2))
    store.register(FeatureSpec.std("s2", 2))
    for tick in ticks:
        store.push("S", tick.ts, tick.values["x"], seq=tick.seq)
    for ts, value, seq in ((5, 100.0, 10), (6, 200.0, 11), (7, 300.0, 12)):
        store.push("S", ts, value, seq=seq)
    store.prune(keep_versions=2)
    store.close()

    reopened = FeatureStore.open(tmp_path)
    latest = max((tick.ts, tick.seq) for tick in ticks)
    for feature in ("m2", "s2"):
        got = reopened.get(feature, "S")
        assert got is not None
        assert same_bits(got, reference[(feature, "S", *latest)])
    versions: dict[tuple[int, int], list[int]] = {}
    for row in reopened.history("m2", "S"):
        versions.setdefault((row.ts, row.seq), []).append(row.version)
    assert all(len(items) <= 2 for items in versions.values())
    reopened.close()


def test_prune_does_not_break_checkpoints(tmp_path: Path) -> None:
    store = FeatureStore.open(tmp_path, columns=("x",), late_policy="recompute", checkpoint_every=5)
    store.register(FeatureSpec.mean("m2", 2))
    for index in range(10):
        store.push("S", index, float(index))
    store.push("S", 3, 99.0, seq=100)
    before = store.get("m2", "S")
    store.prune(keep_versions=1)
    store.close()

    reopened = FeatureStore.open(tmp_path)
    assert same_bits(reopened.get("m2", "S"), before)  # type: ignore[arg-type]
    reopened.push("S", 10, 5.0)
    assert reopened.get("m2", "S") == 7.0
    reopened.close()


def test_prune_memory_storage_rewrites_in_place() -> None:
    store = _store_with_versions()
    report = store.prune(keep_versions=2)
    assert report.partitions
    assert report.partitions[0].before > report.partitions[0].after
    assert report.manifest is None
