"""Checkpoints: exact state restore, tail replay and lean mode."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow.dataset as ds

from conftest import latest_values, standard_specs, synthetic_ticks, time_specs
from feature_store import FeatureSpec, FeatureStore, Tick, compute_batch
from feature_store._bits import same_bits
from feature_store.checkpoint import read_latest


def _feed(store: FeatureStore, ticks: list[Tick]) -> None:
    for tick in ticks:
        store.push_values(tick.series, tick.ts, (tick.values["x"], tick.values["y"]), seq=tick.seq)


def _assert_matches_batch(store: FeatureStore, specs: list, ticks: list[Tick], series: str) -> None:
    expected = compute_batch(specs, ticks, columns=("x", "y"))
    for spec in specs:
        actual = latest_values(store, spec.name, series)
        assert len(actual) == len(ticks)
        for (ts, seq), value in actual.items():
            want = expected[(spec.name, series, ts, seq)]
            assert same_bits(value, want), f"{spec.name} at ({ts}, {seq})"


def test_checkpoint_restores_exact_state_and_continues(tmp_path: Path) -> None:
    ticks = synthetic_ticks(0, 300)
    specs = standard_specs((1, 5, 20)) + time_specs((3, 12))
    store = FeatureStore.open(tmp_path, columns=("x", "y"), checkpoint_every=50)
    reference = FeatureStore.memory(columns=("x", "y"))
    for spec in specs:
        store.register(spec)
        reference.register(spec)
    _feed(store, ticks)
    _feed(reference, ticks)
    store.close()

    reopened = FeatureStore.open(tmp_path)
    assert {spec.name for spec in reopened.features()} == {spec.name for spec in specs}
    for spec in specs:
        value = reopened.get(spec.name, "S")
        expected = reference.get(spec.name, "S")
        assert value is not None and expected is not None
        assert same_bits(value, expected)

    extra = synthetic_ticks(1, 80, start=ticks[-1].ts + 1_000_000_000)
    _feed(reopened, extra)
    _feed(reference, extra)
    for spec in specs:
        value = reopened.get(spec.name, "S")
        expected = reference.get(spec.name, "S")
        assert value is not None and expected is not None
        assert same_bits(value, expected)
    reopened.close()


def test_reopen_replays_only_the_tail(tmp_path: Path) -> None:
    store = FeatureStore.open(tmp_path, columns=("x", "y"), checkpoint_every=None)
    store.register(FeatureSpec.mean("m5", 5))
    ticks = synthetic_ticks(2, 200)
    _feed(store, ticks)
    store.checkpoint()
    _feed(store, synthetic_ticks(3, 30, start=ticks[-1].ts + 1_000_000_000))
    store.flush()
    # No close: the last 30 records are only in the raw log.
    summary_before = store.state_summary()["S"]
    assert summary_before["processed"] == 230

    reopened = FeatureStore.open(tmp_path, checkpoint_every=None)
    summary = reopened.state_summary()["S"]
    assert summary["processed"] == 230
    assert summary["retained"] == 30
    assert same_bits(reopened.get("m5", "S"), store.get("m5", "S"))  # type: ignore[arg-type]
    reopened.close()
    store.flush()


def test_lean_mode_matches_batch_after_reopen(tmp_path: Path) -> None:
    specs = standard_specs((1, 3, 10)) + time_specs((2, 8))
    ticks = synthetic_ticks(4, 320)
    store = FeatureStore.open(
        tmp_path,
        columns=("x", "y"),
        retain_records=False,
        checkpoint_every=40,
    )
    for spec in specs:
        store.register(spec)
    _feed(store, ticks)
    assert store.state_summary()["S"]["retained"] == 0
    _assert_matches_batch(store, specs, ticks, "S")

    store.close()
    reopened = FeatureStore.open(tmp_path)
    assert reopened.retain_records is False
    _assert_matches_batch(reopened, specs, ticks, "S")

    extra = synthetic_ticks(5, 60, start=ticks[-1].ts + 1_000_000_000)
    _feed(reopened, extra)
    _assert_matches_batch(reopened, specs, ticks + extra, "S")
    reopened.close()


def test_late_insert_rebuilds_a_trimmed_engine(tmp_path: Path) -> None:
    specs = [FeatureSpec.mean("m3", 3), FeatureSpec.std("s3", 3)]
    ticks = [
        Tick("S", index * 10, {"x": float(index % 11), "y": 0.0}, index) for index in range(200)
    ]
    store = FeatureStore.open(
        tmp_path,
        columns=("x",),
        late_policy="recompute",
        retain_records=False,
        checkpoint_every=50,
    )
    for spec in specs:
        store.register(spec)
    for tick in ticks:
        store.push("S", tick.ts, tick.values["x"], seq=tick.seq)
    late = Tick("S", 55, {"x": 1000.0, "y": 0.0}, 999)
    result = store.push("S", late.ts, late.values["x"], seq=late.seq)
    assert result.status == "late_recomputed"
    assert result.corrections > 0

    expected = compute_batch(specs, [*ticks, late], columns=("x",))
    for spec in specs:
        assert same_bits(store.get(spec.name, "S"), expected[(spec.name, "S", 1990, 199)])  # type: ignore[arg-type]
    assert max(row.version for row in store.history("m3", "S")) >= 2
    assert store.state_summary()["S"]["retained"] == 0
    store.close()


def test_register_backfills_a_trimmed_engine_from_the_raw_log(tmp_path: Path) -> None:
    ticks = synthetic_ticks(6, 120)
    store = FeatureStore.open(
        tmp_path, columns=("x", "y"), retain_records=False, checkpoint_every=25
    )
    store.register(FeatureSpec.mean("m5", 5))
    _feed(store, ticks)
    new_spec = FeatureSpec.std("s7", 7)
    store.register(new_spec)
    rows = store.history("s7", "S")
    assert len(rows) == len(ticks)
    assert {row.reason for row in rows} == {"backfill"}
    expected = compute_batch([new_spec], ticks, columns=("x", "y"))
    for row in rows:
        assert same_bits(row.value, expected[("s7", "S", row.ts, row.seq)])
    store.close()


def test_mismatched_feature_catalogue_falls_back_to_full_replay(tmp_path: Path) -> None:
    ticks = synthetic_ticks(7, 100)
    store = FeatureStore.open(tmp_path, columns=("x", "y"), checkpoint_every=20)
    store.register(FeatureSpec.mean("m5", 5))
    _feed(store, ticks)
    store.close()

    checkpoint = read_latest(tmp_path)
    assert checkpoint is not None
    manifest_path = checkpoint.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["spec_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reopened = FeatureStore.open(tmp_path)
    summary = reopened.state_summary()["S"]
    assert summary["processed"] == len(ticks)
    assert summary["retained"] == len(ticks)
    assert reopened.get("m5", "S") is not None
    reopened.close()


def test_checkpoint_prunes_old_directories(tmp_path: Path) -> None:
    store = FeatureStore.open(tmp_path, columns=("x",), checkpoint_every=None)
    store.register(FeatureSpec.mean("m2", 2))
    for index in range(10):
        store.push("S", index, float(index))
        store.checkpoint()
    checkpoints = sorted((tmp_path / "state").glob("checkpoint-*"))
    assert len(checkpoints) == 2
    latest = (tmp_path / "state" / "LATEST").read_text(encoding="utf-8").strip()
    assert (tmp_path / "state" / f"checkpoint-{latest}").exists()
    store.close()


def test_checkpoint_manifest_lists_the_consumed_raw_files(tmp_path: Path) -> None:
    store = FeatureStore.open(tmp_path, columns=("x",), checkpoint_every=None)
    store.register(FeatureSpec.mean("m2", 2))
    for index in range(50):
        store.push("S", index, float(index))
    store.flush()
    store.checkpoint()
    checkpoint = read_latest(tmp_path)
    assert checkpoint is not None
    manifest = json.loads((checkpoint.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_files"]
    assert all(path.endswith(".parquet") for path in manifest["raw_files"])
    for path in manifest["raw_files"]:
        assert (tmp_path / path).exists()
    assert checkpoint.series["S"]["processed"] == 50
    store.close()


def test_checkpoint_does_not_touch_the_feature_table_schema(tmp_path: Path) -> None:
    store = FeatureStore.open(tmp_path, columns=("x", "y"), checkpoint_every=None)
    store.register(FeatureSpec.mean("m5", 5))
    for index in range(20):
        store.push_values("S", index, (float(index), float(index) * 2))
    store.flush()
    partition = tmp_path / "features" / "feature=m5" / "series=S"
    before = ds.dataset(partition, format="parquet").to_table().schema
    store.checkpoint()
    after = ds.dataset(partition, format="parquet").to_table().schema
    assert before.equals(after)
    assert before.names == [
        "feature",
        "series",
        "ts",
        "seq",
        "value",
        "version",
        "reason",
        "ingested_ns",
    ]
    store.close()


def test_checkpoint_restores_multiple_series(tmp_path: Path) -> None:
    specs = [FeatureSpec.mean("m3", 3), FeatureSpec.corr("c3", 3)]
    ticks = synthetic_ticks(12, 120, series="A") + synthetic_ticks(13, 140, series="B")
    store = FeatureStore.open(tmp_path, columns=("x", "y"), checkpoint_every=40)
    reference = FeatureStore.memory(columns=("x", "y"))
    for spec in specs:
        store.register(spec)
        reference.register(spec)
    _feed(store, ticks)
    _feed(reference, ticks)
    store.close()

    reopened = FeatureStore.open(tmp_path)
    for series in ("A", "B"):
        for spec in specs:
            value = reopened.get(spec.name, series)
            expected = reference.get(spec.name, series)
            assert value is not None and expected is not None
            assert same_bits(value, expected)
    reopened.close()


def test_streaming_replay_falls_back_on_an_unsorted_raw_log(tmp_path: Path) -> None:
    specs = [FeatureSpec.mean("m3", 3), FeatureSpec.std("s3", 3)]
    store = FeatureStore.open(
        tmp_path, columns=("x",), late_policy="recompute", checkpoint_every=None
    )
    for spec in specs:
        store.register(spec)
    for index in range(50):
        store.push("S", index * 10, float(index % 7))
    store.push("S", 55, 100.0, seq=999)
    store.checkpoint()
    store.close()
    shutil.rmtree(tmp_path / "state")

    reopened = FeatureStore.open(tmp_path, checkpoint_every=None, retain_records=False)
    ticks = [Tick("S", index * 10, {"x": float(index % 7)}, index) for index in range(50)]
    ticks.append(Tick("S", 55, {"x": 100.0}, 999))
    expected = compute_batch(specs, ticks, columns=("x",))
    assert reopened.state_summary()["S"]["processed"] == 51
    for spec in specs:
        assert same_bits(reopened.get(spec.name, "S"), expected[(spec.name, "S", 490, 49)])  # type: ignore[arg-type]
    reopened.close()


def test_time_window_state_survives_a_checkpoint(tmp_path: Path) -> None:
    specs = time_specs((5, 40))
    ticks = synthetic_ticks(8, 150)
    store = FeatureStore.open(
        tmp_path, columns=("x", "y"), retain_records=False, checkpoint_every=30
    )
    for spec in specs:
        store.register(spec)
    _feed(store, ticks)
    store.close()

    reopened = FeatureStore.open(tmp_path)
    extra = synthetic_ticks(9, 40, start=ticks[-1].ts + 1_000_000_000)
    _feed(reopened, extra)
    _assert_matches_batch(reopened, specs, ticks + extra, "S")
    reopened.close()
