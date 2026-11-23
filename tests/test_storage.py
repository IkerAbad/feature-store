"""Parquet layout, row versions, persistence and the catalogue."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
import pytest

from conftest import latest_values, standard_specs, synthetic_ticks
from feature_store import FeatureSpec, FeatureStore
from feature_store._bits import same_bits
from feature_store.errors import FeatureStoreError


def _feed(store: FeatureStore, ticks: list) -> None:
    for tick in ticks:
        store.push_values(tick.series, tick.ts, (tick.values["x"], tick.values["y"]), seq=tick.seq)


def _parquet_rows(directory: Path) -> list[dict[str, Any]]:
    dataset = ds.dataset(directory, format="parquet")
    rows: list[dict[str, Any]] = dataset.to_table().to_pylist()
    return rows


def test_partition_layout_and_row_versions(tmp_path: Path) -> None:
    ticks = synthetic_ticks(0, 20)
    store = FeatureStore.open(tmp_path, columns=("x", "y"), late_policy="recompute")
    store.register(FeatureSpec.mean("mean_5", 5))
    _feed(store, ticks)
    store.flush()

    partition = tmp_path / "features" / "feature=mean_5" / "series=S"
    assert partition.is_dir()
    files = sorted(partition.glob("*.parquet"))
    assert files
    rows = _parquet_rows(partition)
    assert len(rows) == 20
    assert {row["version"] for row in rows} == {1}
    assert {row["reason"] for row in rows} == {"append"}
    assert rows[0]["feature"] == "mean_5"
    assert rows[0]["series"] == "S"

    late = ticks[8]
    store.push_values(late.series, late.ts, (999.0, 999.0), seq=1000)
    store.flush()
    rows = _parquet_rows(partition)
    assert len(rows) > 20
    versions = {(row["ts"], row["seq"]): row["version"] for row in rows}
    assert versions[(late.ts, late.seq)] == 1
    corrected = [row for row in rows if row["version"] > 1]
    assert corrected
    assert all(row["reason"] == "late_recompute" for row in corrected)
    assert (
        store.get("mean_5", "S")
        == latest_values(store, "mean_5", "S")[(ticks[-1].ts, ticks[-1].seq)]
    )
    store.close()


def test_late_drops_are_persisted(tmp_path: Path) -> None:
    store = FeatureStore.open(tmp_path, columns=("x",), late_policy="drop")
    store.register(FeatureSpec.mean("m2", 2))
    for index in range(10):
        store.push("A", index * 10, float(index))
    store.push("A", 15, 100.0)
    store.flush()
    drops_dir = tmp_path / "dropped" / "series=A"
    rows = _parquet_rows(drops_dir)
    assert len(rows) == 1
    assert rows[0]["ts"] == 15
    assert rows[0]["policy"] == "drop"
    store.close()
    reopened = FeatureStore.open(tmp_path)
    assert len(reopened.late_drops()) == 1
    reopened.close()


def test_reopen_restores_catalogue_state_and_history(tmp_path: Path) -> None:
    ticks = synthetic_ticks(1, 40)
    specs = standard_specs((1, 3, 8))
    store = FeatureStore.open(tmp_path, columns=("x", "y"), late_policy="recompute")
    for spec in specs:
        store.register(spec)
    _feed(store, ticks)
    late = ticks[12]
    store.push_values(late.series, late.ts, (77.0, 33.0), seq=500)
    store.flush()
    before = {spec.name: store.get(spec.name, "S") for spec in specs}
    store.close()

    reopened = FeatureStore.open(tmp_path)
    assert {spec.name for spec in reopened.features()} == {spec.name for spec in specs}
    assert reopened.columns == ("x", "y")
    after = {spec.name: reopened.get(spec.name, "S") for spec in specs}
    for name in before:
        left, right = before[name], after[name]
        assert left is not None and right is not None
        assert same_bits(left, right)
    for spec in specs:
        left_rows = reopened.history(spec.name, "S")
        right_rows = store.history(spec.name, "S")
        assert len(left_rows) == len(right_rows)
        for stored, reloaded in zip(left_rows, right_rows, strict=True):
            assert same_bits(stored.value, reloaded.value)
            assert (stored.ts, stored.seq, stored.version, stored.reason) == (
                reloaded.ts,
                reloaded.seq,
                reloaded.version,
                reloaded.reason,
            )

    # The rebuilt state keeps ingesting and correcting with the stored versions.
    reopened.push_values("S", late.ts, (1.0, 2.0), seq=501)
    versions = [row.version for row in reopened.history("mean_3", "S")]
    assert max(versions) >= 3
    reopened.close()


def test_catalogue_is_json_and_lists_the_features(tmp_path: Path) -> None:
    store = FeatureStore.open(tmp_path, columns=("x",), late_policy="recompute", allowed_lateness=7)
    store.register(FeatureSpec.mean("mean_5", 5, description="five ticks"))
    store.close()
    payload = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["columns"] == ["x"]
    assert payload["late_policy"] == "recompute"
    assert payload["allowed_lateness"] == 7
    assert payload["features"][0]["name"] == "mean_5"
    assert payload["features"][0]["description"] == "five ticks"


def test_columns_mismatch_on_reopen_is_an_error(tmp_path: Path) -> None:
    store = FeatureStore.open(tmp_path, columns=("x",))
    store.close()
    with pytest.raises(FeatureStoreError):
        FeatureStore.open(tmp_path, columns=("x", "y"))


def test_empty_store_opens_and_queries_are_none(tmp_path: Path) -> None:
    store = FeatureStore.open(tmp_path)
    store.register(FeatureSpec.mean("m2", 2))
    assert store.get("m2", "nothing") is None
    assert store.point_in_time("m2", "nothing", 0) is None
    assert store.history("m2", "nothing") == []
    assert store.series() == ()
    store.close()
    reopened = FeatureStore.open(tmp_path)
    assert [spec.name for spec in reopened.features()] == ["m2"]
    reopened.close()


def test_series_names_with_spaces_and_unicode(tmp_path: Path) -> None:
    store = FeatureStore.open(tmp_path, columns=("x",))
    store.register(FeatureSpec.mean("m2", 2))
    store.push("BTC / USD", 10, 1.0)
    store.push("BTC / USD", 20, 3.0)
    store.flush()
    directories = list((tmp_path / "features" / "feature=m2").glob("series=*"))
    assert len(directories) == 1
    assert "%20" in directories[0].name
    store.close()
    reopened = FeatureStore.open(tmp_path)
    assert reopened.get("m2", "BTC / USD") == 2.0
    reopened.close()


def test_queries_see_buffered_rows_before_a_flush(tmp_path: Path) -> None:
    store = FeatureStore.open(tmp_path, columns=("x",), flush_every=10_000)
    store.register(FeatureSpec.mean("m2", 2))
    store.push("A", 10, 1.0)
    store.push("A", 20, 3.0)
    assert not list((tmp_path / "features").rglob("*.parquet"))
    assert store.get("m2", "A") == 2.0
    assert store.point_in_time("m2", "A", 10) == 1.0
    store.close()


def test_small_flush_creates_several_files(tmp_path: Path) -> None:
    store = FeatureStore.open(tmp_path, columns=("x",), flush_every=4)
    store.register(FeatureSpec.mean("m2", 2))
    for index in range(20):
        store.push("A", index, float(index))
    store.close()
    files = list((tmp_path / "features" / "feature=m2" / "series=A").glob("*.parquet"))
    assert len(files) >= 4
