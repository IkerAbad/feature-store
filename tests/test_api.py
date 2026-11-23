"""Public API behaviour: catalogue, validation, backfill and stats."""

from __future__ import annotations

import math

import pytest

from conftest import standard_specs, synthetic_ticks
from feature_store import (
    DuplicateFeatureError,
    FeatureSpec,
    FeatureStore,
    FeatureStoreError,
    UnknownColumnError,
    UnknownFeatureError,
    compute_batch,
)
from feature_store._bits import same_bits


def test_register_is_idempotent_for_the_same_spec() -> None:
    store = FeatureStore.memory(columns=("x",))
    spec = FeatureSpec.mean("m5", 5)
    assert store.register(spec) is spec
    assert store.register(FeatureSpec.mean("m5", 5)) == spec
    assert len(store.features()) == 1


def test_register_rejects_a_conflicting_spec() -> None:
    store = FeatureStore.memory(columns=("x",))
    store.register(FeatureSpec.mean("f", 5))
    with pytest.raises(DuplicateFeatureError):
        store.register(FeatureSpec.mean("f", 6))


def test_register_rejects_unknown_columns() -> None:
    store = FeatureStore.memory(columns=("x",))
    with pytest.raises(UnknownColumnError):
        store.register(FeatureSpec.mean("m", 3, input="missing"))
    with pytest.raises(UnknownColumnError):
        store.register(FeatureSpec.corr("c", 3, x="x", y="missing"))


def test_unknown_feature_is_an_error() -> None:
    store = FeatureStore.memory(columns=("x",))
    with pytest.raises(UnknownFeatureError):
        store.get("nope", "S")
    with pytest.raises(UnknownFeatureError):
        store.history("nope", "S")


def test_unknown_column_in_a_record_is_an_error() -> None:
    store = FeatureStore.memory(columns=("x",))
    store.register(FeatureSpec.mean("m", 2))
    with pytest.raises(UnknownColumnError):
        store.ingest("S", 1, z=1.0)


def test_infinite_inputs_are_rejected() -> None:
    store = FeatureStore.memory(columns=("x",))
    store.register(FeatureSpec.mean("m", 2))
    with pytest.raises(ValueError):
        store.push("S", 1, float("inf"))
    with pytest.raises(ValueError):
        store.push_values("S", 1, (float("-inf"),))


def test_nan_is_treated_as_a_missing_observation() -> None:
    store = FeatureStore.memory(columns=("x",))
    store.register(FeatureSpec.mean("m", 2))
    store.push("S", 1, 4.0)
    store.push("S", 2, float("nan"))
    store.push("S", 3, 8.0)
    assert store.get("m", "S") == 6.0


def test_push_values_checks_width() -> None:
    store = FeatureStore.memory(columns=("x", "y"))
    with pytest.raises(ValueError):
        store.push_values("S", 1, (1.0,))
    with pytest.raises(ValueError):
        store.push_values("S", 1, (1.0, 2.0, 3.0))


def test_memory_store_rejects_bad_columns() -> None:
    with pytest.raises(FeatureStoreError):
        FeatureStore.memory(columns=())
    with pytest.raises(FeatureStoreError):
        FeatureStore.memory(columns=("x", "x"))
    with pytest.raises(FeatureStoreError):
        FeatureStore.memory(columns=("x", ""))
    with pytest.raises(FeatureStoreError):
        FeatureStore.memory(allowed_lateness=-1)


def test_late_policy_accepts_strings() -> None:
    store = FeatureStore.memory(columns=("x",), late_policy="recompute")
    assert store.late_policy.value == "recompute"


def test_backfill_after_registration_matches_batch() -> None:
    ticks = synthetic_ticks(9, 30)
    store = FeatureStore.memory(columns=("x", "y"))
    for tick in ticks:
        store.push_values(tick.series, tick.ts, (tick.values["x"], tick.values["y"]), seq=tick.seq)
    specs = [FeatureSpec.mean("m7", 7), FeatureSpec.ewma("e3", 3.0), FeatureSpec.corr("c7", 7)]
    for spec in specs:
        store.register(spec)

    expected = compute_batch(specs, ticks, columns=("x", "y"))
    for spec in specs:
        rows = store.history(spec.name, "S")
        assert len(rows) == len(ticks)
        assert {row.reason for row in rows} == {"backfill"}
        assert {row.version for row in rows} == {1}
        for row in rows:
            assert same_bits(row.value, expected[(spec.name, "S", row.ts, row.seq)])


def test_backfill_does_not_duplicate_existing_outputs() -> None:
    store = FeatureStore.memory(columns=("x",))
    store.register(FeatureSpec.mean("m2", 2))
    for index in range(5):
        store.push("S", index, float(index))
    assert len(store.history("m2", "S")) == 5
    store.register(FeatureSpec.mean("m2", 2))
    assert len(store.history("m2", "S")) == 5


def test_stats_and_context_manager() -> None:
    with FeatureStore.memory(columns=("x", "y")) as store:
        for spec in standard_specs((1, 3)):
            store.register(spec)
        for index in range(6):
            store.ingest("S", index, x=float(index), y=float(index) * 2)
        stats = store.stats()
        assert stats.features == len(standard_specs((1, 3)))
        assert stats.series == 1
        assert stats.records == 6
        assert stats.corrections == 0
        assert stats.late_dropped == 0
        assert stats.rows_emitted == 6 * len(store.features())


def test_columns_property_and_custom_input_names() -> None:
    store = FeatureStore.memory(columns=("px",))
    store.register(FeatureSpec.mean("m", 2, input="px"))
    store.ingest("S", 1, px=10.0)
    store.ingest("S", 2, px=20.0)
    assert store.columns == ("px",)
    assert store.get("m", "S") == 15.0


def test_history_limit_returns_the_most_recent_versions() -> None:
    store = FeatureStore.memory(columns=("x",))
    store.register(FeatureSpec.mean("m2", 2))
    for index in range(10):
        store.push("S", index, float(index))
    full = store.history("m2", "S")
    limited = store.history("m2", "S", limit=3)
    assert limited == full[-3:]


def test_label_is_a_string_and_values_are_floats() -> None:
    store = FeatureStore.memory(columns=("x",))
    store.register(FeatureSpec.mean("m2", 2))
    store.push("S", 1, 1)
    store.push("S", 2, 2)
    value = store.get("m2", "S")
    assert isinstance(value, float)
    assert not math.isnan(value)
