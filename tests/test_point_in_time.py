"""Point-in-time lookups and the no-future-leakage contract."""

from __future__ import annotations

import pytest

from conftest import standard_specs, synthetic_ticks
from feature_store import FeatureSpec, FeatureStore, compute_batch
from feature_store._bits import same_bits


def _feed(store: FeatureStore, ticks: list) -> None:
    for tick in ticks:
        store.push_values(tick.series, tick.ts, (tick.values["x"], tick.values["y"]), seq=tick.seq)


def _value(store: FeatureStore, feature: str, ts: int) -> float:
    value = store.point_in_time(feature, "S", ts)
    assert value is not None
    return value


def test_value_at_t_only_uses_records_up_to_t() -> None:
    ticks = synthetic_ticks(0, 40)
    specs = standard_specs((1, 3, 10))
    store = FeatureStore.memory(columns=("x", "y"))
    for spec in specs:
        store.register(spec)
    _feed(store, ticks)

    for index, tick in enumerate(ticks):
        prefix = compute_batch(specs, ticks[: index + 1], columns=("x", "y"))
        for spec in specs:
            actual = store.point_in_time(spec.name, "S", tick.ts)
            expected = prefix[(spec.name, "S", tick.ts, tick.seq)]
            assert actual is not None
            assert same_bits(actual, expected), f"{spec.name} at {tick.ts}"


def test_future_records_do_not_change_the_past() -> None:
    ticks = synthetic_ticks(1, 30)
    specs = standard_specs((1, 4, 12))
    store = FeatureStore.memory(columns=("x", "y"))
    for spec in specs:
        store.register(spec)
    _feed(store, ticks[:20])
    before = {
        (spec.name, tick.ts): _value(store, spec.name, tick.ts)
        for spec in specs
        for tick in ticks[:20]
    }
    _feed(store, ticks[20:])
    for (feature, ts), value in before.items():
        assert same_bits(_value(store, feature, ts), value)


def test_lookup_is_inclusive_at_an_exact_key() -> None:
    ticks = synthetic_ticks(2, 10)
    store = FeatureStore.memory(columns=("x", "y"))
    store.register(FeatureSpec.mean("m3", 3))
    _feed(store, ticks)
    expected = compute_batch([FeatureSpec.mean("m3", 3)], ticks, columns=("x", "y"))
    tick = ticks[6]
    assert store.point_in_time("m3", "S", tick.ts) == expected[("m3", "S", tick.ts, tick.seq)]


def test_lookup_between_keys_returns_the_previous_one() -> None:
    store = FeatureStore.memory(columns=("x",))
    store.register(FeatureSpec.mean("m2", 2))
    store.push("S", 100, 1.0)
    store.push("S", 200, 3.0)
    value_at_200 = store.point_in_time("m2", "S", 200)
    assert store.point_in_time("m2", "S", 150) == 1.0
    assert store.point_in_time("m2", "S", 199) == 1.0
    assert store.point_in_time("m2", "S", 200) == value_at_200


def test_before_the_first_record_there_is_nothing() -> None:
    store = FeatureStore.memory(columns=("x",))
    store.register(FeatureSpec.mean("m2", 2))
    assert store.get("m2", "S") is None
    assert store.point_in_time("m2", "S", 0) is None
    store.push("S", 100, 1.0)
    assert store.point_in_time("m2", "S", 99) is None
    assert store.point_in_time("m2", "S", 100) == 1.0


def test_dropped_late_record_changes_nothing() -> None:
    ticks = synthetic_ticks(3, 25)
    store = FeatureStore.memory(columns=("x", "y"), late_policy="drop")
    specs = standard_specs((1, 3))
    for spec in specs:
        store.register(spec)
    _feed(store, ticks)
    before = {
        (spec.name, ts): _value(store, spec.name, ts)
        for spec in specs
        for ts in (ticks[4].ts, ticks[10].ts, ticks[-1].ts)
    }
    store.push_values("S", ticks[6].ts, (1e9, 1e9), seq=9999)
    for (feature, ts), value in before.items():
        assert same_bits(_value(store, feature, ts), value)


def test_duplicate_timestamps_follow_seq() -> None:
    store = FeatureStore.memory(columns=("x",))
    store.register(FeatureSpec.cum_return("r", 2))
    store.push("S", 1_000, 10.0, seq=0)
    store.push("S", 1_000, 20.0, seq=1)
    assert store.point_in_time("r", "S", 999) is None
    assert store.point_in_time("r", "S", 1_000) == pytest.approx(1.0)
    assert store.get("r", "S") == pytest.approx(1.0)
