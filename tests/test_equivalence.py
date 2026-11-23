"""Incremental versus batch: bit-for-bit equality.

The acceptance criterion is exact equality, not a tolerance. The batch path
builds fresh state machines and replays the records in event-time order; the
incremental path carries the same machines across calls. If state management is
wrong (wrong eviction, wrong insertion point, wrong warm-up) the bits diverge.
"""

from __future__ import annotations

import random

import pytest

from conftest import latest_values, standard_specs, synthetic_ticks, time_specs
from feature_store import FeatureStore, Tick, compute_batch
from feature_store._bits import same_bits


def _assert_store_matches(store: FeatureStore, specs: list, ticks: list[Tick], series: str) -> None:
    expected = compute_batch(specs, ticks, columns=("x", "y"))
    assert expected
    for spec in specs:
        actual = latest_values(store, spec.name, series)
        assert len(actual) == len(ticks)
        for (ts, seq), value in actual.items():
            want = expected[(spec.name, series, ts, seq)]
            assert same_bits(value, want), (
                f"{spec.name} at ({ts}, {seq}): {value.hex()} != {want.hex()}"
            )


@pytest.mark.parametrize("seed", range(20))
def test_twenty_series_incremental_equals_batch(seed: int) -> None:
    ticks = synthetic_ticks(seed, 40 + 6 * seed, series=f"S{seed}")
    specs = standard_specs((1, 2, 4, 20))
    store = FeatureStore.memory(columns=("x", "y"))
    for spec in specs:
        store.register(spec)
    for tick in ticks:
        store.push_values(
            tick.series,
            tick.ts,
            (tick.values["x"], tick.values["y"]),
            seq=tick.seq,
        )
    _assert_store_matches(store, specs, ticks, f"S{seed}")


@pytest.mark.parametrize("seed", range(5))
def test_shuffled_arrivals_are_also_bit_exact(seed: int) -> None:
    ticks = synthetic_ticks(100 + seed, 70, series="S")
    specs = standard_specs((1, 3, 7))
    store = FeatureStore.memory(
        columns=("x", "y"), late_policy="recompute", allowed_lateness=10**12
    )
    for spec in specs:
        store.register(spec)
    order = list(range(len(ticks)))
    random.Random(seed).shuffle(order)
    for index in order:
        tick = ticks[index]
        store.push_values(tick.series, tick.ts, (tick.values["x"], tick.values["y"]), seq=tick.seq)
    _assert_store_matches(store, specs, ticks, "S")


def test_duplicate_timestamps_are_ordered_by_seq() -> None:
    ticks = [
        Tick("S", 1_000, {"x": 10.0, "y": 1.0}, seq=0),
        Tick("S", 1_000, {"x": 11.0, "y": 2.0}, seq=1),
        Tick("S", 1_000, {"x": 13.0, "y": 4.0}, seq=2),
        Tick("S", 2_000, {"x": 12.0, "y": 3.0}, seq=3),
    ]
    specs = standard_specs((1, 2, 3), halflife=2.0)
    store = FeatureStore.memory(columns=("x", "y"))
    for spec in specs:
        store.register(spec)
    for tick in ticks:
        store.push_values(tick.series, tick.ts, (tick.values["x"], tick.values["y"]), seq=tick.seq)
    _assert_store_matches(store, specs, ticks, "S")


def test_multiple_series_share_the_feature_definitions() -> None:
    ticks = synthetic_ticks(7, 40, series="A") + synthetic_ticks(8, 55, series="B")
    specs = standard_specs((1, 5))
    store = FeatureStore.memory(columns=("x", "y"))
    for spec in specs:
        store.register(spec)
    for tick in ticks:
        store.push_values(tick.series, tick.ts, (tick.values["x"], tick.values["y"]), seq=tick.seq)
    expected = compute_batch(specs, ticks, columns=("x", "y"))
    for spec in specs:
        for series in ("A", "B"):
            actual = latest_values(store, spec.name, series)
            for (ts, seq), value in actual.items():
                assert same_bits(value, expected[(spec.name, series, ts, seq)])


def test_missing_column_keeps_batch_and_incremental_equal() -> None:
    ticks = [Tick("S", index, {"x": float(index + 1)}, seq=index) for index in range(30)]
    specs = standard_specs((1, 3, 8))
    store = FeatureStore.memory(columns=("x", "y"))
    for spec in specs:
        store.register(spec)
    for tick in ticks:
        store.push_values(tick.series, tick.ts, (tick.values["x"], float("nan")), seq=tick.seq)
    _assert_store_matches(store, specs, ticks, "S")
    # Every correlation value must be NaN: the y column was never observed.
    values = latest_values(store, "corr_8", "S")
    assert all(value != value for value in values.values())


@pytest.mark.parametrize("seed", range(20))
def test_time_windows_incremental_equals_batch(seed: int) -> None:
    ticks = synthetic_ticks(seed, 40 + 6 * seed, series=f"T{seed}")
    specs = time_specs((1, 3, 12))
    store = FeatureStore.memory(columns=("x", "y"))
    for spec in specs:
        store.register(spec)
    for tick in ticks:
        store.push_values(
            tick.series,
            tick.ts,
            (tick.values["x"], tick.values["y"]),
            seq=tick.seq,
        )
    _assert_store_matches(store, specs, ticks, f"T{seed}")


@pytest.mark.parametrize("seed", range(5))
def test_time_windows_with_shuffled_arrivals(seed: int) -> None:
    ticks = synthetic_ticks(200 + seed, 70, series="T")
    specs = time_specs((2, 5, 20))
    store = FeatureStore.memory(
        columns=("x", "y"), late_policy="recompute", allowed_lateness=10**12
    )
    for spec in specs:
        store.register(spec)
    order = list(range(len(ticks)))
    random.Random(seed).shuffle(order)
    for index in order:
        tick = ticks[index]
        store.push_values(tick.series, tick.ts, (tick.values["x"], tick.values["y"]), seq=tick.seq)
    _assert_store_matches(store, specs, ticks, "T")


def test_time_windows_with_identical_timestamps() -> None:
    ticks = [
        Tick("S", 1_000, {"x": 10.0, "y": 1.0}, seq=0),
        Tick("S", 1_000, {"x": 11.0, "y": 2.0}, seq=1),
        Tick("S", 1_500, {"x": 13.0, "y": 4.0}, seq=2),
        Tick("S", 2_000, {"x": 12.0, "y": 3.0}, seq=3),
    ]
    specs = time_specs((500, 1_001))
    store = FeatureStore.memory(columns=("x", "y"))
    for spec in specs:
        store.register(spec)
    for tick in ticks:
        store.push_values(tick.series, tick.ts, (tick.values["x"], tick.values["y"]), seq=tick.seq)
    _assert_store_matches(store, specs, ticks, "S")


def test_batch_is_independent_of_input_order() -> None:
    ticks = synthetic_ticks(3, 50)
    specs = standard_specs((1, 4))
    ordered = compute_batch(specs, ticks, columns=("x", "y"))
    shuffled = list(ticks)
    random.Random(11).shuffle(shuffled)
    unordered = compute_batch(specs, shuffled, columns=("x", "y"))
    assert ordered.keys() == unordered.keys()
    for key in ordered:
        assert same_bits(ordered[key], unordered[key])


def test_bit_helper_detects_signed_zero_and_nan() -> None:
    assert same_bits(0.0, -0.0) is False
    assert same_bits(float("nan"), float("nan")) is True
    assert same_bits(1.0, 1.0 + 2**-52) is False
