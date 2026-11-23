"""Property-based checks over small random streams."""

from __future__ import annotations

import random
from collections.abc import Sequence

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from conftest import latest_values
from feature_store import FeatureSpec, FeatureStore, Tick, compute_batch
from feature_store._bits import same_bits

SPECS = [
    FeatureSpec.mean("mean_3", 3),
    FeatureSpec.std("std_3", 3),
    FeatureSpec.zscore("zscore_3", 3),
    FeatureSpec.cum_return("ret_3", 3),
    FeatureSpec.corr("corr_3", 3),
    FeatureSpec.ewma("ewma_2", 2.0),
    FeatureSpec.mean("tmean_2500", time_window=2500),
    FeatureSpec.zscore("tzscore_2500", time_window=2500),
    FeatureSpec.cum_return("tret_2500", time_window=2500),
    FeatureSpec.corr("tcorr_2500", time_window=2500),
]

SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

PRICES = st.floats(
    min_value=0.5,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def streams(draw: st.DrawFn, max_size: int = 14) -> list[Tick]:
    size = draw(st.integers(min_value=2, max_value=max_size))
    timestamps = draw(st.lists(st.integers(0, 10_000), min_size=size, max_size=size, unique=True))
    xs = draw(st.lists(PRICES, min_size=size, max_size=size))
    ys = draw(st.lists(PRICES, min_size=size, max_size=size))
    return [
        Tick("S", ts, {"x": x, "y": y}, seq=index)
        for index, (ts, x, y) in enumerate(zip(timestamps, xs, ys, strict=True))
    ]


def _feed(store: FeatureStore, ticks: Sequence[Tick]) -> None:
    for tick in ticks:
        store.push_values(tick.series, tick.ts, (tick.values["x"], tick.values["y"]), seq=tick.seq)


@given(ticks=streams(), data=st.data())
@SETTINGS
def test_any_arrival_order_ends_in_the_same_state(ticks: list[Tick], data: st.DataObject) -> None:
    order = data.draw(st.permutations(range(len(ticks))))
    store = FeatureStore.memory(
        columns=("x", "y"), late_policy="recompute", allowed_lateness=10**12
    )
    for spec in SPECS:
        store.register(spec)
    _feed(store, [ticks[index] for index in order])

    expected = compute_batch(SPECS, ticks, columns=("x", "y"))
    for spec in SPECS:
        actual = latest_values(store, spec.name, "S")
        assert len(actual) == len(ticks)
        for key, value in actual.items():
            assert same_bits(value, expected[(spec.name, "S", *key)])


@given(ticks=streams(), data=st.data())
@SETTINGS
def test_point_in_time_matches_the_prefix(ticks: list[Tick], data: st.DataObject) -> None:
    store = FeatureStore.memory(
        columns=("x", "y"), late_policy="recompute", allowed_lateness=10**12
    )
    for spec in SPECS:
        store.register(spec)
    _feed(store, ticks)

    query = data.draw(st.integers(min_value=-10, max_value=10_100))
    prefix = [tick for tick in ticks if tick.ts <= query]
    for spec in SPECS:
        actual = store.point_in_time(spec.name, "S", query)
        if not prefix:
            assert actual is None
            continue
        expected = compute_batch(SPECS, prefix, columns=("x", "y"))
        last = max(prefix, key=lambda tick: tick.ts)
        assert actual is not None
        assert same_bits(actual, expected[(spec.name, "S", last.ts, last.seq)])


@given(ticks=streams(), data=st.data())
@SETTINGS
def test_drop_policy_equals_batch_without_the_dropped_records(
    ticks: list[Tick], data: st.DataObject
) -> None:
    order = data.draw(st.permutations(range(len(ticks))))
    store = FeatureStore.memory(columns=("x", "y"), late_policy="drop", allowed_lateness=0)
    for spec in SPECS:
        store.register(spec)

    accepted: list[Tick] = []
    for index in order:
        tick = ticks[index]
        result = store.push_values(
            tick.series, tick.ts, (tick.values["x"], tick.values["y"]), seq=tick.seq
        )
        if result.status != "late_dropped":
            accepted.append(tick)

    assert len(accepted) + len(store.late_drops()) == len(ticks)

    expected = compute_batch(SPECS, accepted, columns=("x", "y"))
    for spec in SPECS:
        actual = latest_values(store, spec.name, "S")
        assert len(actual) == len(accepted)
        for key, value in actual.items():
            assert same_bits(value, expected[(spec.name, "S", *key)])


@given(seed=st.integers(min_value=0, max_value=10_000))
@SETTINGS
def test_random_history_matches_sorted_replay(seed: int) -> None:
    rng = random.Random(seed)
    ticks = []
    for index in range(12):
        ticks.append(
            Tick(
                "S",
                index * 10 + rng.randint(0, 5),
                {"x": rng.uniform(1, 100), "y": rng.uniform(1, 100)},
                seq=index,
            )
        )
    store = FeatureStore.memory(columns=("x", "y"), late_policy="recompute", allowed_lateness=10**9)
    for spec in SPECS:
        store.register(spec)
    for tick in ticks:
        store.push_values(tick.series, tick.ts, (tick.values["x"], tick.values["y"]), seq=tick.seq)

    expected = compute_batch(SPECS, ticks, columns=("x", "y"))
    for spec in SPECS:
        for key, value in latest_values(store, spec.name, "S").items():
            assert same_bits(value, expected[(spec.name, "S", *key)])
