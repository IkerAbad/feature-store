"""Watermarks, late data and the two policies."""

from __future__ import annotations

import pytest

from conftest import latest_values, standard_specs, synthetic_ticks, time_specs
from feature_store import FeatureSpec, FeatureStore, Tick, compute_batch
from feature_store._bits import same_bits
from feature_store.errors import DuplicateRecordError


def _store(policy: str, lateness: int = 0) -> FeatureStore:
    store = FeatureStore.memory(columns=("x", "y"), late_policy=policy, allowed_lateness=lateness)
    for spec in standard_specs((1, 2, 5)):
        store.register(spec)
    return store


def _feed(store: FeatureStore, ticks: list) -> None:
    for tick in ticks:
        store.push_values(tick.series, tick.ts, (tick.values["x"], tick.values["y"]), seq=tick.seq)


def test_drop_policy_ignores_the_late_record() -> None:
    ticks = synthetic_ticks(0, 20)
    store = _store("drop")
    _feed(store, ticks)
    late = ticks[5]
    result = store.push_values(late.series, late.ts, (999.0, 999.0), seq=1000)
    assert result.status == "late_dropped"
    assert result.corrections == 0
    assert result.watermark == ticks[-1].ts

    expected = compute_batch(standard_specs((1, 2, 5)), ticks, columns=("x", "y"))
    for feature in ("mean_5", "std_5", "corr_5"):
        actual = latest_values(store, feature, late.series)
        assert len(actual) == len(ticks)
        for key, value in actual.items():
            assert same_bits(value, expected[(feature, late.series, *key)])

    drops = store.late_drops()
    assert len(drops) == 1
    assert (drops[0].ts, drops[0].seq) == (late.ts, 1000)
    assert drops[0].policy == "drop"
    assert drops[0].watermark == ticks[-1].ts


def test_recompute_policy_restores_batch_equivalence() -> None:
    ticks = synthetic_ticks(1, 25)
    store = _store("recompute", lateness=0)
    _feed(store, ticks)
    late = ticks[7]
    result = store.push_values(late.series, late.ts, (500.0, 250.0), seq=10_000)
    assert result.status == "late_recomputed"
    assert result.corrections > 0

    everything = [
        *ticks,
        Tick(
            series=late.series,
            ts=late.ts,
            values={"x": 500.0, "y": 250.0},
            seq=10_000,
        ),
    ]
    specs = standard_specs((1, 2, 5))
    expected = compute_batch(specs, everything, columns=("x", "y"))
    for spec in specs:
        actual = latest_values(store, spec.name, late.series)
        assert len(actual) == len(everything)
        for key, value in actual.items():
            assert same_bits(value, expected[(spec.name, late.series, *key)])


def test_watermark_boundary_is_inclusive_on_the_watermark() -> None:
    ticks = synthetic_ticks(2, 5)
    store = _store("drop", lateness=10)
    _feed(store, ticks)
    max_ts = ticks[-1].ts
    watermark = max_ts - 10

    accepted = store.push_values("S", watermark, (1.0, 2.0), seq=99)
    assert accepted.status == "out_of_order"

    rejected = store.push_values("S", watermark - 1, (1.0, 2.0), seq=100)
    assert rejected.status == "late_dropped"
    assert rejected.watermark == watermark


def test_out_of_order_inside_lateness_is_replayed() -> None:
    ticks = synthetic_ticks(3, 12)
    store = _store("recompute", lateness=10**18)
    _feed(store, ticks)
    past = store.point_in_time("mean_5", "S", ticks[2].ts)
    result = store.push_values("S", ticks[3].ts + 1, (0.5, 0.25), seq=100)
    assert result.status == "out_of_order"
    assert result.corrections > 0
    assert store.point_in_time("mean_5", "S", ticks[2].ts) == past

    late_tick = Tick(series="S", ts=ticks[3].ts + 1, values={"x": 0.5, "y": 0.25}, seq=100)
    expected = compute_batch(standard_specs((1, 2, 5)), [*ticks, late_tick], columns=("x", "y"))
    assert (
        store.point_in_time("mean_5", "S", late_tick.ts)
        == expected[("mean_5", "S", late_tick.ts, 100)]
    )


def test_time_window_late_recompute_matches_batch() -> None:
    ticks = synthetic_ticks(11, 30)
    specs = time_specs((3, 12))
    store = FeatureStore.memory(columns=("x", "y"), late_policy="recompute", allowed_lateness=0)
    for spec in specs:
        store.register(spec)
    _feed(store, ticks)
    late = Tick("S", ticks[10].ts + 1, {"x": 7.0, "y": 3.0}, seq=10_000)
    result = store.push_values("S", late.ts, (7.0, 3.0), seq=late.seq)
    assert result.status == "late_recomputed"

    everything = [*ticks, late]
    expected = compute_batch(specs, everything, columns=("x", "y"))
    for spec in specs:
        actual = latest_values(store, spec.name, "S")
        assert len(actual) == len(everything)
        for key, value in actual.items():
            assert same_bits(value, expected[(spec.name, "S", *key)])


def test_same_key_twice_is_rejected() -> None:
    store = _store("recompute")
    store.push_values("S", 100, (1.0, 2.0), seq=7)
    with pytest.raises(DuplicateRecordError):
        store.push_values("S", 100, (3.0, 4.0), seq=7)


def test_versions_increase_with_each_correction() -> None:
    store = _store("recompute")
    for index in range(10):
        store.push_values("S", index * 10, (float(index), float(index)), seq=index)
    first = store.push_values("S", 45, (100.0, 200.0), seq=50)
    second = store.push_values("S", 42, (-100.0, -200.0), seq=51)
    assert first.corrections > 0
    assert second.corrections > 0

    rows = store.history("mean_2", "S")
    by_key: dict[tuple[int, int], list[int]] = {}
    for row in rows:
        by_key.setdefault((row.ts, row.seq), []).append(row.version)
    for versions in by_key.values():
        assert versions == sorted(versions)
        assert versions[0] == 1
        assert len(set(versions)) == len(versions)
    assert max(len(versions) for versions in by_key.values()) >= 2


def test_late_record_only_corrects_from_its_position_onwards() -> None:
    store = FeatureStore.memory(columns=("x",), late_policy="recompute", allowed_lateness=0)
    store.register(FeatureSpec.mean("m3", 3))
    for index in range(20):
        store.push("S", index * 10, float(index))
    result = store.push("S", 185, 100.0, seq=999)
    assert result.corrections == 1
    history = store.history("m3", "S")
    corrected = {row.ts for row in history if row.version > 1}
    assert corrected == {190.0}
    assert len({(row.ts, row.seq) for row in history}) == 21


def test_late_record_does_not_touch_the_past() -> None:
    ticks = synthetic_ticks(4, 20)
    store = _store("recompute")
    _feed(store, ticks)
    before = {
        ts: store.point_in_time("mean_5", "S", ts) for ts in (ticks[2].ts, ticks[5].ts, ticks[8].ts)
    }
    store.push_values("S", ticks[10].ts + 1, (1e6, 1e6), seq=10_000)
    after = {
        ts: store.point_in_time("mean_5", "S", ts) for ts in (ticks[2].ts, ticks[5].ts, ticks[8].ts)
    }
    assert before == after


def test_stats_track_corrections_and_drops() -> None:
    dropping = _store("drop")
    _feed(dropping, synthetic_ticks(5, 10))
    dropping.push_values("S", 1, (0.0, 0.0), seq=500)
    stats = dropping.stats()
    assert stats.late_dropped == 1
    assert stats.corrections == 0
    assert stats.records == 10

    recomputing = _store("recompute")
    _feed(recomputing, synthetic_ticks(5, 10))
    recomputing.push_values("S", 1, (0.0, 0.0), seq=500)
    stats = recomputing.stats()
    assert stats.late_dropped == 0
    assert stats.corrections > 0
    assert stats.records == 11
