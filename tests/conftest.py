"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence

import pytest

from feature_store import FeatureRow, FeatureSpec, FeatureStore, Tick
from feature_store._bits import same_bits


def synthetic_ticks(
    seed: int,
    n: int,
    *,
    series: str = "S",
    start: int = 1_700_000_000_000_000_000,
    step: int = 1_000_000_000,
    scale: float = 100.0,
) -> list[Tick]:
    """Random-walk ticks with two correlated columns."""
    rng = random.Random(seed)
    price = scale
    ticks: list[Tick] = []
    for index in range(n):
        price *= math.exp(rng.gauss(0.0, 0.01))
        correlated = price * 0.5 + rng.gauss(0.0, 0.05)
        ticks.append(
            Tick(
                series=series,
                ts=start + index * step,
                values={"x": price, "y": correlated},
                seq=index,
            )
        )
    return ticks


def standard_specs(
    windows: Sequence[int] = (1, 2, 4, 20), halflife: float = 5.0
) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for window in windows:
        specs.extend(
            [
                FeatureSpec.mean(f"mean_{window}", window),
                FeatureSpec.std(f"std_{window}", window),
                FeatureSpec.zscore(f"zscore_{window}", window),
                FeatureSpec.cum_return(f"ret_{window}", window),
                FeatureSpec.corr(f"corr_{window}", window),
            ]
        )
    specs.append(FeatureSpec.ewma(f"ewma_{halflife:g}", halflife))
    return specs


def time_specs(durations: Sequence[int] = (3, 12, 60), halflife: float = 5.0) -> list[FeatureSpec]:
    """One spec of every kind for each time-window width in ticks."""
    specs: list[FeatureSpec] = []
    for duration in durations:
        specs.extend(
            [
                FeatureSpec.mean(f"tmean_{duration}", time_window=duration),
                FeatureSpec.std(f"tstd_{duration}", time_window=duration),
                FeatureSpec.zscore(f"tzscore_{duration}", time_window=duration),
                FeatureSpec.cum_return(f"tret_{duration}", time_window=duration),
                FeatureSpec.corr(f"tcorr_{duration}", time_window=duration),
            ]
        )
    specs.append(FeatureSpec.ewma(f"ewma_{halflife:g}", halflife))
    return specs


def assert_same_bits(actual: float, expected: float, context: str = "") -> None:
    assert same_bits(actual, expected), (
        f"{context}: expected {expected!r} ({expected.hex()}), got {actual!r} ({actual.hex()})"
    )


def latest_values(store: FeatureStore, feature: str, series: str) -> dict[tuple[int, int], float]:
    """Latest version of every key, as a plain mapping."""
    rows: dict[tuple[int, int], FeatureRow] = {}
    for row in store.history(feature, series):
        key = (row.ts, row.seq)
        previous = rows.get(key)
        if previous is None or row.version > previous.version:
            rows[key] = row
    return {key: row.value for key, row in rows.items()}


@pytest.fixture
def ticks() -> list[Tick]:
    return synthetic_ticks(0, 50)


def iter_values(ticks: Iterable[Tick], column: str) -> list[float]:
    return [tick.values[column] for tick in ticks]
