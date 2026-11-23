"""Each feature against an independent NumPy implementation.

The incremental path is exact with respect to its own recurrence (see
``test_equivalence``). These tests check that the recurrence implements the
textbook statistic, with a relative tolerance because the two formulations
round differently.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from feature_store import FeatureKind, FeatureSpec, FeatureStore

# The state machines keep Welford moments, so the agreement with NumPy is
# limited by the rounding of the sliding removal step, not by the formula.
RTOL = 1e-8
ATOL = 1e-12


def _run(spec: FeatureSpec, xs: list[float], ys: list[float] | None = None) -> list[float]:
    store = FeatureStore.memory(columns=("x", "y"))
    store.register(spec)
    values: list[float] = []
    for index, x in enumerate(xs):
        if ys is None:
            store.ingest("A", index, x=x)
        else:
            store.ingest("A", index, x=x, y=ys[index])
        value = store.point_in_time(spec.name, "A", index)
        assert value is not None
        values.append(value)
    return values


def _reference(spec: FeatureSpec, xs: list[float], ys: list[float] | None) -> list[float]:
    result: list[float] = []
    if spec.kind.value == "ewma":
        assert spec.halflife is not None
        alpha = 1.0 - 0.5 ** (1.0 / spec.halflife)
        beta = 1.0 - alpha
        for index in range(len(xs)):
            terms = [beta**index * xs[0]]
            terms.extend(alpha * beta ** (index - k) * xs[k] for k in range(1, index + 1))
            result.append(math.fsum(terms))
        return result

    assert spec.window is not None
    window_size = spec.window
    for index in range(len(xs)):
        start = max(0, index - window_size + 1)
        window = np.asarray(xs[start : index + 1], dtype=np.float64)
        if spec.kind.value == "mean":
            value = float(np.mean(window))
        elif spec.kind.value == "std":
            value = float(np.std(window, ddof=0))
        elif spec.kind.value == "zscore":
            if len(window) < 2 or np.std(window, ddof=0) == 0.0:
                value = math.nan
            else:
                value = float((window[-1] - np.mean(window)) / np.std(window, ddof=0))
        elif spec.kind.value == "cum_return":
            if len(window) < 2 or window[0] == 0.0:
                value = math.nan
            else:
                value = float(window[-1] / window[0] - 1.0)
        elif spec.kind.value == "corr":
            assert ys is not None
            other = np.asarray(ys[start : index + 1], dtype=np.float64)
            if len(window) < 2 or np.std(window) == 0.0 or np.std(other) == 0.0:
                value = math.nan
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    value = float(np.corrcoef(window, other)[0, 1])
        else:  # pragma: no cover - guard against a new kind
            raise AssertionError(spec.kind)
        result.append(value)
    return result


@pytest.mark.parametrize("window", [1, 2, 3, 5, 17])
@pytest.mark.parametrize("kind", ["mean", "std", "zscore", "cum_return", "corr"])
def test_matches_numpy_reference(kind: str, window: int) -> None:
    rng = np.random.default_rng(window * 100 + len(kind))
    xs = list(100.0 + rng.standard_normal(80).cumsum())
    ys = list(50.0 + rng.standard_normal(80).cumsum())
    if kind == "corr":
        spec = FeatureSpec.corr(f"f_{kind}_{window}", window)
    else:
        spec = FeatureSpec(
            f"f_{kind}_{window}",
            FeatureKind(kind),
            window=window,
            inputs=("x",),
        )
    actual = _run(spec, xs, ys)
    expected = _reference(spec, xs, ys)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("halflife", [0.5, 1.0, 5.0, 60.0])
def test_ewma_matches_closed_form(halflife: float) -> None:
    rng = np.random.default_rng(int(halflife * 10))
    xs = list(100.0 + rng.standard_normal(60).cumsum())
    spec = FeatureSpec.ewma("e", halflife)
    np.testing.assert_allclose(
        _run(spec, xs), _reference(spec, xs, None), rtol=RTOL, atol=ATOL, equal_nan=True
    )


def test_warmup_values() -> None:
    xs = [10.0, 11.0, 12.0]
    mean = _run(FeatureSpec.mean("m", 5), xs)
    assert mean == pytest.approx([10.0, 10.5, 11.0])

    std = _run(FeatureSpec.std("s", 5), xs)
    assert std[0] == 0.0
    assert math.isnan(_run(FeatureSpec.zscore("z", 5), xs)[0])
    assert math.isnan(_run(FeatureSpec.cum_return("r", 5), xs)[0])

    corr = _run(FeatureSpec.corr("c", 5), xs, xs)
    assert math.isnan(corr[0])

    ewma = _run(FeatureSpec.ewma("e", 3.0), xs)
    assert ewma[0] == 10.0


def test_constant_series() -> None:
    xs = [42.0] * 10
    assert _run(FeatureSpec.mean("m", 4), xs) == [42.0] * 10
    assert _run(FeatureSpec.std("s", 4), xs) == [0.0] * 10
    assert all(math.isnan(value) for value in _run(FeatureSpec.zscore("z", 4), xs))
    returns = _run(FeatureSpec.cum_return("r", 4), xs)
    assert math.isnan(returns[0])
    assert returns[1:] == [0.0] * 9
    assert all(math.isnan(value) for value in _run(FeatureSpec.corr("c", 4), xs, xs))


def test_missing_input_is_skipped_not_counted() -> None:
    spec = FeatureSpec.mean("m", 2)
    store = FeatureStore.memory(columns=("x",))
    store.register(spec)
    values = [1.0, float("nan"), 3.0, 5.0]
    outputs = []
    for index, value in enumerate(values):
        store.push("A", index, value)
        outputs.append(store.point_in_time("m", "A", index))
    assert outputs[0] == 1.0
    assert math.isnan(outputs[1])  # type: ignore[arg-type]
    assert outputs[2] == 2.0  # only 1 and 3 are in the window
    assert outputs[3] == 4.0


def test_cum_return_undefined_with_zero_base() -> None:
    returns = _run(FeatureSpec.cum_return("r", 2), [0.0, 1.0, 2.0])
    assert math.isnan(returns[0])
    assert math.isnan(returns[1])
    assert returns[2] == pytest.approx(1.0)


def test_correlation_is_clamped() -> None:
    xs = [float(index) for index in range(10)]
    corr = _run(FeatureSpec.corr("c", 9), xs, [2.0 * x for x in xs])
    assert corr[-1] == 1.0
    assert all(-1.0 <= value <= 1.0 for value in corr if not math.isnan(value))


def test_window_one() -> None:
    xs = [3.0, 7.0, 11.0]
    assert _run(FeatureSpec.mean("m", 1), xs) == xs
    assert _run(FeatureSpec.std("s", 1), xs) == [0.0, 0.0, 0.0]
    assert all(math.isnan(value) for value in _run(FeatureSpec.cum_return("r", 1), xs))
