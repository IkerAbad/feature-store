"""Validation and serialisation of FeatureSpec."""

from __future__ import annotations

import pytest

from feature_store import FeatureKind, FeatureSpec, LatePolicy


def test_windowed_constructors() -> None:
    assert FeatureSpec.mean("m", 5) == FeatureSpec("m", FeatureKind.MEAN, window=5, inputs=("x",))
    assert FeatureSpec.std("s", 5).kind is FeatureKind.STD
    assert FeatureSpec.zscore("z", 5).kind is FeatureKind.ZSCORE
    assert FeatureSpec.cum_return("r", 5).kind is FeatureKind.CUM_RETURN
    assert FeatureSpec.corr("c", 5).inputs == ("x", "y")


def test_ewma_constructor() -> None:
    spec = FeatureSpec.ewma("e", 3.5)
    assert spec.kind is FeatureKind.EWMA
    assert spec.window is None
    assert spec.halflife == pytest.approx(3.5)


def test_custom_inputs() -> None:
    spec = FeatureSpec.corr("c", 5, x="bid", y="ask")
    assert spec.inputs == ("bid", "ask")
    assert FeatureSpec.mean("m", 5, input="px").inputs == ("px",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "kind": FeatureKind.MEAN, "window": 5},
        {"name": "1bad", "kind": FeatureKind.MEAN, "window": 5},
        {"name": "has space", "kind": FeatureKind.MEAN, "window": 5},
        {"name": "ok", "kind": FeatureKind.MEAN},
        {"name": "ok", "kind": FeatureKind.MEAN, "window": 0},
        {"name": "ok", "kind": FeatureKind.MEAN, "window": 5, "halflife": 2.0},
        {"name": "ok", "kind": FeatureKind.EWMA, "window": 5, "halflife": 2.0},
        {"name": "ok", "kind": FeatureKind.EWMA},
        {"name": "ok", "kind": FeatureKind.EWMA, "halflife": 0.0},
        {"name": "ok", "kind": FeatureKind.EWMA, "halflife": float("inf")},
        {"name": "ok", "kind": FeatureKind.CORR, "window": 5, "inputs": ("x",)},
        {"name": "ok", "kind": FeatureKind.CORR, "window": 5, "inputs": ("x", "x")},
        {"name": "ok", "kind": FeatureKind.MEAN, "window": 5, "inputs": ()},
        {"name": "ok", "kind": FeatureKind.MEAN, "window": 5, "inputs": ("",)},
    ],
)
def test_invalid_specs(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        FeatureSpec(**kwargs)  # type: ignore[arg-type]


def test_json_round_trip() -> None:
    specs = [
        FeatureSpec.mean("mean_5", 5, description="five ticks"),
        FeatureSpec.corr("corr_10", 10, x="a", y="b"),
        FeatureSpec.ewma("ewma_2", 2.5),
    ]
    for spec in specs:
        assert FeatureSpec.from_dict(spec.to_dict()) == spec


def test_from_dict_accepts_strings() -> None:
    spec = FeatureSpec.from_dict({"name": "m", "kind": "mean", "window": 3, "inputs": ["px"]})
    assert spec.kind is FeatureKind.MEAN
    assert spec.inputs == ("px",)


def test_str_is_readable() -> None:
    assert "window=5" in str(FeatureSpec.mean("m", 5))
    assert "halflife=2" in str(FeatureSpec.ewma("e", 2.0))


def test_late_policy_values() -> None:
    assert LatePolicy("drop") is LatePolicy.DROP
    assert LatePolicy("recompute") is LatePolicy.RECOMPUTE
    with pytest.raises(ValueError):
        LatePolicy("ignore")
