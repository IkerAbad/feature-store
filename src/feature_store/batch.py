"""From-scratch batch computation.

``compute_batch`` is the offline counterpart of the incremental store: it sorts
the records by ``(ts, seq)``, builds a fresh state machine per feature and
series, and replays the whole history. It is the oracle used by the tests and
by the benchmark.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .engine import SeriesEngine
from .spec import FeatureSpec

NaN = float("nan")


@dataclass(frozen=True, slots=True)
class Tick:
    """One input record for the batch path."""

    series: str
    ts: int
    values: Mapping[str, float]
    seq: int = 0


def compute_batch(
    specs: Sequence[FeatureSpec],
    ticks: Iterable[Tick],
    *,
    columns: Sequence[str] = ("x",),
) -> dict[tuple[str, str, int, int], float]:
    """Compute every feature for every tick from scratch.

    Returns a mapping keyed by ``(feature, series, ts, seq)``. Records are
    processed in ``(ts, seq)`` order per series, exactly like the incremental
    store does after it has accepted them. Keys are expected to be unique
    within a series, as the store enforces.
    """
    names = tuple(columns)
    grouped: dict[str, list[Tick]] = {}
    for tick in ticks:
        grouped.setdefault(tick.series, []).append(tick)
    result: dict[tuple[str, str, int, int], float] = {}
    for series, series_ticks in grouped.items():
        engine = SeriesEngine(series, names)
        for spec in specs:
            missing = sorted(set(spec.inputs) - set(names))
            if missing:
                raise ValueError(f"feature {spec.name!r} needs missing columns {missing}")
            engine.add_feature(spec)
        for tick in sorted(series_ticks, key=lambda item: (item.ts, item.seq)):
            encoded = tuple(float(tick.values.get(name, NaN)) for name in names)
            engine.append(tick.ts, tick.seq, encoded)
        for slot in engine.slots:
            for (ts, seq, _), value in zip(engine.records, slot.values, strict=True):
                result[(slot.spec.name, series, ts, seq)] = value
    return result


__all__ = ["Tick", "compute_batch"]
