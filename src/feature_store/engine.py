"""Incremental state machines.

Every feature value is produced by one state machine. The incremental path
feeds the machine one record at a time and keeps the state alive. The batch
path builds a fresh machine per feature and replays the records in event-time
order. Both paths execute the same Python function with the same arguments in
the same order, so the resulting IEEE-754 values are identical, not merely
close.

That property is the whole design: the recurrence below *is* the definition
of the feature. An independent NumPy implementation of the textbook formula is
used in the tests with a relative tolerance, so the recurrence is also checked
against the mathematics it approximates.

The rolling moments use Welford's recurrence with the matching removal step
(Chan, Golub and LeVeque, 1979): the state keeps the mean and the sum of
squared deviations, never the difference of two large sums of squares. The
window is a ring buffer, so every update is O(1) and every eviction is O(1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from ._bits import same_bits
from .spec import FeatureKind, FeatureSpec

NaN = float("nan")


class _State(Protocol):
    def update(self, values: Sequence[float]) -> float:
        """Consume one record and return the feature value for it."""


# ----------------------------------------------------------------------
# State machines. Every one of them is O(1) per record and O(window) memory.
# ----------------------------------------------------------------------
class _MeanState:
    __slots__ = ("_buf", "_count", "_i", "_mean", "_start", "_w")

    def __init__(self, window: int, index: int) -> None:
        self._w = window
        self._i = index
        self._buf = [0.0] * window
        self._start = 0
        self._count = 0
        self._mean = 0.0

    def _add(self, x: float) -> None:
        self._count += 1
        self._mean += (x - self._mean) / self._count

    def _remove(self, x: float) -> None:
        self._count -= 1
        if self._count == 0:
            self._mean = 0.0
        else:
            self._mean -= (x - self._mean) / self._count

    def update(self, values: Sequence[float]) -> float:
        x = values[self._i]
        if x != x:
            return NaN
        if self._count == self._w:
            self._remove(self._buf[self._start])
            self._buf[self._start] = x
            self._add(x)
            start = self._start + 1
            self._start = 0 if start == self._w else start
        else:
            self._buf[(self._start + self._count) % self._w] = x
            self._add(x)
        return self._mean


class _MomentState:
    """Shared Welford state for ``std`` and ``zscore``."""

    __slots__ = ("_buf", "_count", "_i", "_m2", "_mean", "_standardize", "_start", "_w")

    def __init__(self, window: int, index: int, *, standardize: bool) -> None:
        self._w = window
        self._i = index
        self._buf = [0.0] * window
        self._start = 0
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._standardize = standardize

    def _add(self, x: float) -> None:
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (x - self._mean)

    def _remove(self, x: float) -> None:
        self._count -= 1
        if self._count == 0:
            self._mean = 0.0
            self._m2 = 0.0
            return
        delta = x - self._mean
        self._mean -= delta / self._count
        self._m2 -= delta * (x - self._mean)

    def update(self, values: Sequence[float]) -> float:
        x = values[self._i]
        if x != x:
            return NaN
        if self._count == self._w:
            self._remove(self._buf[self._start])
            self._buf[self._start] = x
            self._add(x)
            start = self._start + 1
            self._start = 0 if start == self._w else start
        else:
            self._buf[(self._start + self._count) % self._w] = x
            self._add(x)
        variance = self._m2 / self._count
        if variance < 0.0:
            variance = 0.0
        if not self._standardize:
            return math.sqrt(variance)
        if self._count < 2:
            return NaN
        std = math.sqrt(variance)
        if std == 0.0:
            return NaN
        return (x - self._mean) / std


class _CumReturnState:
    __slots__ = ("_buf", "_count", "_i", "_start", "_w")

    def __init__(self, window: int, index: int) -> None:
        self._w = window
        self._i = index
        self._buf = [0.0] * window
        self._start = 0
        self._count = 0

    def update(self, values: Sequence[float]) -> float:
        x = values[self._i]
        if x != x:
            return NaN
        if self._count == self._w:
            self._buf[self._start] = x
            start = self._start + 1
            self._start = 0 if start == self._w else start
        else:
            self._buf[(self._start + self._count) % self._w] = x
            self._count += 1
        if self._count < 2:
            return NaN
        base = self._buf[self._start] if self._count == self._w else self._buf[0]
        if base == 0.0:
            return NaN
        return x / base - 1.0


class _CorrState:
    __slots__ = (
        "_bx",
        "_by",
        "_c",
        "_count",
        "_ix",
        "_iy",
        "_m2x",
        "_m2y",
        "_mx",
        "_my",
        "_start",
        "_w",
    )

    def __init__(self, window: int, index_x: int, index_y: int) -> None:
        self._w = window
        self._ix = index_x
        self._iy = index_y
        self._bx = [0.0] * window
        self._by = [0.0] * window
        self._start = 0
        self._count = 0
        self._mx = 0.0
        self._my = 0.0
        self._m2x = 0.0
        self._m2y = 0.0
        self._c = 0.0

    def _add(self, x: float, y: float) -> None:
        self._count += 1
        dx = x - self._mx
        dy = y - self._my
        self._mx += dx / self._count
        self._my += dy / self._count
        self._m2x += dx * (x - self._mx)
        self._m2y += dy * (y - self._my)
        self._c += dx * (y - self._my)

    def _remove(self, x: float, y: float) -> None:
        self._count -= 1
        if self._count == 0:
            self._mx = 0.0
            self._my = 0.0
            self._m2x = 0.0
            self._m2y = 0.0
            self._c = 0.0
            return
        dx = x - self._mx
        dy = y - self._my
        self._mx -= dx / self._count
        self._my -= dy / self._count
        self._m2x -= dx * (x - self._mx)
        self._m2y -= dy * (y - self._my)
        self._c -= dx * (y - self._my)

    def update(self, values: Sequence[float]) -> float:
        x = values[self._ix]
        y = values[self._iy]
        if x != x or y != y:
            return NaN
        if self._count == self._w:
            self._remove(self._bx[self._start], self._by[self._start])
            self._bx[self._start] = x
            self._by[self._start] = y
            self._add(x, y)
            start = self._start + 1
            self._start = 0 if start == self._w else start
        else:
            slot = (self._start + self._count) % self._w
            self._bx[slot] = x
            self._by[slot] = y
            self._add(x, y)
        if self._count < 2:
            return NaN
        if self._m2x <= 0.0 or self._m2y <= 0.0:
            return NaN
        if self._count == 2:
            # Two points define a line: the Pearson correlation is exactly ±1.
            return 1.0 if self._c > 0.0 else -1.0
        value = self._c / math.sqrt(self._m2x * self._m2y)
        if value > 1.0:
            return 1.0
        if value < -1.0:
            return -1.0
        return value


class _EwmaState:
    __slots__ = ("_alpha", "_beta", "_i", "_seen", "_value")

    def __init__(self, halflife: float, index: int) -> None:
        self._i = index
        self._alpha = 1.0 - 0.5 ** (1.0 / halflife)
        self._beta = 1.0 - self._alpha
        self._value = 0.0
        self._seen = False

    def update(self, values: Sequence[float]) -> float:
        x = values[self._i]
        if x != x:
            return NaN
        if self._seen:
            self._value = self._alpha * x + self._beta * self._value
        else:
            self._value = x
            self._seen = True
        return self._value


def make_state(spec: FeatureSpec, columns: Sequence[str]) -> _State:
    """Build a fresh state machine for ``spec`` over the given columns."""
    index = {column: position for position, column in enumerate(columns)}
    if spec.kind is FeatureKind.MEAN:
        return _MeanState(spec.window or 0, index[spec.inputs[0]])
    if spec.kind is FeatureKind.STD:
        return _MomentState(spec.window or 0, index[spec.inputs[0]], standardize=False)
    if spec.kind is FeatureKind.ZSCORE:
        return _MomentState(spec.window or 0, index[spec.inputs[0]], standardize=True)
    if spec.kind is FeatureKind.CUM_RETURN:
        return _CumReturnState(spec.window or 0, index[spec.inputs[0]])
    if spec.kind is FeatureKind.CORR:
        return _CorrState(spec.window or 0, index[spec.inputs[0]], index[spec.inputs[1]])
    if spec.kind is FeatureKind.EWMA:
        return _EwmaState(spec.halflife or 1.0, index[spec.inputs[0]])
    raise ValueError(f"unsupported feature kind: {spec.kind}")


# ----------------------------------------------------------------------
# Per-series engine
# ----------------------------------------------------------------------
class _Slot:
    """One feature attached to one series: its state and its output column."""

    __slots__ = ("spec", "state", "values")

    def __init__(self, spec: FeatureSpec, state: _State) -> None:
        self.spec = spec
        self.state = state
        self.values: list[float] = []


class SeriesEngine:
    """Incremental state for every registered feature of one series.

    Records are kept in memory because late data and feature backfills need to
    replay the series. The output column of each feature is kept parallel to
    the record list, so a correction can be compared against the previous
    value without touching the storage layer.
    """

    __slots__ = ("columns", "keys", "max_ts", "next_seq", "records", "series", "slots")

    def __init__(self, series: str, columns: Sequence[str]) -> None:
        self.series = series
        self.columns = tuple(columns)
        self.keys: list[tuple[int, int]] = []
        self.records: list[tuple[int, int, tuple[float, ...]]] = []
        self.slots: list[_Slot] = []
        self.max_ts: int | None = None
        self.next_seq = 0

    def __len__(self) -> int:
        return len(self.records)

    def add_feature(self, spec: FeatureSpec) -> list[float]:
        """Attach a feature and backfill it over the records already seen."""
        slot = _Slot(spec, make_state(spec, self.columns))
        for _, _, values in self.records:
            slot.values.append(slot.state.update(values))
        self.slots.append(slot)
        return slot.values

    def append(self, ts: int, seq: int, values: tuple[float, ...]) -> list[float]:
        """Append a record whose key is greater than every key seen so far."""
        self.keys.append((ts, seq))
        self.records.append((ts, seq, values))
        if self.max_ts is None or ts > self.max_ts:
            self.max_ts = ts
        if seq >= self.next_seq:
            self.next_seq = seq + 1
        output: list[float] = []
        for slot in self.slots:
            value = slot.state.update(values)
            slot.values.append(value)
            output.append(value)
        return output

    def insert(
        self, position: int, ts: int, seq: int, values: tuple[float, ...]
    ) -> list[tuple[str, int, float, float]]:
        """Insert a record at ``position`` and replay the series.

        Returns every output that changed, as ``(feature, position, old, new)``
        tuples. The record that was just inserted is not a correction: callers
        emit it as a new row. The replay rebuilds every state machine from the
        first record, which is the same arithmetic the batch path performs.
        """
        previous = [slot.values for slot in self.slots]
        self.keys.insert(position, (ts, seq))
        self.records.insert(position, (ts, seq, values))
        if self.max_ts is None or ts > self.max_ts:
            self.max_ts = ts
        if seq >= self.next_seq:
            self.next_seq = seq + 1
        self.replay()
        corrections: list[tuple[str, int, float, float]] = []
        for slot, old_values in zip(self.slots, previous, strict=True):
            for index in range(position + 1, len(slot.values)):
                old = old_values[index - 1]
                new = slot.values[index]
                if not same_bits(old, new):
                    corrections.append((slot.spec.name, index, old, new))
        return corrections

    def replay(self) -> None:
        """Rebuild every state machine from the first record."""
        rebuilt = [_Slot(slot.spec, make_state(slot.spec, self.columns)) for slot in self.slots]
        for _, _, values in self.records:
            for slot in rebuilt:
                slot.values.append(slot.state.update(values))
        for slot, fresh in zip(self.slots, rebuilt, strict=True):
            slot.state = fresh.state
            slot.values = fresh.values
