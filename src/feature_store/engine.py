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
squared deviations, never the difference of two large sums of squares.

Count windows keep a ring buffer; time windows keep a deque of
``(timestamp, value)`` pairs and evict everything older than ``ts - duration``.
Both are O(1) amortised per record.

Every state can be serialised with :meth:`snapshot` and rebuilt with
:meth:`load`. The feature store uses that to checkpoint the engine.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from typing import Any, Protocol

from ._bits import same_bits
from .spec import FeatureKind, FeatureSpec

NaN = float("nan")
Snapshot = dict[str, Any]


class _State(Protocol):
    def update(self, ts: int, values: Sequence[float]) -> float:
        """Consume one record and return the feature value for it."""

    def snapshot(self) -> Snapshot:
        """Exact, serialisable copy of the state."""

    def load(self, payload: Snapshot) -> None:
        """Restore a snapshot produced by :meth:`snapshot`."""


def _empty_snapshot() -> Snapshot:
    return {
        "n": 0,
        "mean": 0.0,
        "m2": 0.0,
        "ewma": 0.0,
        "seen": False,
        "mean_y": 0.0,
        "m2_y": 0.0,
        "c": 0.0,
        "times": [],
        "xs": [],
        "ys": [],
    }


# ----------------------------------------------------------------------
# Count windows
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

    def update(self, ts: int, values: Sequence[float]) -> float:
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

    def snapshot(self) -> Snapshot:
        payload = _empty_snapshot()
        payload.update(n=self._count, mean=self._mean)
        payload["xs"] = [self._buf[(self._start + k) % self._w] for k in range(self._count)]
        return payload

    def load(self, payload: Snapshot) -> None:
        xs = [float(value) for value in payload["xs"]]
        self._count = len(xs)
        self._mean = float(payload["mean"])
        self._buf = xs + [0.0] * (self._w - len(xs))
        self._start = 0


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

    def update(self, ts: int, values: Sequence[float]) -> float:
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

    def snapshot(self) -> Snapshot:
        payload = _empty_snapshot()
        payload.update(n=self._count, mean=self._mean, m2=self._m2)
        payload["xs"] = [self._buf[(self._start + k) % self._w] for k in range(self._count)]
        return payload

    def load(self, payload: Snapshot) -> None:
        xs = [float(value) for value in payload["xs"]]
        self._count = len(xs)
        self._mean = float(payload["mean"])
        self._m2 = float(payload["m2"])
        self._buf = xs + [0.0] * (self._w - len(xs))
        self._start = 0


class _CumReturnState:
    __slots__ = ("_buf", "_count", "_i", "_start", "_w")

    def __init__(self, window: int, index: int) -> None:
        self._w = window
        self._i = index
        self._buf = [0.0] * window
        self._start = 0
        self._count = 0

    def update(self, ts: int, values: Sequence[float]) -> float:
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

    def snapshot(self) -> Snapshot:
        payload = _empty_snapshot()
        payload["n"] = self._count
        payload["xs"] = [self._buf[(self._start + k) % self._w] for k in range(self._count)]
        return payload

    def load(self, payload: Snapshot) -> None:
        xs = [float(value) for value in payload["xs"]]
        self._count = len(xs)
        self._buf = xs + [0.0] * (self._w - len(xs))
        self._start = 0


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

    def _value(self) -> float:
        if self._count < 2:
            return NaN
        if self._m2x <= 0.0 or self._m2y <= 0.0:
            return NaN
        if self._count == 2:
            return 1.0 if self._c > 0.0 else -1.0
        value = self._c / math.sqrt(self._m2x * self._m2y)
        if value > 1.0:
            return 1.0
        if value < -1.0:
            return -1.0
        return value

    def update(self, ts: int, values: Sequence[float]) -> float:
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
        return self._value()

    def snapshot(self) -> Snapshot:
        payload = _empty_snapshot()
        payload.update(
            n=self._count, mean=self._mx, mean_y=self._my, m2=self._m2x, m2_y=self._m2y, c=self._c
        )
        payload["xs"] = [self._bx[(self._start + k) % self._w] for k in range(self._count)]
        payload["ys"] = [self._by[(self._start + k) % self._w] for k in range(self._count)]
        return payload

    def load(self, payload: Snapshot) -> None:
        xs = [float(value) for value in payload["xs"]]
        ys = [float(value) for value in payload["ys"]]
        self._count = len(xs)
        self._mx = float(payload["mean"])
        self._my = float(payload["mean_y"])
        self._m2x = float(payload["m2"])
        self._m2y = float(payload["m2_y"])
        self._c = float(payload["c"])
        self._bx = xs + [0.0] * (self._w - len(xs))
        self._by = ys + [0.0] * (self._w - len(ys))
        self._start = 0


class _EwmaState:
    __slots__ = ("_alpha", "_beta", "_i", "_seen", "_value")

    def __init__(self, halflife: float, index: int) -> None:
        self._i = index
        self._alpha = 1.0 - 0.5 ** (1.0 / halflife)
        self._beta = 1.0 - self._alpha
        self._value = 0.0
        self._seen = False

    def update(self, ts: int, values: Sequence[float]) -> float:
        x = values[self._i]
        if x != x:
            return NaN
        if self._seen:
            self._value = self._alpha * x + self._beta * self._value
        else:
            self._value = x
            self._seen = True
        return self._value

    def snapshot(self) -> Snapshot:
        payload = _empty_snapshot()
        payload.update(ewma=self._value, seen=self._seen)
        return payload

    def load(self, payload: Snapshot) -> None:
        self._value = float(payload["ewma"])
        self._seen = bool(payload["seen"])


# ----------------------------------------------------------------------
# Time windows
# ----------------------------------------------------------------------
class _TimeMeanState:
    __slots__ = ("_buf", "_duration", "_i", "_mean")

    def __init__(self, duration: int, index: int) -> None:
        self._duration = duration
        self._i = index
        self._buf: deque[tuple[int, float]] = deque()
        self._mean = 0.0

    def update(self, ts: int, values: Sequence[float]) -> float:
        x = values[self._i]
        if x != x:
            return NaN
        buf = self._buf
        buf.append((ts, x))
        self._mean += (x - self._mean) / len(buf)
        threshold = ts - self._duration
        while buf[0][0] < threshold:
            _, old = buf.popleft()
            if buf:
                self._mean -= (old - self._mean) / len(buf)
            else:
                self._mean = 0.0
        if len(buf) == 1:
            # A one-element window is exact by definition.
            self._mean = buf[0][1]
        return self._mean

    def snapshot(self) -> Snapshot:
        payload = _empty_snapshot()
        payload["n"] = len(self._buf)
        payload["mean"] = self._mean
        payload["times"] = [ts for ts, _ in self._buf]
        payload["xs"] = [x for _, x in self._buf]
        return payload

    def load(self, payload: Snapshot) -> None:
        times = [int(value) for value in payload["times"]]
        xs = [float(value) for value in payload["xs"]]
        self._buf = deque(zip(times, xs, strict=True))
        self._mean = float(payload["mean"])


class _TimeMomentState:
    __slots__ = ("_buf", "_duration", "_i", "_m2", "_mean", "_standardize")

    def __init__(self, duration: int, index: int, *, standardize: bool) -> None:
        self._duration = duration
        self._i = index
        self._buf: deque[tuple[int, float]] = deque()
        self._mean = 0.0
        self._m2 = 0.0
        self._standardize = standardize

    def update(self, ts: int, values: Sequence[float]) -> float:
        x = values[self._i]
        if x != x:
            return NaN
        buf = self._buf
        buf.append((ts, x))
        n = len(buf)
        delta = x - self._mean
        self._mean += delta / n
        self._m2 += delta * (x - self._mean)
        threshold = ts - self._duration
        while buf[0][0] < threshold:
            _, old = buf.popleft()
            n = len(buf)
            if n == 0:
                self._mean = 0.0
                self._m2 = 0.0
            else:
                delta = old - self._mean
                self._mean -= delta / n
                self._m2 -= delta * (old - self._mean)
        if n == 1:
            # A one-element window has zero dispersion and its mean is the
            # surviving value; resetting avoids carrying rounding residue.
            self._mean = buf[0][1]
            self._m2 = 0.0
        variance = self._m2 / n if n else 0.0
        if variance < 0.0:
            variance = 0.0
        if not self._standardize:
            return math.sqrt(variance)
        if n < 2:
            return NaN
        std = math.sqrt(variance)
        if std == 0.0:
            return NaN
        return (x - self._mean) / std

    def snapshot(self) -> Snapshot:
        payload = _empty_snapshot()
        payload["n"] = len(self._buf)
        payload["mean"] = self._mean
        payload["m2"] = self._m2
        payload["times"] = [ts for ts, _ in self._buf]
        payload["xs"] = [x for _, x in self._buf]
        return payload

    def load(self, payload: Snapshot) -> None:
        times = [int(value) for value in payload["times"]]
        xs = [float(value) for value in payload["xs"]]
        self._buf = deque(zip(times, xs, strict=True))
        self._mean = float(payload["mean"])
        self._m2 = float(payload["m2"])


class _TimeCumReturnState:
    __slots__ = ("_buf", "_duration", "_i")

    def __init__(self, duration: int, index: int) -> None:
        self._duration = duration
        self._i = index
        self._buf: deque[tuple[int, float]] = deque()

    def update(self, ts: int, values: Sequence[float]) -> float:
        x = values[self._i]
        if x != x:
            return NaN
        buf = self._buf
        buf.append((ts, x))
        threshold = ts - self._duration
        while buf[0][0] < threshold:
            buf.popleft()
        if len(buf) < 2:
            return NaN
        base = buf[0][1]
        if base == 0.0:
            return NaN
        return x / base - 1.0

    def snapshot(self) -> Snapshot:
        payload = _empty_snapshot()
        payload["n"] = len(self._buf)
        payload["times"] = [ts for ts, _ in self._buf]
        payload["xs"] = [x for _, x in self._buf]
        return payload

    def load(self, payload: Snapshot) -> None:
        times = [int(value) for value in payload["times"]]
        xs = [float(value) for value in payload["xs"]]
        self._buf = deque(zip(times, xs, strict=True))


class _TimeCorrState:
    __slots__ = ("_buf", "_c", "_duration", "_ix", "_iy", "_m2x", "_m2y", "_mx", "_my")

    def __init__(self, duration: int, index_x: int, index_y: int) -> None:
        self._duration = duration
        self._ix = index_x
        self._iy = index_y
        self._buf: deque[tuple[int, float, float]] = deque()
        self._mx = 0.0
        self._my = 0.0
        self._m2x = 0.0
        self._m2y = 0.0
        self._c = 0.0

    def update(self, ts: int, values: Sequence[float]) -> float:
        x = values[self._ix]
        y = values[self._iy]
        if x != x or y != y:
            return NaN
        buf = self._buf
        buf.append((ts, x, y))
        n = len(buf)
        dx = x - self._mx
        dy = y - self._my
        self._mx += dx / n
        self._my += dy / n
        self._m2x += dx * (x - self._mx)
        self._m2y += dy * (y - self._my)
        self._c += dx * (y - self._my)
        threshold = ts - self._duration
        while buf[0][0] < threshold:
            _, old_x, old_y = buf.popleft()
            n = len(buf)
            if n == 0:
                self._mx = 0.0
                self._my = 0.0
                self._m2x = 0.0
                self._m2y = 0.0
                self._c = 0.0
            else:
                dx = old_x - self._mx
                dy = old_y - self._my
                self._mx -= dx / n
                self._my -= dy / n
                self._m2x -= dx * (old_x - self._mx)
                self._m2y -= dy * (old_y - self._my)
                self._c -= dx * (old_y - self._my)
        if n == 1:
            _, self._mx, self._my = buf[0]
            self._m2x = 0.0
            self._m2y = 0.0
            self._c = 0.0
        if n < 2:
            return NaN
        if self._m2x <= 0.0 or self._m2y <= 0.0:
            return NaN
        if n == 2:
            return 1.0 if self._c > 0.0 else -1.0
        value = self._c / math.sqrt(self._m2x * self._m2y)
        if value > 1.0:
            return 1.0
        if value < -1.0:
            return -1.0
        return value

    def snapshot(self) -> Snapshot:
        payload = _empty_snapshot()
        payload["n"] = len(self._buf)
        payload.update(mean=self._mx, mean_y=self._my, m2=self._m2x, m2_y=self._m2y, c=self._c)
        payload["times"] = [ts for ts, _, _ in self._buf]
        payload["xs"] = [x for _, x, _ in self._buf]
        payload["ys"] = [y for _, _, y in self._buf]
        return payload

    def load(self, payload: Snapshot) -> None:
        times = [int(value) for value in payload["times"]]
        xs = [float(value) for value in payload["xs"]]
        ys = [float(value) for value in payload["ys"]]
        self._buf = deque(zip(times, xs, ys, strict=True))
        self._mx = float(payload["mean"])
        self._my = float(payload["mean_y"])
        self._m2x = float(payload["m2"])
        self._m2y = float(payload["m2_y"])
        self._c = float(payload["c"])


def make_state(spec: FeatureSpec, columns: Sequence[str]) -> _State:
    """Build a fresh state machine for ``spec`` over the given columns."""
    index = {column: position for position, column in enumerate(columns)}
    if spec.kind is FeatureKind.MEAN:
        if spec.is_time_window:
            return _TimeMeanState(spec.window_size, index[spec.inputs[0]])
        return _MeanState(spec.window_size, index[spec.inputs[0]])
    if spec.kind is FeatureKind.STD:
        if spec.is_time_window:
            return _TimeMomentState(spec.window_size, index[spec.inputs[0]], standardize=False)
        return _MomentState(spec.window_size, index[spec.inputs[0]], standardize=False)
    if spec.kind is FeatureKind.ZSCORE:
        if spec.is_time_window:
            return _TimeMomentState(spec.window_size, index[spec.inputs[0]], standardize=True)
        return _MomentState(spec.window_size, index[spec.inputs[0]], standardize=True)
    if spec.kind is FeatureKind.CUM_RETURN:
        if spec.is_time_window:
            return _TimeCumReturnState(spec.window_size, index[spec.inputs[0]])
        return _CumReturnState(spec.window_size, index[spec.inputs[0]])
    if spec.kind is FeatureKind.CORR:
        if spec.is_time_window:
            return _TimeCorrState(spec.window_size, index[spec.inputs[0]], index[spec.inputs[1]])
        return _CorrState(spec.window_size, index[spec.inputs[0]], index[spec.inputs[1]])
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

    When ``retain_records`` is true the engine keeps every accepted record and
    every output, which is what late-data corrections and feature backfills
    compare against. When it is false the engine only keeps the state
    machines and the counters; the store rebuilds from the raw log when it
    needs the history. Trimming a retained engine has the same effect.
    """

    __slots__ = (
        "columns",
        "keys",
        "last_key",
        "max_ts",
        "next_seq",
        "processed",
        "records",
        "retain_records",
        "series",
        "slots",
    )

    def __init__(self, series: str, columns: Sequence[str], *, retain_records: bool = True) -> None:
        self.series = series
        self.columns = tuple(columns)
        self.keys: list[tuple[int, int]] = []
        self.records: list[tuple[int, int, tuple[float, ...]]] = []
        self.slots: list[_Slot] = []
        self.max_ts: int | None = None
        self.next_seq = 0
        self.processed = 0
        self.retain_records = retain_records
        self.last_key: tuple[int, int] | None = None

    @property
    def has_full_history(self) -> bool:
        """Whether ``records`` covers every record ever appended."""
        return self.retain_records and self.processed == len(self.records)

    def __len__(self) -> int:
        return self.processed

    def add_feature(self, spec: FeatureSpec) -> list[float]:
        """Attach a feature and backfill it over the records already seen.

        Requires the full history; the store backfills from the raw log when
        the engine has been trimmed.
        """
        if self.processed != len(self.records):
            raise RuntimeError("cannot backfill from a trimmed engine")
        slot = _Slot(spec, make_state(spec, self.columns))
        for ts, _, values in self.records:
            slot.values.append(slot.state.update(ts, values))
        self.slots.append(slot)
        return slot.values

    def attach(self, spec: FeatureSpec, state: _State) -> None:
        """Attach a feature whose state is already at the current record."""
        self.slots.append(_Slot(spec, state))

    def append(self, ts: int, seq: int, values: tuple[float, ...]) -> list[float]:
        """Append a record whose key is greater than every key seen so far."""
        self.processed += 1
        if self.retain_records:
            self.keys.append((ts, seq))
            self.records.append((ts, seq, values))
        if self.max_ts is None or ts > self.max_ts:
            self.max_ts = ts
        if seq >= self.next_seq:
            self.next_seq = seq + 1
        key = (ts, seq)
        if self.last_key is None or key > self.last_key:
            self.last_key = key
        output: list[float] = []
        for slot in self.slots:
            value = slot.state.update(ts, values)
            if self.retain_records:
                slot.values.append(value)
            output.append(value)
        return output

    def insert(
        self, position: int, ts: int, seq: int, values: tuple[float, ...]
    ) -> list[tuple[str, int, float, float]]:
        """Insert a record at ``position`` and replay the series.

        Only valid when the full history is retained.
        """
        if not self.has_full_history:
            raise RuntimeError("cannot insert into a trimmed engine")
        previous = [slot.values for slot in self.slots]
        self.keys.insert(position, (ts, seq))
        self.records.insert(position, (ts, seq, values))
        self.processed += 1
        if self.max_ts is None or ts > self.max_ts:
            self.max_ts = ts
        if seq >= self.next_seq:
            self.next_seq = seq + 1
        key = (ts, seq)
        if self.last_key is None or key > self.last_key:
            self.last_key = key
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
        if not self.has_full_history:
            raise RuntimeError("cannot replay a trimmed engine")
        rebuilt = [_Slot(slot.spec, make_state(slot.spec, self.columns)) for slot in self.slots]
        for ts, _, values in self.records:
            for slot in rebuilt:
                slot.values.append(slot.state.update(ts, values))
        for slot, fresh in zip(self.slots, rebuilt, strict=True):
            slot.state = fresh.state
            slot.values = fresh.values

    def trim(self) -> None:
        """Drop the retained records and outputs, keeping the state machines."""
        self.records.clear()
        self.keys.clear()
        for slot in self.slots:
            slot.values.clear()

    def snapshot(self) -> list[tuple[str, Snapshot]]:
        """Exact serialisable copy of every feature state."""
        return [(slot.spec.name, slot.state.snapshot()) for slot in self.slots]
