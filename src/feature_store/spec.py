"""Declarative feature definitions.

A :class:`FeatureSpec` is a frozen value object: it holds the name, the kind,
the window parameters and the input columns, and validates them on creation.
Nothing in this module computes anything. The arithmetic lives in
:mod:`feature_store.engine` and is documented in ``docs/design.md``.

A windowed feature uses either a count window (``window=20``, the last 20
observations) or a time window (``time_window=60_000_000_000``, every
observation with a timestamp in ``[t - time_window, t]``). Exactly one of the
two must be given.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class FeatureKind(StrEnum):
    """Kinds of feature this library can compute."""

    MEAN = "mean"
    STD = "std"
    ZSCORE = "zscore"
    CUM_RETURN = "cum_return"
    CORR = "corr"
    EWMA = "ewma"

    @property
    def is_windowed(self) -> bool:
        """Whether the feature needs a window over the trailing observations."""
        return self is not FeatureKind.EWMA


class LatePolicy(StrEnum):
    """What to do with a record that arrives behind the watermark."""

    DROP = "drop"
    """Discard the record and write an audit entry. The feature values are unchanged."""

    RECOMPUTE = "recompute"
    """Insert the record in event-time order and recompute the affected tail."""


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Immutable, serialisable definition of one feature.

    Parameters
    ----------
    name:
        Unique feature name. Letters, digits and underscores, must not start
        with a digit.
    kind:
        Which statistic to compute.
    window:
        Number of trailing observations. Mutually exclusive with
        ``time_window``; required for every kind except ``EWMA``.
    time_window:
        Width of the trailing time window, in the same integer units as the
        record timestamps. Mutually exclusive with ``window``.
    halflife:
        Decay parameter in ticks for ``EWMA``, in ``(0, +inf)``. Required for
        ``EWMA`` and forbidden otherwise.
    inputs:
        Columns consumed by the feature. One column for every kind except
        ``CORR``, which takes two. Defaults to the conventional columns
        ``("x",)`` and ``("x", "y")``.
    description:
        Free-form note, stored in the catalogue.
    """

    name: str
    kind: FeatureKind
    window: int | None = None
    time_window: int | None = None
    halflife: float | None = None
    inputs: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FeatureKind):
            object.__setattr__(self, "kind", FeatureKind(self.kind))
        if not isinstance(self.inputs, tuple):
            object.__setattr__(self, "inputs", tuple(self.inputs))
        if not _NAME_PATTERN.match(self.name):
            raise ValueError(
                f"invalid feature name {self.name!r}: use letters, digits and underscores, "
                "starting with a letter or underscore, at most 64 characters"
            )
        if self.kind.is_windowed:
            has_count = self.window is not None
            has_time = self.time_window is not None
            if has_count == has_time:
                raise ValueError(f"{self.kind.value} requires exactly one of window or time_window")
            if has_count:
                if not isinstance(self.window, int) or isinstance(self.window, bool):
                    raise ValueError("window must be an integer")
                if self.window < 1:
                    raise ValueError(f"window must be >= 1, got {self.window}")
            if has_time:
                if not isinstance(self.time_window, int) or isinstance(self.time_window, bool):
                    raise ValueError("time_window must be an integer")
                if self.time_window < 1:
                    raise ValueError(f"time_window must be >= 1, got {self.time_window}")
            if self.halflife is not None:
                raise ValueError("halflife is only valid for the EWMA kind")
        else:
            if self.window is not None or self.time_window is not None:
                raise ValueError("EWMA does not take a window")
            if self.halflife is None or not math.isfinite(self.halflife) or self.halflife <= 0:
                raise ValueError("EWMA requires a finite halflife > 0")

        if self.kind is FeatureKind.CORR:
            if len(self.inputs) != 2:
                raise ValueError("corr requires exactly two input columns")
            if self.inputs[0] == self.inputs[1]:
                raise ValueError("corr input columns must be different")
        elif len(self.inputs) != 1:
            raise ValueError(f"{self.kind.value} requires exactly one input column")
        if not all(isinstance(column, str) and column for column in self.inputs):
            raise ValueError("input column names must be non-empty strings")

    @property
    def is_time_window(self) -> bool:
        """Whether the window is measured in timestamp units rather than ticks."""
        return self.time_window is not None

    @property
    def window_size(self) -> int:
        """The count or the time width, whichever the feature uses.

        Raises for ``EWMA``, which has no window.
        """
        if self.window is not None:
            return self.window
        if self.time_window is not None:
            return self.time_window
        raise ValueError(f"{self.kind.value} has no window")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def mean(
        cls,
        name: str,
        window: int | None = None,
        *,
        time_window: int | None = None,
        input: str = "x",
        description: str = "",
    ) -> FeatureSpec:
        """Rolling arithmetic mean over the last ``window`` observations or
        over the last ``time_window`` timestamp units."""
        return cls(
            name,
            FeatureKind.MEAN,
            window=window,
            time_window=time_window,
            inputs=(input,),
            description=description,
        )

    @classmethod
    def std(
        cls,
        name: str,
        window: int | None = None,
        *,
        time_window: int | None = None,
        input: str = "x",
        description: str = "",
    ) -> FeatureSpec:
        """Rolling population standard deviation (``ddof=0``)."""
        return cls(
            name,
            FeatureKind.STD,
            window=window,
            time_window=time_window,
            inputs=(input,),
            description=description,
        )

    @classmethod
    def zscore(
        cls,
        name: str,
        window: int | None = None,
        *,
        time_window: int | None = None,
        input: str = "x",
        description: str = "",
    ) -> FeatureSpec:
        """Rolling z-score of the incoming value: ``(x - mean) / std``."""
        return cls(
            name,
            FeatureKind.ZSCORE,
            window=window,
            time_window=time_window,
            inputs=(input,),
            description=description,
        )

    @classmethod
    def cum_return(
        cls,
        name: str,
        window: int | None = None,
        *,
        time_window: int | None = None,
        input: str = "x",
        description: str = "",
    ) -> FeatureSpec:
        """Return from the oldest observation in the window to the newest."""
        return cls(
            name,
            FeatureKind.CUM_RETURN,
            window=window,
            time_window=time_window,
            inputs=(input,),
            description=description,
        )

    @classmethod
    def corr(
        cls,
        name: str,
        window: int | None = None,
        *,
        time_window: int | None = None,
        x: str = "x",
        y: str = "y",
        description: str = "",
    ) -> FeatureSpec:
        """Rolling Pearson correlation between the columns ``x`` and ``y``."""
        return cls(
            name,
            FeatureKind.CORR,
            window=window,
            time_window=time_window,
            inputs=(x, y),
            description=description,
        )

    @classmethod
    def ewma(
        cls, name: str, halflife: float, *, input: str = "x", description: str = ""
    ) -> FeatureSpec:
        """Exponentially weighted moving average with the given half-life."""
        return cls(
            name,
            FeatureKind.EWMA,
            halflife=float(halflife),
            inputs=(input,),
            description=description,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible representation."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "window": self.window,
            "time_window": self.time_window,
            "halflife": self.halflife,
            "inputs": list(self.inputs),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeatureSpec:
        """Rebuild a spec from :meth:`to_dict` output.

        Catalogues written before time windows existed have no
        ``time_window`` key; they are read as count-window features.
        """
        return cls(
            name=str(payload["name"]),
            kind=FeatureKind(payload["kind"]),
            window=payload.get("window"),
            time_window=payload.get("time_window"),
            halflife=payload.get("halflife"),
            inputs=tuple(payload.get("inputs", ())),
            description=str(payload.get("description", "")),
        )

    def __str__(self) -> str:
        if self.kind is FeatureKind.EWMA:
            parameters = f"halflife={self.halflife:g}"
        elif self.time_window is not None:
            parameters = f"time_window={self.time_window}"
        else:
            parameters = f"window={self.window}"
        return f"{self.name}[{self.kind.value}, {parameters}, inputs={','.join(self.inputs)}]"
