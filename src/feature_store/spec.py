"""Declarative feature definitions.

A :class:`FeatureSpec` is a frozen value object: it holds the name, the kind,
the window parameters and the input columns, and validates them on creation.
Nothing in this module computes anything. The arithmetic lives in
:mod:`feature_store.engine` and is documented in ``docs/design.md``.
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
        """Whether the feature needs a count window over the trailing ticks."""
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
        Number of trailing observations. Required for every kind except
        ``EWMA``, forbidden for it.
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
            if not isinstance(self.window, int) or isinstance(self.window, bool):
                raise ValueError(f"{self.kind.value} requires an integer window")
            if self.window < 1:
                raise ValueError(f"window must be >= 1, got {self.window}")
            if self.halflife is not None:
                raise ValueError("halflife is only valid for the EWMA kind")
        else:
            if self.window is not None:
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

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def mean(
        cls, name: str, window: int, *, input: str = "x", description: str = ""
    ) -> FeatureSpec:
        """Rolling arithmetic mean over the last ``window`` observations."""
        return cls(name, FeatureKind.MEAN, window=window, inputs=(input,), description=description)

    @classmethod
    def std(cls, name: str, window: int, *, input: str = "x", description: str = "") -> FeatureSpec:
        """Rolling population standard deviation (``ddof=0``)."""
        return cls(name, FeatureKind.STD, window=window, inputs=(input,), description=description)

    @classmethod
    def zscore(
        cls, name: str, window: int, *, input: str = "x", description: str = ""
    ) -> FeatureSpec:
        """Rolling z-score of the incoming value: ``(x - mean) / std``."""
        return cls(
            name, FeatureKind.ZSCORE, window=window, inputs=(input,), description=description
        )

    @classmethod
    def cum_return(
        cls, name: str, window: int, *, input: str = "x", description: str = ""
    ) -> FeatureSpec:
        """Return from the oldest observation in the window to the newest."""
        return cls(
            name, FeatureKind.CUM_RETURN, window=window, inputs=(input,), description=description
        )

    @classmethod
    def corr(
        cls,
        name: str,
        window: int,
        *,
        x: str = "x",
        y: str = "y",
        description: str = "",
    ) -> FeatureSpec:
        """Rolling Pearson correlation between the columns ``x`` and ``y``."""
        return cls(name, FeatureKind.CORR, window=window, inputs=(x, y), description=description)

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
            "halflife": self.halflife,
            "inputs": list(self.inputs),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeatureSpec:
        """Rebuild a spec from :meth:`to_dict` output."""
        return cls(
            name=str(payload["name"]),
            kind=FeatureKind(payload["kind"]),
            window=payload.get("window"),
            halflife=payload.get("halflife"),
            inputs=tuple(payload.get("inputs", ())),
            description=str(payload.get("description", "")),
        )

    def __str__(self) -> str:
        if self.kind is FeatureKind.EWMA:
            parameters = f"halflife={self.halflife:g}"
        else:
            parameters = f"window={self.window}"
        return f"{self.name}[{self.kind.value}, {parameters}, inputs={','.join(self.inputs)}]"
