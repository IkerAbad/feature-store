"""Bit-exact helpers used by the engine, the store and the tests."""

from __future__ import annotations

import struct


def same_bits(left: float, right: float) -> bool:
    """Return ``True`` when two floats have the same IEEE-754 binary64 bits.

    NaNs compare equal when their payloads match, ``+0.0`` and ``-0.0`` do not.
    """
    return struct.pack("<d", left) == struct.pack("<d", right)
