"""Incremental, point-in-time-correct financial features.

Public surface::

    from feature_store import FeatureSpec, FeatureStore, compute_batch

    store = FeatureStore.open("data")
    store.register(FeatureSpec.mean("sma_20", window=20))
    store.push("BTC", ts=1_700_000_000_000_000_000, value=100.0)
    store.get("sma_20", "BTC")
"""

from __future__ import annotations

from .batch import Tick, compute_batch
from .errors import (
    DuplicateFeatureError,
    DuplicateRecordError,
    FeatureStoreError,
    UnknownColumnError,
    UnknownFeatureError,
)
from .spec import FeatureKind, FeatureSpec, LatePolicy
from .storage import (
    DroppedRecord,
    FeatureRow,
    MemoryStorage,
    NullStorage,
    ParquetStorage,
)
from .store import FeatureStore, IngestResult, StoreStats

__version__ = "0.1.0"

__all__ = [
    "DroppedRecord",
    "DuplicateFeatureError",
    "DuplicateRecordError",
    "FeatureKind",
    "FeatureRow",
    "FeatureSpec",
    "FeatureStore",
    "FeatureStoreError",
    "IngestResult",
    "LatePolicy",
    "MemoryStorage",
    "NullStorage",
    "ParquetStorage",
    "StoreStats",
    "Tick",
    "UnknownColumnError",
    "UnknownFeatureError",
    "__version__",
    "compute_batch",
]
