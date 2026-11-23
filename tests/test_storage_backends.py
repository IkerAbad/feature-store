"""The small storage backends, tested directly."""

from __future__ import annotations

from feature_store import FeatureSpec, FeatureStore
from feature_store.storage import DroppedRecord, FeatureRow, MemoryStorage, NullStorage, RawRecord


def test_null_storage_counts_and_returns_nothing() -> None:
    storage = NullStorage()
    row = FeatureRow("m", "S", 1, 0, 1.0, 1, "append", 0)
    storage.write_rows([row])
    storage.write_raw([])
    storage.write_dropped([])
    assert storage.rows_written == 1
    assert storage.raw_written == 0
    assert storage.dropped_written == 0
    assert storage.rows("m", "S") == []
    assert storage.partitions() == []
    assert list(storage.iter_raw()) == []
    storage.rewrite_partition("m", "S", [row])
    assert storage.read_raw() == []
    assert storage.read_dropped() == []
    storage.flush()
    storage.close()


def test_memory_storage_partitions_and_rewrite() -> None:
    storage = MemoryStorage()
    first = FeatureRow("m", "A", 1, 0, 1.0, 1, "append", 0)
    second = FeatureRow("m", "A", 1, 0, 2.0, 2, "append", 0)
    other = FeatureRow("m", "B", 1, 0, 3.0, 1, "append", 0)
    storage.write_rows([first, second, other])
    assert storage.partitions() == [("m", "A"), ("m", "B")]
    batches = list(storage.iter_raw())
    assert batches == []
    storage.write_raw([RawRecord("A", 1, 0, (1.0,), 0)])
    assert list(storage.iter_raw()) == [("A", [RawRecord("A", 1, 0, (1.0,), 0)])]
    storage.rewrite_partition("m", "A", [second])
    assert storage.rows("m", "A") == [second]
    assert storage.rows("m", "B") == [other]


def test_feature_store_with_a_null_backend_skips_writes() -> None:
    storage = NullStorage()
    store = FeatureStore(storage, columns=("x",))
    store.register(FeatureSpec.mean("m2", 2))
    store.push("S", 1, 1.0)
    store.push("S", 2, 3.0)
    assert storage.rows_written == 0
    assert store.get("m2", "S") is None
    assert store.stats().records == 2


def test_memory_storage_filters_rows() -> None:
    storage = MemoryStorage()
    row = FeatureRow("m", "A", 1, 0, 1.0, 1, "append", 0)
    other = FeatureRow("m", "B", 1, 0, 2.0, 1, "append", 0)
    storage.write_rows([row, other])
    assert storage.rows("m", "A") == [row]
    assert storage.rows("m", "missing") == []
    storage.write_raw([])
    storage.write_dropped([DroppedRecord("A", 0, 0, 0, "drop", 0)])
    assert len(storage.read_dropped()) == 1
    storage.flush()
    storage.close()


def test_feature_store_with_a_null_backend_still_computes() -> None:
    store = FeatureStore.null(columns=("x",))
    store.register(FeatureSpec.mean("m2", 2))
    store.push("S", 1, 1.0)
    store.push("S", 2, 3.0)
    assert store.get("m2", "S") is None
    assert store.stats().records == 2
    assert store.stats().rows_emitted == 0
