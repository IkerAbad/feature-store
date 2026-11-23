"""Exact engine checkpoints.

A checkpoint is one immutable directory with two files:

``state.parquet``
    One row per ``(series, feature)`` with the exact state-machine fields and
    the window contents. Float64 values keep their IEEE-754 bits, so a restored
    engine continues with the same arithmetic.

``manifest.json``
    The feature catalogue hash, the last key and record count of every series,
    and the raw files that were already consumed when the checkpoint was taken.

``state/LATEST`` points to the newest checkpoint. A new checkpoint is written
to a temporary directory and renamed into place; ``LATEST`` is replaced
atomically, so a crash leaves either the previous checkpoint or the new one,
never a half-written mix.

The checkpoint schema is versioned separately from the feature table. Feature
rows keep their original schema, so enabling checkpoints does not migrate or
rewrite any data.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

FORMAT = 1

_STATE_SCHEMA = pa.schema(
    [
        pa.field("series", pa.string(), nullable=False),
        pa.field("feature", pa.string(), nullable=False),
        pa.field("n", pa.int64(), nullable=False),
        pa.field("mean", pa.float64(), nullable=False),
        pa.field("m2", pa.float64(), nullable=False),
        pa.field("ewma", pa.float64(), nullable=False),
        pa.field("seen", pa.bool_(), nullable=False),
        pa.field("mean_y", pa.float64(), nullable=False),
        pa.field("m2_y", pa.float64(), nullable=False),
        pa.field("c", pa.float64(), nullable=False),
        pa.field("times", pa.list_(pa.int64()), nullable=False),
        pa.field("xs", pa.list_(pa.float64()), nullable=False),
        pa.field("ys", pa.list_(pa.float64()), nullable=False),
    ]
)


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A loaded checkpoint."""

    id: str
    path: Path
    manifest: Mapping[str, Any]
    rows: Sequence[Mapping[str, Any]]

    @property
    def raw_files(self) -> frozenset[str]:
        """Relative POSIX paths of the raw files already replayed."""
        return frozenset(str(item) for item in self.manifest.get("raw_files", ()))

    @property
    def series(self) -> Mapping[str, Mapping[str, Any]]:
        value = self.manifest.get("series", {})
        return value if isinstance(value, dict) else {}


def checkpoint_root(root: Path) -> Path:
    return root / "state"


def write_checkpoint(
    root: Path,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Checkpoint:
    """Persist a new checkpoint and make it the latest one."""
    state_dir = checkpoint_root(root)
    state_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_id = f"{time.time_ns():019d}"
    temporary = state_dir / f"checkpoint-{checkpoint_id}.tmp"
    final = state_dir / f"checkpoint-{checkpoint_id}"
    if final.exists():  # pragma: no cover - clock resolution collision
        checkpoint_id = f"{time.time_ns():019d}-1"
        temporary = state_dir / f"checkpoint-{checkpoint_id}.tmp"
        final = state_dir / f"checkpoint-{checkpoint_id}"
    temporary.mkdir()
    payload = dict(manifest)
    payload["format"] = FORMAT
    payload["id"] = checkpoint_id
    table = pa.Table.from_pylist(list(rows), schema=_STATE_SCHEMA)
    pq.write_table(table, temporary / "state.parquet", compression="zstd")
    (temporary / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, final)
    latest = state_dir / "LATEST.tmp"
    latest.write_text(checkpoint_id + "\n", encoding="utf-8")
    os.replace(latest, state_dir / "LATEST")
    return Checkpoint(
        id=checkpoint_id,
        path=final,
        manifest=payload,
        rows=tuple(rows),
    )


def read_latest(root: Path) -> Checkpoint | None:
    """Load the newest checkpoint, or ``None`` when the store has none."""
    state_dir = checkpoint_root(root)
    pointer = state_dir / "LATEST"
    if not pointer.exists():
        return None
    checkpoint_id = pointer.read_text(encoding="utf-8").strip()
    path = state_dir / f"checkpoint-{checkpoint_id}"
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():  # pragma: no cover - corrupted pointer
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("format", 0)) != FORMAT:
        return None
    table = pq.read_table(path / "state.parquet")
    return Checkpoint(
        id=checkpoint_id,
        path=path,
        manifest=manifest,
        rows=tuple(table.to_pylist()),
    )


def prune_checkpoints(root: Path, keep: int = 2) -> list[Path]:
    """Delete all but the ``keep`` newest checkpoints."""
    state_dir = checkpoint_root(root)
    if not state_dir.exists():
        return []
    directories = sorted(
        (path for path in state_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    removed: list[Path] = []
    for directory in directories[keep:]:
        shutil.rmtree(directory, ignore_errors=True)
        removed.append(directory)
    return removed
