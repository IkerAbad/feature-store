"""Exceptions raised by the feature store."""

from __future__ import annotations


class FeatureStoreError(Exception):
    """Base class for every error raised by this package."""


class DuplicateFeatureError(FeatureStoreError):
    """A different specification was registered under an existing name."""


class UnknownFeatureError(FeatureStoreError):
    """The feature was never registered."""


class UnknownColumnError(FeatureStoreError):
    """A feature or record references a column the store does not have."""


class DuplicateRecordError(FeatureStoreError):
    """A record with the same ``(ts, seq)`` already exists in the series."""
