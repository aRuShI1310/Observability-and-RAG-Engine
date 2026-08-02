
# Reads a large, intraday metrics CSV (exported daily from Vertica)
# in memory-efficient chunks, normalizes types, and returns a dataframe
# ready for windowing and eventization.
#
# Expected CSV columns (adapt as needed):
# - timestamp (ISO8601 UTC)
# - service, environment, region
# - metric_name, metric_value, unit
# - labels (optional string; 'key=value;key2=value2')
# - source_csv (optional; we add if missing)

import os
import pandas as pd
import numpy as np

DTYPES = {
    "service": "category",
    "environment": "category",
    "region": "category",
    "metric_name": "category",
    "unit": "category",
    "labels": "string"
}

def read_csv_chunked(path: str, chunksize: int = 1_000_000) -> pd.DataFrame:
    base = os.path.basename(path)
    frames = []
    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=DTYPES):
        # Normalize timestamp to UTC and ensure numeric value
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce")
        chunk = chunk.dropna(subset=["timestamp"])
        chunk["metric_value"] = pd.to_numeric(chunk["metric_value"], errors="coerce")
        chunk = chunk.dropna(subset=["metric_value"])
        # Derive date for grouping/filters
        chunk["date"] = chunk["timestamp"].dt.date
        # Fill NA labels so groupby in windowing produces groups
        if "labels" in chunk.columns:
            chunk["labels"] = chunk["labels"].fillna("")
        # Add provenance if not present
        if "source_csv" not in chunk.columns:
            chunk["source_csv"] = base
        # Ensure labels is never NA so groupby in windowing yields groups
        if "labels" in chunk.columns:
            chunk["labels"] = chunk["labels"].fillna("").astype("string")
        frames.append(chunk)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
