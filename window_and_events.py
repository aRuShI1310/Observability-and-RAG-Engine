
# Resamples intraday samples into fixed windows (e.g., hourly),
# computes statistical features, and detects anomalies:
# - Spike/anomaly via z-score vs baseline
# - Sustained saturation via p95 threshold (percent metrics)
# - Level shift via mean change across recent vs previous windows
# - Optional SLO breach if configs are defined

import numpy as np
import pandas as pd
from config import WINDOW_SIZE, Z_THRESH, SATURATION_P95, LEVEL_SHIFT_K_SIGMA, SLO_CONFIGS

GROUP_KEYS = ["service","environment","region","metric_name","unit","labels"]

def resample_to_windows(df: pd.DataFrame) -> pd.DataFrame:
    # Group by entity+metric and resample 'metric_value' into fixed windows.
    # Produces per-window stats and trend/volatility markers.
    results = []
    for keys, g in df.groupby(GROUP_KEYS, sort=False, observed=True):
        g = g.sort_values("timestamp").set_index("timestamp")
        # Resample & aggregate stats
        agg = g["metric_value"].resample(WINDOW_SIZE).agg(["min","mean","max"])
        p50 = g["metric_value"].resample(WINDOW_SIZE).quantile(0.50)
        p95 = g["metric_value"].resample(WINDOW_SIZE).quantile(0.95)
        std = g["metric_value"].resample(WINDOW_SIZE).std()

        out = pd.DataFrame({
            "win_start": agg.index,
            "min": agg["min"].values,
            "avg": agg["mean"].values,
            "max": agg["max"].values,
            "p50": p50.values,
            "p95": p95.values,
            "volatility": std.values
        })
        out["win_end"] = out["win_start"] + pd.to_timedelta(WINDOW_SIZE)

        # Simple trend label based on change in avg vs previous window
        out["trend"] = np.sign(out["avg"].diff().fillna(0.0)).map({-1:"down",0:"flat",1:"up"})

        # Attach keys
        for k, v in zip(GROUP_KEYS, keys):
            out[k] = v
        # Date (for filter convenience)
        out["date"] = out["win_start"].dt.date

        results.append(out)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def compute_baseline(history_windows: pd.DataFrame) -> pd.DataFrame:
    # Computes baseline mean/std per (service, env, region, metric_name)
    # from historical window summaries. This snapshot is used
    # to compute z-scores for today's windows.
    key_cols = ["service","environment","region","metric_name"]
    stats = history_windows.groupby(key_cols)["avg"].agg(["mean","std"]).reset_index()
    stats.rename(columns={"mean":"baseline_mean","std":"baseline_std"}, inplace=True)
    return stats

def apply_anomaly_rules(win_df: pd.DataFrame, baseline_stats: pd.DataFrame) -> pd.DataFrame:
    # Joins windows with baseline mean/std, computes z-score,
    # saturation, level shift, and SLO breach flags.
    df = win_df.merge(baseline_stats, on=["service","environment","region","metric_name"], how="left")

    # z-score (spikes), safe epsilon for zero-std
    eps = 1e-9
    df["baseline_std"] = df["baseline_std"].fillna(0.0)
    df["zscore"] = (df["avg"] - df["baseline_mean"]) / df["baseline_std"].replace(0, eps)
    df["zscore"] = df["zscore"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["is_anomaly"] = df["zscore"].abs() >= Z_THRESH

    # sustained saturation (percent metrics)
    df["is_sustained"] = df.apply(
        lambda r: (str(r["metric_name"]).endswith("_percent") and (r["p95"] or 0) >= SATURATION_P95),
        axis=1
    )

    # simple level shift: compare last 3 windows vs previous 3 windows per day+metric
    def mark_level_shift(group: pd.DataFrame) -> pd.DataFrame:
        N = 3
        group = group.sort_values("win_start")
        if len(group) >= 2*N:
            mu_prev = group["avg"].iloc[-2*N:-N].mean()
            mu_recent = group["avg"].iloc[-N:].mean()
            sigma = group["avg"].std() or 1.0
            group.loc[group.index[-1], "is_level_shift"] = abs(mu_recent - mu_prev) >= LEVEL_SHIFT_K_SIGMA * sigma
        else:
            group["is_level_shift"] = False
        return group

    df = df.groupby(["date","service","environment","region","metric_name"], group_keys=False).apply(mark_level_shift)

    # SLO breach (if configured)
    def slo_breach(row) -> bool:
        key = (row["service"], row["metric_name"]) 
        cfg = SLO_CONFIGS.get(key)
        if not cfg: return False
        field = cfg.get("field", "p95")
        target = cfg["target"]
        op = cfg.get("op", "<=")
        actual = row.get(field, None)
        if actual is None: return False
        if op == "<=":
            return actual > target
        elif op == ">=":
            return actual < target
        return False

    df["slo_breached"] = df.apply(slo_breach, axis=1)
    return df
