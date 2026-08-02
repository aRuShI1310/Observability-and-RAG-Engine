
# Generates one summary per (service, env, region, metric_name, date)
# with mean/p50/p95/max and day-over-day/week-over-week changes.
# Also computes a rolling zscore vs last 28 days of daily means.

import numpy as np
import pandas as pd

KEY = ["service","environment","region","metric_name"]

def daily_rollups_from_intraday(intraday_df: pd.DataFrame) -> pd.DataFrame:
    grp = ["date"] + KEY + ["unit","labels"]
    daily = intraday_df.groupby(grp)["metric_value"].agg(
        mean="mean",
        p50=lambda s: s.quantile(0.50),
        p95=lambda s: s.quantile(0.95),
        max="max",
    ).reset_index()

    # Sort for pct_change
    daily = daily.sort_values(KEY + ["date"])
    daily["d1_pct"] = daily.groupby(KEY)["mean"].pct_change(1).fillna(0.0)
    daily["d7_pct"] = daily.groupby(KEY)["mean"].pct_change(7).fillna(0.0)

    # Trend label based on mean change vs yesterday
    delta = daily.groupby(KEY)["mean"].diff().fillna(0.0)
    daily["trend"] = np.sign(delta).map({-1:"down", 0:"flat", 1:"up"})

    # z(28d) vs rolling mean/std per metric
    def zscore_group(g):
        g["roll_mean"] = g["mean"].rolling(28, min_periods=7).mean()
        g["roll_std"]  = g["mean"].rolling(28, min_periods=7).std()
        eps = 1e-9
        g["zscore_28d"] = (g["mean"] - g["roll_mean"]) / g["roll_std"].replace(0, eps)
        g["zscore_28d"] = g["zscore_28d"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return g

    daily = daily.groupby(KEY, group_keys=False).apply(zscore_group)
    return daily
