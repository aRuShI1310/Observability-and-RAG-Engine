
# Daily job orchestrator:
# 1) Load intraday CSV (chunked)
# 2) Build window summaries
# 3) Compute baseline and eventization
# 4) Generate daily rollups
# 5) Create text statements (anomaly + summary)
# 6) Embed and upsert into Chroma with metadata

import os
import argparse
import pandas as pd
from ingest_csv import read_csv_chunked
from window_and_events import resample_to_windows, compute_baseline, apply_anomaly_rules
from daily_rollups import daily_rollups_from_intraday
from statements import anomaly_card, daily_summary_card
from embeddings import Embedder
from store_chroma import get_chroma_collection, upsert_docs
from config import CHROMA_COLLECTION

def run_daily_ingest(csv_path: str, history_windows_parquet_dir: str = None):
    # 1) Load
    intraday = read_csv_chunked(csv_path)
    if intraday.empty:
        print("No data ingested.")
        return

    # 2) Window summaries for today
    win_today = resample_to_windows(intraday)
    if win_today.empty:
        print("No windowed data.")
        return

    # Load history windows (optional) to compute baseline (last 28 days). If you persist them daily, load here.
    history = pd.DataFrame()
    if history_windows_parquet_dir and os.path.isdir(history_windows_parquet_dir):
        files = sorted([f for f in os.listdir(history_windows_parquet_dir) if f.endswith(".parquet")])[-28:]
        for f in files:
            dfh = pd.read_parquet(os.path.join(history_windows_parquet_dir, f))
            history = pd.concat([history, dfh], ignore_index=True)

    baseline = compute_baseline(history if not history.empty else win_today)  # fallback to today's distro if no history

    # 3) Eventization (mark windows with anomalies/sustained/slo_breach/level_shift)
    win_events = apply_anomaly_rules(win_today, baseline)

    # 4) Daily rollups
    daily = daily_rollups_from_intraday(intraday)
    daily = daily.drop_duplicates(
    subset=["service", "metric_name", "environment", "region", "date"]
)


    # 5) Build text statements + metadata + ids
    anomaly_rows = win_events[(win_events["is_anomaly"]) | (win_events["is_sustained"]) | (win_events["slo_breached"]) | (win_events["is_level_shift"])].copy()

    anomaly_texts, anomaly_ids, anomaly_meta = [], [], []
    for _, r in anomaly_rows.iterrows():
        text = anomaly_card(r)
        # Deterministic id ensures idempotent upserts
        uid = f"{r['service']}|{r['metric_name']}|{r['environment']}|{r['region']}|{r['win_start'].isoformat()}|anomaly"
        meta = {
            "type": "anomaly",
            "service": r["service"],
            "environment": r["environment"],
            "region": r["region"],
            "metric_name": r["metric_name"],
            "unit": r["unit"],
            "date": str(r["date"]),
            "win_start": r["win_start"].isoformat(),
            "win_end": r["win_end"].isoformat(),
            "labels": r.get("labels",""),
            "zscore": float(r.get("zscore",0.0)),
            "volatility": float(r.get("volatility",0.0)),
            "is_sustained": bool(r.get("is_sustained", False)),
            "is_level_shift": bool(r.get("is_level_shift", False)),
            "slo_breached": bool(r.get("slo_breached", False)),
            "source_csv": r.get("source_csv","")
        }
        anomaly_texts.append(text); anomaly_ids.append(uid); anomaly_meta.append(meta)

    summary_texts, summary_ids, summary_meta = [], [], []
    for _, r in daily.iterrows():
        text = daily_summary_card(r)
        uid = f"{r['service']}|{r['metric_name']}|{r['environment']}|{r['region']}|{r['date']}|summary"
        meta = {
            "type": "summary",
            "service": r["service"],
            "environment": r["environment"],
            "region": r["region"],
            "metric_name": r["metric_name"],
            "unit": r["unit"],
            "date": str(r["date"]),
            "labels": r.get("labels",""),
            "mean": float(r["mean"]),
            "p50": float(r["p50"]),
            "p95": float(r["p95"]),
            "max": float(r["max"]),
            "d1_pct": float(r.get("d1_pct", 0.0)),
            "d7_pct": float(r.get("d7_pct", 0.0)),
            "zscore_28d": float(r.get("zscore_28d", 0.0)),
            "trend": r.get("trend", "flat"),
            "source_csv": r.get("source_csv","")
        }
        summary_texts.append(text); summary_ids.append(uid); summary_meta.append(meta)

    # 6) Embed and upsert to Chroma
    embedder = Embedder()
    texts_all = anomaly_texts + summary_texts
    ids_all = anomaly_ids + summary_ids
    meta_all = anomaly_meta + summary_meta
    if len(texts_all) == 0:
        print("No anomalies or summaries produced → nothing to upsert.")
        return
    # --- FORCE UNIQUE IDS SAFETY ---
    unique_map = {}
    unique_texts = []
    unique_meta = []
    unique_ids = []

    for i, uid in enumerate(ids_all):
        if uid not in unique_map:
            unique_map[uid] = i
            unique_ids.append(uid)
            unique_texts.append(texts_all[i])
            unique_meta.append(meta_all[i])

    ids_all = unique_ids
    texts_all = unique_texts
    meta_all = unique_meta
# --------------------------------

    vecs_all = embedder.encode(texts_all)

    coll = get_chroma_collection()
    upsert_docs(coll, ids_all, texts_all, vecs_all, meta_all)
    print(f"Upserted anomalies={len(anomaly_ids)} summaries={len(summary_ids)} into Chroma collection '{CHROMA_COLLECTION}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily ingest for Observability RAG")
    parser.add_argument("--csv", required=True, help="Path to daily CSV export")
    parser.add_argument("--history", required=False, help="Dir with historical window parquet files (optional)")
    args = parser.parse_args()
    run_daily_ingest(args.csv, args.history)
