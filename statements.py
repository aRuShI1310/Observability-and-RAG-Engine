# Build compact, human-readable text for both document types:
# - Anomaly/Event cards (window-level; only for interesting windows)
# - Daily Summary cards (one per metric/day)
# These are the strings that get embedded into vectors.

def labels_str(x: str) -> str:
    return x if isinstance(x, str) and x.strip() else "none"

def anomaly_card(row: dict) -> str:
    return (
        f"[Anomaly] {row['service']} {row['metric_name']} "
        f"({row['environment']}/{row['region']}) on {row['date']} "
        f"{row['win_start']}–{row['win_end']} UTC\n"
        f"avg: {row['avg']:.2f}{row['unit']} p95: {row['p95']:.2f}{row['unit']} max: {row['max']:.2f}{row['unit']}\n"
        f"Pattern: {row['trend']}, volatility: {row['volatility']:.2f}; z={row.get('zscore', 0):.2f}\n"
        f"Sustained: {'yes' if row.get('is_sustained', False) else 'no'} | "
        f"Level shift: {'yes' if row.get('is_level_shift', False) else 'no'} | "
        f"SLO breach: {'yes' if row.get('slo_breached', False) else 'no'}\n"
        f"Labels: {labels_str(row.get('labels',''))}\n"
        f"Provenance: {row.get('source_csv','')}"
    )

def daily_summary_card(row: dict) -> str:
    return (
        f"[Daily Summary] {row['service']} {row['metric_name']} "
        f"({row['environment']}/{row['region']}) on {row['date']}\n"
        f"mean: {row['mean']:.2f}{row['unit']} p50: {row['p50']:.2f}{row['unit']} "
        f"p95: {row['p95']:.2f}{row['unit']} max: {row['max']:.2f}{row['unit']}\n"
        f"DoD: {100*row['d1_pct']:+.1f}% WoW: {100*row['d7_pct']:+.1f}% z(28d)={row.get('zscore_28d',0):.2f}\n"
        f"Trend: {row['trend']}; Level shift: no\n"
        f"Labels: {labels_str(row.get('labels',''))}\n"
        f"Provenance: {row.get('source_csv','')}"
    )
