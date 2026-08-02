# Global configuration constants and thresholds used across the pipeline.
# Adjust these to tune sensitivity and storage paths.

import os

# Windowing granularity (use "15min" for detailed troubleshooting, else "60min" for daily)
WINDOW_SIZE = "60min"

# Baseline history window used for z-score calculations (days)
BASELINE_DAYS = 28

# Anomaly thresholds (starting points; tune per metric)
Z_THRESH = 2.0            # z-score threshold for spike detection
SATURATION_P95 = 85.0     # p95 threshold for percent metrics (e.g., CPU%)
LEVEL_SHIFT_K_SIGMA = 1.0 # change in mean vs sigma for regime shift

# Chroma storage (persistent) — use absolute path so ingest and Streamlit use the same store
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(_PROJECT_DIR, "chroma_store")
CHROMA_COLLECTION = "observability_docs"

# Embedding model (change to e.g., 'intfloat/e5-large' if you want)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384   # MiniLM-L6-v2 dimension

# Gemini LLM for RAG answer generation (set GOOGLE_API_KEY in env or .env)
# Use a current model: gemini-2.5-flash (stable), gemini-2.5-pro, or gemini-3-flash-preview
GEMINI_MODEL = "gemini-2.5-flash"

# Known SLO configs (optional; provide as needed)
# Format: {(service, metric_name): {"field": "p95", "op": "<=", "target": 300, "unit": "ms"}}
SLO_CONFIGS = {
    ("checkout-api", "p95_latency_ms"): {"field": "p95", "op": "<=", "target": 300, "unit": "ms"},
    ("checkout-api", "error_rate_percent"): {"field": "p95", "op": "<=", "target": 1.0, "unit": "%"},
}
