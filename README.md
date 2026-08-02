# Observa-RAG (Chroma + LlamaIndex/LangChain + Gemini)

**Daily Observability metrics → Windowing → Eventization (anomalies, saturation, level-shifts, SLO) → Daily rollups → Embeddings → RAG → UI**

This repository lets you ingest a daily CSV of intraday metrics, turn raw signals into **semantic statements** (anomaly events + daily summaries), embed them in **Chroma**, and query them in natural language via a **Streamlit** UI. Optional **Gemini** LLM generates concise answers from retrieved context.

---

## ✨ Features

- **Chunked CSV ingest** and type normalization
- **Windowing** (15m/60m) and statistical features
- **Eventization**: z-score anomalies, saturation, level shift, SLO breach
- **Daily rollups** with DoD/WoW deltas and 28-day z-scores
- **Chroma** vector store with metadata filters
- **LlamaIndex** and **LangChain** RAG examples
- **Streamlit** UI: filters, semantic search, and **Gemini**-generated answers
- **Two document types**: `anomaly` (window-level events) and `summary` (daily rollups)

---

## 📋 Prerequisites

- **Python 3.9+** (3.10 or 3.11 recommended)
- **Git** (to clone the repo)
- **~2 GB disk** for the virtual environment and embedding model (first run downloads the model)
- **Google AI API key** (optional but recommended) for LLM-generated answers: [Get one here](https://aistudio.google.com/apikey)

---

## 📦 Project structure

```
observa-rag/
├── app_streamlit.py          # Streamlit UI (search + Gemini answers)
├── config.py                 # Thresholds, paths, Gemini model
├── daily_rollups.py          # Per-metric/day summaries
├── embeddings.py             # HuggingFace embeddings
├── ingest_csv.py             # Chunked CSV reader
├── rag_langchain.py         # LangChain RAG + Gemini chain
├── rag_llamaindex.py         # LlamaIndex query engine
├── run_daily_ingest.py       # Ingest → events → summaries → Chroma
├── statements.py             # Text templates for cards
├── store_chroma.py           # Chroma client helpers
├── window_and_events.py      # Windowing + anomaly detection
├── requirements.txt          # Python dependencies
├── .env                      # Your GOOGLE_API_KEY (create from .env.example; not committed)
└── data/
    └── daily_csv/            # Put your daily CSVs here
```

---

## 🚀 Running the project (from a fresh clone)

Follow these steps after cloning the repo (or when someone else clones it).

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/observa-rag.git
cd observa-rag
```

(Replace with your actual repo URL.)

### 2. Create a virtual environment

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

You should see `(.venv)` in your prompt.

### 3. Install dependencies

With the virtual environment **activated**:

```bash
pip install -r requirements.txt
```

This installs pandas, numpy, sentence-transformers, chromadb, llama-index, langchain, langchain-google-genai, streamlit, and related packages. The first run may take a few minutes; the embedding model is downloaded on first use.

### 4. (Optional) Set up Gemini for LLM answers

To get **AI-generated answers** in the Streamlit app (instead of only retrieved chunks):

1. Get an API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Create a `.env` file in the **project root** (same folder as `app_streamlit.py`). You can copy `.env.example` and add your key:

   ```bash
   cp .env.example .env
   # Then edit .env and set GOOGLE_API_KEY=your_actual_key
   ```

   Or create `.env` with a single line: `GOOGLE_API_KEY=your_api_key_here`

   **Important:** `.env` is in `.gitignore` and must not be committed. `.env.example` is committed as a template only (no real keys).

3. The app loads `.env` automatically. Alternatively you can set the variable in the shell before running Streamlit:

   **Windows (PowerShell):** `$env:GOOGLE_API_KEY = "your_key"`  
   **Linux/macOS:** `export GOOGLE_API_KEY=your_key`

If `GOOGLE_API_KEY` is not set, the app still runs but will show a message asking you to set it for Gemini answers; retrieved documents will still appear.

### 5. Add metrics data (CSV)

Place at least one daily metrics CSV in `data/daily_csv/`. The pipeline expects columns: `timestamp`, `service`, `environment`, `region`, `metric_name`, `metric_value`, `unit`, and optionally `labels`, `source_csv`. See [CSV schema](#-csv-schema-recommended) below.

If you don’t have a CSV yet, you can run the app anyway; searches will return no documents until you run an ingest.

### 6. Run the daily ingest

Ingest processes the CSV, builds windows, detects anomalies, creates daily summaries, embeds them, and upserts into Chroma. Run **once per CSV** (or whenever you have new data):

```bash
python run_daily_ingest.py --csv data/daily_csv/your_file.csv
```

Example:

```bash
python run_daily_ingest.py --csv data/daily_csv/metrics_2025-02-14.csv
```

You should see output like: `Upserted anomalies=6 summaries=8 into Chroma collection 'observability_docs'.`

### 7. Launch the Streamlit UI

```bash
streamlit run app_streamlit.py
```

The app opens in your browser (typically `http://localhost:8501`). You can:

- Type a question in the text area
- Use the sidebar to filter by type (anomaly/summary), environment, service, region, and Top K
- Click **Search** to run retrieval (and Gemini, if the API key is set)
- View the **Answer** (from Gemini) and **Retrieved Documents** below

### 8. (Optional) Run RAG scripts without the UI

- **LangChain + Gemini (used by the app):** The app calls `build_chroma_rag()` from `rag_langchain.py`; you can reuse the same function in your own scripts.
- **LlamaIndex:**  
  ```bash
  python rag_llamaindex.py
  ```

---

## ⚙️ CSV schema (recommended)

| Column        | Type        | Notes                          |
|---------------|-------------|--------------------------------|
| timestamp     | ISO8601 UTC | Intraday sample time           |
| date          | date        | Can be derived from timestamp  |
| service       | text        | e.g. `checkout-api`           |
| environment   | text        | e.g. `prod`, `staging`, `dev` |
| region        | text        | e.g. `ap-south-1`             |
| metric_name   | text        | e.g. `p95_latency_ms`         |
| metric_value  | numeric     | Raw value                      |
| unit          | text        | e.g. `ms`, `%`, `count`       |
| labels        | text        | Optional; leave empty or `k=v` |
| source_csv    | text        | Optional; filename for trace  |

Empty `labels` are normalized to `""` during ingest so windowing works correctly.

---

## 🧪 Configuration and tuning

Edit `config.py` to adjust:

- **WINDOW_SIZE** – `"60min"` or `"15min"` for window granularity
- **Z_THRESH** – z-score threshold for anomalies (e.g. 2.0 or 3.0)
- **SATURATION_P95** – threshold for percent metrics (e.g. 85)
- **CHROMA_DIR** – uses an absolute path so ingest and Streamlit share the same store
- **GEMINI_MODEL** – e.g. `gemini-2.5-flash` or `gemini-2.5-pro` (use a [current model](https://ai.google.dev/gemini-api/docs/models))

---

## 🔧 Troubleshooting

### "No windowed data" during ingest

- Ensure your CSV has the required columns and that `labels` is present (can be empty). The ingest fills missing `labels` so that groupby in windowing produces groups.

### Chroma: "Nothing found on disk" / "Error creating hnsw segment reader"

- The Chroma store may be corrupted or from a different path. Delete the `chroma_store` folder in the project root and re-run the ingest so Chroma is recreated.
- The app uses an absolute path for Chroma (in `config.py`) so that ingest and Streamlit always use the same directory.

### Gemini: 404 NOT_FOUND for model

- Older model IDs (e.g. `gemini-1.5-flash`) are no longer available. In `config.py` set **GEMINI_MODEL** to a current model such as `gemini-2.5-flash` or `gemini-2.5-pro`. See [Gemini models](https://ai.google.dev/gemini-api/docs/models).

### No documents found in the UI

- Run the ingest first: `python run_daily_ingest.py --csv data/daily_csv/your_file.csv`. The UI only shows results for data that has been ingested into Chroma.

### Streamlit or scripts can’t find GOOGLE_API_KEY

- Create a `.env` in the project root with `GOOGLE_API_KEY=your_key`.
- Or set the variable in the shell before running:  
  `$env:GOOGLE_API_KEY = "your_key"` (PowerShell) or `export GOOGLE_API_KEY=your_key` (Linux/macOS).

---

## 🧰 Pushing to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Observability RAG with Chroma, LangChain, Gemini"

# Create the repo on GitHub, then:
git branch -M main
git remote add origin https://github.com/<your-username>/observa-rag.git
git push -u origin main
```

**Do not commit:**

- `.env` (API keys) – already in `.gitignore`
- `.venv/` – already in `.gitignore`
- `chroma_store/` – already in `.gitignore`

New clones should follow the steps in [Running the project](#-running-the-project-from-a-fresh-clone) and create their own `.env` and `chroma_store` locally.


