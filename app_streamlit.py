# app_streamlit.py
from pathlib import Path

# Load .env from project root so GOOGLE_API_KEY is available
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import streamlit as st
from rag_langchain import build_chroma_rag

st.set_page_config(page_title="Observability RAG on Chroma", layout="wide")
st.title("Observability RAG (Anomalies + Summaries)")

# -----------------------------
# Sidebar Filters
# -----------------------------
with st.sidebar:
    st.header("Filters")

    doc_type = st.selectbox("Type", ["any", "anomaly", "summary"])
    environment = st.selectbox("Environment", ["any", "prod", "staging", "dev"])
    service = st.text_input("Service (optional)")
    region = st.text_input("Region (optional)")
    top_k = st.slider("Top K", 5, 50, 10)

query = st.text_area(
    "Ask a question",
    placeholder="e.g., Show latency anomalies in prod last week for checkout-api"
)

# -----------------------------
# Build RAG function
# -----------------------------
rag_chain = build_chroma_rag(top_k=top_k)

# -----------------------------
# Execute search
# -----------------------------
if st.button("Search") and query.strip():

    # Build proper Chroma filter
    filters = []
    if doc_type != "any":
        filters.append({"type": doc_type})
    if environment != "any":
        filters.append({"environment": environment})
    if service:
        filters.append({"service": service})
    if region:
        filters.append({"region": region})

    if len(filters) == 0:
        where = None
    elif len(filters) == 1:
        where = filters[0]
    else:
        where = {"$and": filters}

    # Perform search
    results = rag_chain(query, where=where)

    # -----------------------------
    # Show Gemini answer (if present)
    # -----------------------------
    if results.get("answer"):
        st.subheader("Answer")
        st.write(results["answer"])

    # -----------------------------
    # Show Retrieved Documents
    # -----------------------------
    st.subheader("Retrieved Documents")

    docs = results.get("source_documents", [])

    if not docs:
        st.write("No documents found.")
    else:
        for i, doc in enumerate(docs, 1):
            meta = doc["metadata"]
            title = (
                f"{i}. "
                f"{meta.get('type','summary').upper()} — "
                f"{meta.get('service','')} "
                f"{meta.get('metric_name','')} — "
                f"{meta.get('date','')}"
            )
            with st.expander(title):
                st.write(doc["content"])
                st.json(meta)
