
# Two ways to store/retrieve with Chroma:
# A) Direct Chroma client (low-level, explicit control)
# B) Adapters used by LlamaIndex / LangChain (see rag files)
# We create a single collection 'observability_docs' holding BOTH
# 'anomaly' and 'summary' documents, distinguished by 'type' metadata.

import chromadb
from chromadb.config import Settings
from config import CHROMA_DIR, CHROMA_COLLECTION

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))
    # Create or get collection; note: Chroma infers schema based on upserts
    coll = client.get_or_create_collection(name=CHROMA_COLLECTION)
    return coll

def upsert_docs(collection, ids, texts, embeddings, metadatas):
    # Ensure no duplicate IDs inside this batch
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs detected inside current ingest batch.")

    # Convert embeddings safely
    embeddings_clean = [
        e.tolist() if hasattr(e, "tolist") else e
        for e in embeddings
    ]

    # Delete existing documents with same IDs (idempotent behavior)
    try:
        collection.delete(ids=ids)
    except Exception:
        pass  # ignore if they don't exist

    # Now safely upsert
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings_clean,
        metadatas=metadatas
    )


def query_docs(collection, query_text, n_results=10, where: dict = None):
    # Example semantic query with optional metadata filters using Chroma's 'where' param.
    return collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where or {}
    )
