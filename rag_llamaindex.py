
# Build a LlamaIndex VectorStore using Chroma, and a QueryEngine that
# supports semantic search + metadata filtering.

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL

# Configure LlamaIndex to use HuggingFace embeddings
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

def build_llamaindex_query_engine():
    # Connect to existing Chroma collection
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=ChromaSettings(allow_reset=True))
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # LlamaIndex builds an index on top of the vector store
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Create a query engine; you can add metadata filters via retriever in advanced API
    query_engine = index.as_query_engine(similarity_top_k=8)
    return query_engine

if __name__ == "__main__":
    qe = build_llamaindex_query_engine()
    # Example anomaly-focused query
    resp = qe.query("Show latency anomalies in prod last week for checkout-api")
    print(resp)
