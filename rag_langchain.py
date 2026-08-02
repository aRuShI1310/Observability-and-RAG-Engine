# rag_langchain.py
import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import CHROMA_DIR, CHROMA_COLLECTION, GEMINI_MODEL

RAG_PROMPT = """You are an observability assistant. Answer the user's question using only the following retrieved context about metrics, anomalies, and daily summaries. If the context does not contain relevant information, say so briefly. Keep the answer concise and factual.

Context:
{context}

Question: {query}

Answer:"""


def build_chroma_rag(top_k=6, use_llm=True):
    """
    Returns a RAG function: retrieve from Chroma, then (optional) generate answer with Gemini.
    Input: query string, optional where filter
    Output: dict with 'answer' (if use_llm) and 'source_documents'
    """
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR
    )

    llm = None
    api_key = os.environ.get("GOOGLE_API_KEY")
    if use_llm and api_key:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("user", RAG_PROMPT),
        ])
        chain = prompt | llm | StrOutputParser()
    else:
        chain = None

    def rag_chain(query: str, where: dict | None = None) -> dict:
        if where:
            results = vectorstore._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where
            )
        else:
            results = vectorstore._collection.query(
                query_texts=[query],
                n_results=top_k
            )

        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })

        out = {"source_documents": docs}

        if chain and docs:
            context = "\n\n---\n\n".join(d["content"] for d in docs)
            out["answer"] = chain.invoke({"context": context, "query": query})
        elif chain and not docs:
            out["answer"] = "No relevant documents were found to answer your question."
        elif use_llm and not api_key:
            out["answer"] = "Set GOOGLE_API_KEY in your environment to enable Gemini-generated answers."

        return out

    return rag_chain
