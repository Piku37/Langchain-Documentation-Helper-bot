# backend/core.py
from dotenv import load_dotenv

load_dotenv()

import os
from typing import Any, Dict, List, Optional

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from consts import INDEX_NAME
from logger import log_info, log_error

# Optional: try to initialize the pinecone client if available
try:
    import pinecone

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV") or os.getenv("PINECONE_ENVIRONMENT")
    if PINECONE_API_KEY and PINECONE_ENV:
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        log_info("Initialized Pinecone client from environment variables")
except Exception:
    # If pinecone client isn't installed or init fails, the LangChain Pinecone wrapper
    # may still work if it reads env vars; we'll let PineconeVectorStore handle it.
    pass

# --- Global singletons (reuse across calls) ---
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")
# Use INDEX_NAME from consts for consistency
VECTORSTORE = PineconeVectorStore(index_name=INDEX_NAME, embedding=EMBEDDINGS)

# Reuse chat model where appropriate
CHAT_DEFAULT = ChatOpenAI(verbose=True, temperature=0)


def _normalize_result_to_text(result: Any) -> str:
    if result is None:
        return ""

    if isinstance(result, dict):
        for k in ("output_text", "answer", "text", "result", "output"):
            v = result.get(k)
            if isinstance(v, str) and v.strip():
                return v
        for v in result.values():
            if isinstance(v, dict):
                for k2 in ("text", "answer", "output_text"):
                    if k2 in v and isinstance(v[k2], str):
                        return v[k2]
        return str(result)

    for attr in ("output_text", "answer", "text"):
        if hasattr(result, attr):
            maybe = getattr(result, attr)
            if isinstance(maybe, str):
                return maybe

    return str(result)


def format_docs(docs: List[Document]) -> str:
    """Concatenate retrieved documents' page_content into one string (used by run_llm2)."""
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in (docs or []))


def run_llm(
    query: str, chat_history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Returns a dict:
      {
        "answer": "<plain string answer>",
        "context": List[Document],    # retrieved documents (so UI can read metadata/source)
        "raw": <original chain result>
      }
    """
    chat_history = chat_history or []
    try:
        docsearch = VECTORSTORE
        chat = CHAT_DEFAULT

        rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        # Build chain
        stuff_documents_chain = create_stuff_documents_chain(
            chat, retrieval_qa_chat_prompt
        )
        history_aware_retriever = create_history_aware_retriever(
            llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
        )
        qa = create_retrieval_chain(
            retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
        )

        # Retrieve docs separately to return as `context` for UI
        retriever = (
            history_aware_retriever
            if hasattr(history_aware_retriever, "get_relevant_documents")
            else docsearch.as_retriever()
        )
        try:
            retrieved_docs = retriever.get_relevant_documents(query)
        except Exception:
            retrieved_docs = docsearch.as_retriever().get_relevant_documents(query)

        # Invoke chain for final answer
        result = qa.invoke(input={"input": query, "chat_history": chat_history})
        answer_text = _normalize_result_to_text(result)

        return {"answer": answer_text, "context": retrieved_docs, "raw": result}
    except Exception as e:
        log_error(f"run_llm error: {e}")
        return {"answer": f"Error: {e}", "context": [], "raw": None}


def run_llm2(
    query: str, chat_history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Runnable-based RAG. Returns dict with same shape as run_llm for consistency.
    """
    chat_history = chat_history or []
    try:
        docsearch = VECTORSTORE
        chat = ChatOpenAI(model="gpt-4o-mini", verbose=True, temperature=0)

        rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        rag_chain = (
            {
                "context": docsearch.as_retriever() | format_docs,
                "input": RunnablePassthrough(),
            }
            | retrieval_qa_chat_prompt
            | chat
            | StrOutputParser()
        )

        retrieve_docs_chain = (lambda x: x["input"]) | docsearch.as_retriever()
        chain = RunnablePassthrough.assign(context=retrieve_docs_chain).assign(
            answer=rag_chain
        )

        # Retrieve docs separately for UI/context
        retrieved_docs = docsearch.as_retriever().get_relevant_documents(query)

        # Invoke the pipeline to compute the answer
        result = chain.invoke({"input": query, "chat_history": chat_history})
        answer_text = _normalize_result_to_text(result)
        return {"answer": answer_text, "context": retrieved_docs, "raw": result}
    except Exception as e:
        log_error(f"run_llm2 error: {e}")
        return {"answer": f"Error: {e}", "context": [], "raw": None}
