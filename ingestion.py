import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=True,
    chunk_size=50,
    retry_min_seconds=10,
)
# Using Pinecone as requested
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index", embedding=embeddings
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously and add them to the vectorstore."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            # Prefer async add if available
            if hasattr(vectorstore, "aadd_documents"):
                await vectorstore.aadd_documents(batch)
            else:
                # Try sync methods in a thread to avoid blocking the event loop
                texts = [d.page_content for d in batch]
                metadatas = [d.metadata for d in batch]

                if hasattr(vectorstore, "add_documents"):
                    # Many vectorstores accept List[Document]
                    await asyncio.to_thread(vectorstore.add_documents, batch)
                elif hasattr(vectorstore, "add_texts"):
                    # Some accept raw texts + metadatas
                    await asyncio.to_thread(vectorstore.add_texts, texts, metadatas)
                else:
                    # Last-resort: try to build what Pinecone expects (indexing via upsert)
                    raise RuntimeError(
                        "Vectorstore has no supported add method (no aadd_documents, add_documents or add_texts)."
                    )

            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently (bounded concurrency would be better for very large numbers)
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


def to_document(item: Any) -> Document:
    """
    Normalize an item (Document | dict | str | other) into a langchain_core.documents.Document
    Tries common keys: page_content, text, content, html, body.
    Keeps other fields as metadata.
    """
    # already a Document
    if isinstance(item, Document):
        return item

    # dict -> try common content keys
    if isinstance(item, dict):
        for key in ("page_content", "text", "content", "html", "body"):
            if key in item and item[key]:
                content = item[key]
                break
        else:
            # fallback: try nested content or stringify
            content = item.get("content") or item.get("html") or str(item)

        # metadata: everything except the chosen content key
        metadata: Dict[str, Any] = {}
        for k, v in item.items():
            if k not in ("page_content", "text", "content", "html", "body"):
                metadata[k] = v

        return Document(page_content=str(content), metadata=metadata)

    # string or other -> stringify
    return Document(page_content=str(item), metadata={})


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )

    # Crawl the documentation site
    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 5,
            "extract_depth": "advanced",
        }
    )

    # res["results"] is commonly a list - be defensive
    raw_results = res.get("results") if isinstance(res, dict) else res

    if raw_results is None:
        log_error("TavilyCrawl returned no results (res['results'] is None). Aborting.")
        return

    # Normalize to list if a single item was returned
    if not isinstance(raw_results, list):
        raw_results = [raw_results]

    # Convert all raw items to Document objects
    all_docs: List[Document] = [to_document(d) for d in raw_results]

    # optional: log sample doc shape for debugging
    if raw_results:
        sample = raw_results[0]
        if isinstance(sample, dict):
            log_info(f"Sample raw doc keys: {list(sample.keys())}")
        else:
            log_info(f"Sample raw doc type: {type(sample)}")

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
