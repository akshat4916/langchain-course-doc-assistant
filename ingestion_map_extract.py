import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi

from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import *

# configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=1536,
    chunk_size=50,
    # retry_min_seconds=10
)

#chroma = chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(
    index_name = "langchain-docs-2025",
    embedding= embeddings
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(
    max_depth = 3,
    max_breadth=3,
    max_pages = 1000
)
# tavily_crawl = TavilyCrawl()

def chunk_urls(urls:List[str], chunk_size: int= 20) -> List[List[str]]:
    """Split URLS into chunks of specified size."""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunks.append(urls[i:i + chunk_size])
    return chunks

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs."""
    try:
        log_info(
            f" TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs",
            Colors.BLUE
        )
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(
            f"TavilyExtract: Completed batch {batch_num} - extracted {len(docs.get('results', []))} documents."
        )
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return []
    
async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENT EXTRATION PHASE")
    log_info(
        f" Tavily Extract: Starting concurrent extraction of {len(url_batches)} batches",
        Colors.DARKCYAN
    )

    tasks = [extract_batch(batch, idx + 1) for idx, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and flatten results
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            failed_batches += 1
            log_error(f"TavilyExtract: Batch extraction failed with exception: {result}")
        else:
            for extracted_page in result["results"]:
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]}
                )
                all_pages.append(document)

    log_success(
        f"TavilyExtract: Successfully extracted {len(all_pages)} pages."
    )
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed during extraction.")  
    
    return all_pages

async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Asynchronously index documents in batches."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f" Vector Store: Preparing to add {len(documents)} documents to vector store.",
        Colors.DARKCYAN
    )

    # Create batches
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

    log_info(
        f" Vector Store Indexing:Split into {len(batches)} batches of {batch_size} documents each."
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(f"Vector Store: Successfully added batch {batch_num}/{len(batches)}  ({len(batch)} documents)")
        except Exception as e:
            log_error(f"Vector Store Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True
    
    # Process batches concurrently
    tasks = [add_batch(batch, idx + 1) for idx, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successes and failures
    success_count = sum(1 for result in results if result is True)

    if success_count == len(batches):
        log_success(f"Vector Store Indexing: Successfully indexed all {success_count}/{len(batches)} batches.")
    else:
        log_warning(f"Vector Store Indexing: Indexed {success_count}/{len(batches)} batches successfully.")



async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        " Tavily Crawl: Starting to Crawl documentation from https://python.langchain.com/",
        Colors.PURPLE
    )

    site_map = tavily_map.invoke("https://python.langchain.com/")

    log_success(f"TavilyMap: Successfully mapped {len(site_map['results'])} URLs from the site map.")

    # Split URLs into chunks for extraction
    url_batches = chunk_urls(list(site_map["results"]), chunk_size=20)
    log_info(
        f" URL Processing: Split {len(site_map['results'])} URLs into {len(url_batches)} batches",
        Colors.BLUE
    )

    # Extract documents concurrently from URLs
    all_docs = await async_extract(url_batches)

    # Split documents into chunks and add to vector store
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f" Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitter_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitter_docs)} chunks from {len(all_docs)} documents."
    )

    # Process documents asynchronously and add to vector store
    await index_documents_async(splitter_docs, batch_size=50)

    log_header("PIPELINE COMPLETED")
    log_success("Documentation ingestion pipeline completed successfully!")
    log_info("Summary:", Colors.BOLD)
    log_info(f" URLs mapped: {len(site_map['results'])}")
    log_info(f" Documents extracted: {len(all_docs)}")
    log_info(f" Chunks created: {len(splitter_docs)}")


    # Crawl the documentation site
    # res = tavily_crawl.invoke({
    #     "url": "https://python.langchain.com/",
    #     "max_depth": 5,
    #     "extract_depth": "advanced",
    #     "instructions": "content on ai agents"
    # })

    # all_docs = [Document(page_content=doc["raw_content"], metadata={"source": doc["url"]}) for doc in res["results"]]
    # log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documents.")

if __name__ == "__main__":
    asyncio.run(main())