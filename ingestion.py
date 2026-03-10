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
    retry_min_seconds=10
)

#chroma = chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(
    index_name = "langchain-docs-2025",
    embedding= embeddings
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(
    max_depth = 5,
    max_breadth=20,
    max_pages = 1000
)
tavily_crawl = TavilyCrawl()



async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        " Tavily Crawl: Starting to Crawl documentation from https://python.langchain.com/",
        Colors.PURPLE
    )

    # Crawl the documentation site
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 5,
        "extract_depth": "advanced",
        "instructions": "content on ai agents"
    })

    all_docs = [Document(page_content=doc["raw_content"], metadata={"source": doc["url"]}) for doc in res["results"]]
    log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documents.")

if __name__ == "__main__":
    asyncio.run(main())