import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


# Initialize the Google Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=1536,
    chunk_size=50,
)

# Initialize the Pinecone vector store
vectorstore = PineconeVectorStore(
    index_name="langchain-docs-2025",
    embedding=embeddings
)

# Initialize the Google Gemini chat model
chat_model = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """ Retrieve relevant documentation to help answer user queries about LangChain."""
    # Retrieve top 4 most similar documents
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # Return both serialized content and original documents as artifacts
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query (str): The user query to answer.
    
    Returns:
        Dictionary containing"
            - answer: the generated answer
            - context: List of retrieved documents
    """
    # Create an agent with the chat model and retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    
    agent = create_agent(
        chat_model,
        tools = [retrieve_context],
        system_prompt=system_prompt
    )

    # Build messages list
    messages = [{"role": "user", "content": query}]

    # Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the last AI message
    answer = response["messages"][-1].content

    # Extract retrieved context from tool calls
    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message,"artifact"):
            # the artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    
    return {
        "answer":answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)