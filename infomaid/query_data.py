#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Query Data Module

This module provides the core query functionality for the Infomaid RAG system.
It handles document retrieval from the ChromaDB vector database and generates
responses using language models. The module supports both standard vector
similarity search and enhanced retrieval methods.

Key Features:
    - Vector similarity search using ChromaDB
    - Integration with Ollama language models
    - Rich console output formatting
    - Enhanced RAG capabilities (when available)
    - Configurable prompt templates

Dependencies:
    - ChromaDB for vector storage and retrieval
    - LangChain for prompt templating and model integration
    - Ollama for language model inference
    - Rich for enhanced console output

Usage:
    python query_data.py "What is machine learning?"
    python query_data.py --enhanced "Explain neural networks"
"""

# Parts of this code taken from reference: https://github.com/pixegami/rag-tutorial-v2
import argparse

# Configure ChromaDB before any other imports to suppress telemetry
from infomaid.chromadb_config import configure_chromadb
configure_chromadb()

try:
    from langchain_chroma import Chroma
except ImportError:
    # Fallback to the older import if the new package is not available
    from langchain_community.vectorstores import Chroma

from langchain.prompts import ChatPromptTemplate

try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    # Fallback to the older import if the new package is not available
    from langchain_community.llms.ollama import Ollama

from infomaid import get_embedding_function
from rich.console import Console

# Import enhanced RAG functionality with graceful fallback
try:
    from infomaid.enhanced_rag import EnhancedQueryProcessor
    ENHANCED_RAG_AVAILABLE = True
except ImportError:
    ENHANCED_RAG_AVAILABLE = False
    console = Console()
    console.print("\t [yellow] Enhanced RAG features not available. Install scikit-learn for improved functionality.[/yellow]")

# Configuration constants
CHROMA_PATH = "chroma"  # Default path to ChromaDB database

# Default prompt template for response generation
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

console = Console()


def main(query_text, myModel, use_enhanced=False, retrieval_method="hybrid"):
    """
    Main query function that handles both standard and enhanced RAG queries.
    
    This function serves as the primary interface for querying the RAG system.
    It supports both standard vector similarity search and enhanced retrieval
    methods including TF-IDF, BM25, and hybrid approaches.
    
    Args:
        query_text (str): The user's search query
        myModel (str): The embedding model name to use
        use_enhanced (bool): Whether to use enhanced RAG features
        retrieval_method (str): Method for enhanced RAG ("hybrid", "vector", "tfidf", "bm25")
        
    Returns:
        str: Generated response based on retrieved context
        
    Side Effects:
        - Prints status messages to console
        - Exits on Ollama connection errors
        
    Raises:
        SystemExit: If Ollama server is not available
    """
    # console.print("This is query_data.main()") # for debugging

    myResult = None  # Initialize result variable for error handling
    try:
        # Choose between enhanced and standard RAG based on availability and preference
        if use_enhanced and ENHANCED_RAG_AVAILABLE:
            console.print("\t [cyan] Using Enhanced RAG system[/cyan]")
            processor = EnhancedQueryProcessor(model=myModel)
            myResult = processor.enhanced_query_rag(query_text, retrieval_method=retrieval_method)
        else:
            if use_enhanced and not ENHANCED_RAG_AVAILABLE:
                console.print("\t [yellow] Enhanced RAG not available, falling back to standard RAG[/yellow]")
            myResult = query_rag(query_text, myModel)
    except Exception:
        console.print(
            "\t :poop: [red]There seems to be a problem. Is Ollama server installed and running?[/red]"
        )
        exit()
    return myResult


# end of main()


def query_rag(query_text: str, myModel: str):
    """
    Standard RAG query function using vector similarity search.
    
    This function implements the basic RAG approach using ChromaDB for
    vector similarity search and Ollama for response generation. It
    retrieves relevant documents based on semantic similarity and
    generates contextual responses.
    
    Args:
        query_text (str): The user's search query
        myModel (str): The embedding model name to use
        
    Returns:
        str: Generated response based on retrieved context
        
    Process:
        1. Initialize embedding function and ChromaDB connection
        2. Determine optimal k value based on available documents
        3. Perform vector similarity search
        4. Format context and generate prompt
        5. Generate response using Ollama model
        
    Side Effects:
        - Prints search information to console
        - Falls back to default k=5 if document count unavailable
    """
    # console.print("This is query_data.query_rag()") # for debugging

    # Initialize the vector database with specified embedding model
    embedding_function = get_embedding_function.get_embedding_function(myModel)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Determine optimal number of documents to retrieve
    try:
        collection = db._collection
        available_docs = collection.count()
        # Adjust k to not exceed available documents to prevent errors
        k = min(5, available_docs) if available_docs > 0 else 1
        console.print(f"\t [cyan] Searching {available_docs} documents with k={k}[/cyan]")
    except Exception:
        k = 5  # fallback to original value if collection access fails
    
    # Perform vector similarity search to find relevant documents
    results = db.similarity_search_with_score(query_text, k=k)

    # Combine retrieved documents into a single context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Format the prompt using the template and retrieved context
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)  # Uncomment for debugging prompt content

    # Generate response using Ollama language model
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)
    
    # Display the generated response with highlighting
    console.print(f"\t [bright_blue] {response_text}[/bright_blue]")
    return response_text


# end of query_rag
