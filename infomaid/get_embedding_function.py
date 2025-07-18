#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Embedding Function Module

This module provides embedding functions for converting text into vector
representations. It serves as the core interface for generating embeddings
used in the RAG system's vector similarity search capabilities.

Key Features:
    - Ollama embedding model integration
    - Flexible model selection
    - Fallback imports for compatibility
    - Support for local embedding generation
    - Optional cloud provider integration (AWS Bedrock)

Supported Embedding Models:
    - nomic-embed-text (recommended for general use)
    - Custom Ollama models
    - AWS Bedrock models (commented out)

Dependencies:
    - Ollama for local embedding generation
    - LangChain for embedding abstractions

Usage:
    embeddings = get_embedding_function("nomic-embed-text")
    vector = embeddings.embed_query("sample text")
"""

# Parts of this code taken from reference: https://github.com/pixegami/rag-tutorial-v2

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    # Fallback to the older import if the new package is not available
    from langchain_community.embeddings.ollama import OllamaEmbeddings

# from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function(myModel):
    """
    Create and return an embedding function using the specified model.
    
    This function initializes an embedding model that converts text into
    high-dimensional vectors for semantic similarity calculations. It
    primarily uses Ollama for local embedding generation but supports
    cloud-based alternatives.
    
    Args:
        myModel (str): The name of the embedding model to use
                      Recommended: "nomic-embed-text" for general purpose
                      
    Returns:
        OllamaEmbeddings: Configured embedding function that can convert
                         text to vectors for storage and similarity search
                         
    Models:
        - nomic-embed-text: General-purpose embedding model (recommended)
        - Custom models: Any Ollama-compatible embedding model
        
    Example:
        embeddings = get_embedding_function("nomic-embed-text")
        vector = embeddings.embed_query("What is machine learning?")
        
    Notes:
        - Requires Ollama server to be running locally
        - AWS Bedrock integration available but commented out
        - Model must be downloaded in Ollama before use
    """
    # Alternative cloud-based embedding option (AWS Bedrock):
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # ) # use Amazon Web Server
    
    # Use local Ollama server for embedding generation
    embeddings = OllamaEmbeddings(model=myModel)
    return embeddings
