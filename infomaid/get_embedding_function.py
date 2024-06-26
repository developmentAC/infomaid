#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Parts of this code taken from reference: https://github.com/pixegami/rag-tutorial-v2

from langchain_community.embeddings.ollama import OllamaEmbeddings

# from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function(myModel):
    """Funtion to use Langchain with a model (recommended: nomic-embed-text) to create the embeddings for project."""
    # embeddings = BedrockEmbeddings(
    # credentials_profile_name="default", region_name="us-east-1"
    # ) # use Amazon Web Server
    embeddings = OllamaEmbeddings(model=myModel)  # use local server with Ollama
    return embeddings
