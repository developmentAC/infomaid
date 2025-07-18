#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG System Test Module

This module contains unit tests for the Infomaid RAG system components.
It validates the functionality of query processing, response generation,
and integration between different system modules.

Test Categories:
    - Basic functionality tests
    - Query processing validation
    - Response accuracy verification
    - Model integration testing

Dependencies:
    - pytest for test framework
    - infomaid modules for testing targets
    - Sample data for validation

Usage:
    pytest test_rag.py
    python -m pytest test_rag.py -v
"""

# from query_data import query_rag
from infomaid import query_data as qd
from langchain_community.llms.ollama import Ollama

from infomaid import main as m


def test_bighelp():
    """
    Test the getBigHelp function for basic functionality.
    
    This is a simple unit test that verifies the getBigHelp function
    returns the expected string value. It serves as a basic smoke test
    for the main module functionality.
    
    Asserts:
        getBigHelp() returns the string "getBigHelp"
    """
    assert m.getBigHelp() == "getBigHelp"


# end of test_bighelp()


def test_astroBillStreetAddress():
    """
    Test the basic RAG query functionality with a specific query.
    
    This test validates that the RAG system can correctly retrieve
    and process information from the knowledge base. It uses a specific
    query about a character's street address to verify end-to-end
    functionality.
    
    Test Details:
        - Query: Street address of AstroBill character
        - Expected: Response contains "Celestial Street"
        - Model: nomic-embed-text embedding model
        
    Asserts:
        The response contains the expected street name
        
    Notes:
        - Requires populated database with AstroBill data
        - Tests both retrieval and response generation
        - Validates model integration with ChromaDB
    """
    query_text = "What street does AstroBill live on. Answer with the street name only."
    expected_response = "Celestial Street"
    useThisModel = "nomic-embed-text"
    assert expected_response in qd.query_rag(query_text, useThisModel)


# end of test_astroBillStreetAddress()
