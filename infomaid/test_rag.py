#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from query_data import query_rag
from infomaid import query_data as qd
from langchain_community.llms.ollama import Ollama

from infomaid import main as m


def test_bighelp():
    """dummyTest"""
    assert m.getBigHelp() == "getBigHelp"


# end of test_bighelp()


def test_astroBillStreetAddress():
    """testing the basic query code"""
    query_text = "What street does AstroBill live on. Answer with the street name only."
    expected_response = "Celestial Street"
    useThisModel = "nomic-embed-text"
    assert expected_response in qd.query_rag(query_text, useThisModel)


# end of test_astroBillStreetAddress()
