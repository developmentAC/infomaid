#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Parts of this code taken from reference: https://github.com/pixegami/rag-tutorial-v2
import argparse

# from langchain.vectorstores.chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from infomaid import get_embedding_function
from rich.console import Console

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

console = Console()

def main(query_text, myModel):
    # console.print("This is query_data.main()") # for debugging
    
    myResult = None # define variable used below in loop
    try:
        myResult = query_rag(query_text, myModel)
    except Exception:
        console.print("\t :poop: [red]There seems to be a problem. Is Ollama server installed and running?[/red]")
        exit()
    return myResult

def query_rag(query_text: str, myModel: str):
    # console.print("This is query_data.query_rag()") # for debugging

    # Prepare the DB.
    embedding_function = get_embedding_function.get_embedding_function(myModel)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}" 

    # console.print(f"\t [pruple]{formatted_response}[/purple]") #For debugging
    return response_text
