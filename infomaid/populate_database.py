#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Parts of this code taken from reference: https://github.com/pixegami/rag-tutorial-v2

import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# from get_embedding_function import get_embedding_function
from infomaid import get_embedding_function
from langchain.vectorstores.chroma import Chroma

from rich.console import Console
from infomaid import main

CHROMA_PATH = "chroma"
DATA_PATH = "data"

console = Console()


def main(resetDB: bool, myModel: str) -> None:
    # console.print("\t This is populate_databases.main()")
    console.print("\t :sparkles: Resetting database: {resetDB}")
    if resetDB:  # clear out old data from the database
        print("\t ✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks, myModel)


def load_documents():
    """ Load data into Document objects from pdf files located in the path defined above. """
    # console.print("This is load_documents()") # for debugging
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()





def split_documents(documents: list[Document]):
    # console.print("This is split_documents()") # for debugging

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document], myModel: str):
    # Load the existing database.
    # console.print("This is add_to_chroma()") # for debugging
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function.get_embedding_function(myModel),
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    console.print(
        f"\t [cyan] Number of existing documents in DB: {len(existing_ids)}[/cyan]"
    )

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    # console.print("chunks in add_to_chroma()") # for debugging

    try:

        if len(new_chunks):
            print(f"\t 👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            # db.persist() # deprecated?
        else:
            print("\t ✅ No new documents to add")

    except Exception:
        console.print(
            "\t :poop: [red]There seems to be a problem. Is Ollama server installed and running?[/red]"
        )
        exit()

    # console.print("end of add_to_chroma()") # for debugging


def calculate_chunk_ids(chunks):

    # This will create IDs like "data_test/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
