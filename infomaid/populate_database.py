#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Parts of this code taken from reference: https://github.com/pixegami/rag-tutorial-v2

import argparse
import os
import shutil
import nltk

# from langchain.document_loaders.pdf import PyPDFDirectoryLoader # deprecated code
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader

from langchain_core.documents import Document


# from get_embedding_function import get_embedding_function
from infomaid import get_embedding_function

# from langchain.vectorstores.chroma import Chroma # deprecated code
from langchain_community.vectorstores import Chroma

from rich.console import Console
from infomaid import main

CHROMA_PATH = "chroma"
DATA_PATH = "data"

console = Console()

def setupNLTK() -> str:
    """ A function to install NLTK word banks in the local directory."""

    # Define a local directory for storing NLTK data
    local_nltk_dir = os.path.join(os.getcwd(), "nltk_data")

    # Ensure the directory exists
    os.makedirs(local_nltk_dir, exist_ok=True)

    # Set the NLTK data path to the local directory
    nltk.data.path.append(local_nltk_dir)

    # Download the 'averaged_perceptron_tagger_eng' package to the local directory
    nltk.download('averaged_perceptron_tagger_eng', download_dir=local_nltk_dir)

    print(f"Downloaded 'averaged_perceptron_tagger_eng' to {local_nltk_dir}")

# end of setupNLTK()


def main(resetDB: bool, myModel: str, usePDF: bool, useXML: bool, useTXT: bool) -> None:
    # console.print("\t This is populate_databases.main()")
    if not usePDF and not useXML and not useTXT:
        console.print("\t :scream: [bright_green]Nothing to do![/bright_green]")
        console.print(
            "\t [bright_cyan]Note: Use options: [bright_yellow]--usepdf[/bright_yellow] or [bright_yellow]--usexml [/bright_yellow]"
        )
        exit()

    console.print(
        f"\t[bright_green] :rocket: Resetting database:[bright_yellow] {resetDB}[/bright_yellow]"
    )

    if resetDB:  # clear out old data from the database
        console.print("\t :rocket: [bright_green]Clearing Database[/bright_green]")
        clear_database()
        console.print("\t :sparkles: [bright_cyan]Setting up language models in local directory ...[/bright_cyan]")
        setupNLTK() # install language models for txt and nxml tasks

    if usePDF:
        # print(f"  +++ Using option : usePDF: {usePDF}")

        # Create (or update) the data store.
        try:
            documentsPDF_list = load_documents_PDF()
        except Exception:
            console.print("\t :scream:[bright_red] There seems to be a unexpected file with < pdf > in the\n\t filename in < data/ >. Please check your files and try again.[/bright_red]")
            exit()
        # print(f"use pdf; documents: {documentsPDF_list}, {type(documentsPDF_list)}") # returns a list
        #   input("Current Chunk above. Press any key to continue!!")
        chunks = split_documents(documentsPDF_list)
        add_to_chroma(chunks, myModel)
        # console.print("\t :smiley: Populating complete")

    if useXML:
        # print(f"  +++ Using option : useXML: {useXML}")
        try:
            documentsXML_list = load_documents_NXML()
        except Exception:
            console.print("\t :scream:[bright_red] There seems to be a unexpected file with < xml > in the\n\t filename in < data/ >. Please check your files and try again.[/bright_red]")
            exit()
        # console.print(f"[cyan]main() documentsXML_list : [bright_yellow]{documentsXML_list}[/bright_yellow]") #list
        for i in range(len(documentsXML_list)):
            chunks = split_documents(documentsXML_list[i])
            # print(f"{i}: useXML chunks : {chunks}")
            #   input("Current Chunk above. Press any key to continue!!")
            add_to_chroma(chunks, myModel)

    if useTXT:
        # print(f"  +++ Using option : useTXT: {useTXT}")
        try:
            documentsTXT_list = load_documents_TEXT()
        except Exception:
            console.print("\t :scream:[bright_red] There seems to be a unexpected file with < txt > in the\n\t filename in < data/ >. Please check your files and try again.[/bright_red]")
            exit()            
        # console.print(f"[cyan]main() documentsXML_list : [bright_yellow]{documentsXML_list}[/bright_yellow]") #list
        for i in range(len(documentsTXT_list)):
            chunks = split_documents(documentsTXT_list[i])
            # print(f"{i}: useXML chunks : {chunks}")
            #   input("Current Chunk above. Press any key to continue!!")
            add_to_chroma(chunks, myModel)

    console.print("\t :smiley: [bright_green]Populating complete![/bright_green]")


# end of main()


def load_documents_PDF():
    """Load data into Document objects from pdf files located in the path defined above."""
    # console.print("This is load_documents_PDF()") # for debugging
    myFiles_list = get_files_list_from_directory(
        DATA_PATH, "pdf"
    )  # length is the number of articles found
    console.print("\t [bright_green]:sparkles: Found files ... [/bright_green]")
    for thisFile in myFiles_list:
        console.print(f"\t [cyan]File :[bright_yellow] {thisFile}[/bright_yellow]")
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


# end of load_documents_PDF()


def load_documents_NXML():
    """A function to load the xml files, to produce document objects to store in a dictionary; key:filename, value: document object."""
    # print("This is load_documents_NXML()")

    myArticles_list = []  # list to contain all article documents
    # find which files we are working with
    myFiles_list = get_files_list_from_directory(
        DATA_PATH, "nxml"
    )  # length is the number of articles found
    # print(f"myFiles_list: myFiles_list, LENGTH = {len(myFiles_list)}")
    for thisFile in myFiles_list:
        currentFile_str = thisFile
        console.print(
            f"\t [bright_cyan]Current File :[bright_yellow] {currentFile_str}[/bright_yellow]"
        )
        loader = UnstructuredXMLLoader(
            currentFile_str,
        )  # open, parse nxml documents
        docs = loader.load()
        myArticles_list.append(docs)
    return myArticles_list


# end of load_documents_NXML()

def load_documents_TEXT():
    """A function to load the xml files, to produce document objects to store in a dictionary; key:filename, value: document object."""
    # print("This is load_documents_NXML()")

    myArticles_list = []  # list to contain all article documents
    # find which files we are working with
    myFiles_list = get_files_list_from_directory(
        DATA_PATH, "txt"
    )  # length is the number of articles found
    # print(f"myFiles_list: myFiles_list, LENGTH = {len(myFiles_list)}")
    for thisFile in myFiles_list:
        currentFile_str = thisFile
        console.print(
            f"\t [bright_cyan]Current File :[bright_yellow] {currentFile_str}[/bright_yellow]"
        )
        loader = TextLoader(
            currentFile_str,
        )  # open, parse nxml documents
        docs = loader.load()
        myArticles_list.append(docs)
    return myArticles_list


# end of load_documents_TEXT()


def get_files_list_from_directory(directory, myFileExt_str):
    """A function to determine the nxml files from the directory. Return a list of files with paths. myFileExt_str is the extension for types of files that we want to list (i.e., pdf, nxml or other type.)"""
    file_list = []
    # Traverse the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if (
                myFileExt_str.lower() in file.lower()
            ):  # is this the correct  extension of file to list?
                # Append file names to the list
                file_list.append(os.path.join(root, file))
    # print(f"get_files_list_from_directory(): file_list = {file_list}")
    return file_list


# end of get_files_list_from_directory()


def split_documents(documents: list[Document]):
    # console.print("This is split_documents()") # for debugging

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


# end of split_documents()


def add_to_chroma(chunks: list[Document], myModel: str):
    # Load the existing database.
    # console.print(f"++++++ This is add_to_chroma() :: {chunks}") # for debugging
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
        f"\n\t [bright_cyan]Number of existing documents in DB: {len(existing_ids)}[/bright_cyan]"
    )

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    # console.print("chunks in add_to_chroma()") # for debugging

    try:

        if len(new_chunks):
            console.print(
                f"\t [bright_green]👉 Adding new documents: [bright_yellow]{len(new_chunks)}[/bright_yellow]"
            )
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)

        else:
            console.print("\t []bright_green]✅ No new documents to add.[/bright_green]")

    except Exception:
        console.print(
            "\t  :scream: [red]There appears to be no connection to Ollama or to a model. Is Ollama client running? Has the model been loaded?\n\t  Try this command: ollama pull name-your-model \n\t  Exiting... [/red]"
        )
        exit()
        # console.print(
        #     "\t :poop: [red]There seems to be a problem. Is Ollama server installed and running?[/red]"
        # )
        # exit()

    # console.print("end of add_to_chroma()") # for debugging


# end of add_to_chroma()


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


# end of calculate_chunk_ids()


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


# end of clear_database()
