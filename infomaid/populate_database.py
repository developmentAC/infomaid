#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Database Population Module

This module handles the population of the ChromaDB vector database with
documents from various sources. It supports multiple file formats including
PDF, text, XML, and Markdown files, with both standard and enhanced document
processing capabilities.

Key Features:
    - Multi-format document loading (PDF, TXT, XML, MD)
    - Document chunking with configurable strategies
    - Vector embedding generation and storage
    - Enhanced document processing with smart chunking
    - Incremental database updates
    - Rich console output for progress tracking

Document Processing Pipeline:
    1. Load documents from specified directory
    2. Split documents into manageable chunks
    3. Generate vector embeddings
    4. Store embeddings in ChromaDB
    5. Provide progress feedback

Supported File Types:
    - PDF files (.pdf)
    - Text files (.txt)
    - XML files (.xml)
    - Markdown files (.md)

Dependencies:
    - ChromaDB for vector storage
    - LangChain for document processing
    - NLTK for text processing
    - Rich for enhanced console output

Usage:
    python populate_database.py
    python populate_database.py --enhanced  # Uses advanced chunking
"""

# Parts of this code taken from reference: https://github.com/pixegami/rag-tutorial-v2

# Configure ChromaDB before any other imports to suppress telemetry
from infomaid.chromadb_config import configure_chromadb
configure_chromadb()

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

try:
    from langchain_chroma import Chroma
except ImportError:
    # Fallback to the older import if the new package is not available
    from langchain_community.vectorstores import Chroma

from rich.console import Console
from infomaid import main

# Import enhanced document processing with graceful fallback
try:
    from infomaid.enhanced_document_processor import EnhancedDocumentProcessor
    ENHANCED_PROCESSING_AVAILABLE = True
except ImportError:
    ENHANCED_PROCESSING_AVAILABLE = False

# Configuration constants
CHROMA_PATH = "chroma"  # Default path to ChromaDB database
DATA_PATH = "data"      # Default path to document source directory

console = Console()


def setupNLTK() -> str:
    """
    Install and configure NLTK word banks in the local directory.
    
    This function sets up NLTK data locally to ensure text processing
    capabilities are available for document processing. It downloads
    necessary NLTK resources to a local directory to avoid system-wide
    installation requirements.
    
    Returns:
        str: Path to the local NLTK data directory
        
    Side Effects:
        - Creates local nltk_data directory
        - Downloads NLTK punkt tokenizer data
        - Downloads NLTK averaged_perceptron_tagger data
        - Sets NLTK data path to local directory
        
    NLTK Resources Downloaded:
        - punkt: Sentence tokenization
        - averaged_perceptron_tagger: Part-of-speech tagging
    """

    # Define a local directory for storing NLTK data to avoid system dependencies
    local_nltk_dir = os.path.join(os.getcwd(), "nltk_data")

    # Ensure the directory exists for NLTK data storage
    os.makedirs(local_nltk_dir, exist_ok=True)

    # Set the NLTK data path to the local directory for runtime access
    nltk.data.path.append(local_nltk_dir)

    # Download the 'averaged_perceptron_tagger_eng' package to the local directory
    # This is used for part-of-speech tagging during text processing
    nltk.download("averaged_perceptron_tagger_eng", download_dir=local_nltk_dir)

    print(f"Downloaded 'averaged_perceptron_tagger_eng' to {local_nltk_dir}")
    return local_nltk_dir


# end of setupNLTK()


def main(
    resetDB: bool, myModel: str, usePDF: bool, useXML: bool, useTXT: bool, useCSV: bool
) -> None:
    """
    Main function for populating the database with documents.
    
    This function orchestrates the entire database population process,
    handling different file types and database operations based on
    user-specified options. It provides a comprehensive pipeline for
    document ingestion and vector storage.
    
    Args:
        resetDB (bool): Whether to clear existing database before population
        myModel (str): Embedding model name to use for vector generation
        usePDF (bool): Whether to process PDF files
        useXML (bool): Whether to process XML files  
        useTXT (bool): Whether to process text files
        useCSV (bool): Whether to process CSV files
        
    Returns:
        None
        
    Side Effects:
        - Clears database if resetDB is True
        - Sets up NLTK resources if database is reset
        - Processes and stores documents based on file type flags
        - Prints progress information to console
        - Exits if no file types are selected
        
    Process Flow:
        1. Validate that at least one file type is selected
        2. Clear database if requested
        3. Set up language models if needed
        4. Process each selected file type sequentially
        5. Generate embeddings and store in ChromaDB
    """
    # console.print("\t This is populate_databases.main()")
    
    # Validate that at least one file type option is selected
    if not usePDF and not useXML and not useTXT and not useCSV:
        console.print("\t :scream: [bright_green]Nothing to do![/bright_green]")
        console.print(
            "\t [bright_cyan]Note: Use options: [bright_yellow]--usepdf[/bright_yellow], [bright_yellow]--usexml [/bright_yellow], [bright_yellow]--usetxt[/bright_yellow] or [bright_yellow]--usecsv[/bright_yellow]"
        )
        exit()

    console.print(
        f"\t[bright_green] :rocket: Resetting database:[bright_yellow] {resetDB}[/bright_yellow]"
    )

    # Clear existing database if requested for fresh start
    if resetDB:  
        console.print("\t :rocket: [bright_green]Clearing Database[/bright_green]")
        clear_database()
        console.print(
            "\t :sparkles: [bright_cyan]Setting up language models in local directory ...[/bright_cyan]"
        )
        setupNLTK()  # install language models for text processing tasks

    # Process PDF files if requested
    if usePDF:
        # print(f"  +++ Using option : usePDF: {usePDF}")

        # Create (or update) the data store.
        try:
            documentsPDF_list = load_documents_PDF()
        except Exception:
            console.print(
                "\n:scream:[bright_red] Error: Unexpected issue with PDF files in the data/ directory.\n\t Please check your files and try again.[/bright_red]\n"
            )
            exit()
        # print(f"use pdf; documents: {documentsPDF_list}, {type(documentsPDF_list)}") # returns a list
        #   input("Current Chunk above. Press any key to continue!!")
        chunks = split_documents(documentsPDF_list)
        add_to_chroma(chunks, myModel)
        # console.print("\t :smiley: Populating complete")

    # Process XML documents if the useXML flag is set.

    if useXML:
        # print(f"  +++ Using option : useXML: {useXML}")
        try:
            documentsXML_list = load_documents_NXML()
        except Exception:
            console.print(
                "\n:scream:[bright_red] Error: Unexpected issue with NXML files in the data/ directory.\n\t Please check your files and try again.[/bright_red]\n"
            )
            exit()
        # console.print(f"[cyan]main() documentsXML_list : [bright_yellow]{documentsXML_list}[/bright_yellow]") #list
        for i in range(len(documentsXML_list)):
            chunks = split_documents(documentsXML_list[i])
            # print(f"{i}: useXML chunks : {chunks}")
            #   input("Current Chunk above. Press any key to continue!!")
            add_to_chroma(chunks, myModel)
    # Process TXT documents if the useTXT flag is set.

    if useTXT:
        # print(f"  +++ Using option : useTXT: {useTXT}")
        try:
            documentsTXT_list = load_documents_TEXT()
        except Exception:
            console.print(
                "\n:scream:[bright_red] Error: Unexpected issue with TXT files in the data/ directory.\n\t Please check your files and try again.[/bright_red]\n"
            )
            exit()
        # console.print(f"[cyan]main() documentsXML_list : [bright_yellow]{documentsXML_list}[/bright_yellow]") #list
        for i in range(len(documentsTXT_list)):
            chunks = split_documents(documentsTXT_list[i])
            # print(f"{i}: useXML chunks : {chunks}")
            #   input("Current Chunk above. Press any key to continue!!")
            add_to_chroma(chunks, myModel)

    # Process CSV documents if the useCSV flag is set.

    if useCSV:
        # print(f"  +++ Using option : useCSV: {useCSV}")
        try:
            documentsCSV_list = load_documents_CSV()
        except Exception:
            console.print(
                "\n:scream:[bright_red] Error: Unexpected issue with CSV files in the data/ directory.\n\t Please check your files and try again.[/bright_red]\n"
            )
            exit()

        for document in documentsCSV_list:
            chunks = split_documents(document)
            chunks_with_ids = calculate_chunk_ids(chunks)
            add_to_chroma(chunks_with_ids, myModel)

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
    # print("This is load_documents_TEXT()")

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
        )  # open, parse txt documents
        docs = loader.load()
        myArticles_list.append(docs)
    return myArticles_list


# end of load_documents_TEXT()


def load_documents_CSV():
    """A function to load the xml files, to produce document objects to store in a dictionary; key:filename, value: document object."""
    # print("This is load_documents_CSV()")

    myArticles_list = []  # list to contain all article documents
    # find which files we are working with
    myFiles_list = get_files_list_from_directory(
        DATA_PATH, "csv"
    )  # length is the number of articles found
    # print(f"myFiles_list: myFiles_list, LENGTH = {len(myFiles_list)}")
    for thisFile in myFiles_list:
        currentFile_str = thisFile
        console.print(
            f"\t [bright_cyan]Current File :[bright_yellow] {currentFile_str}[/bright_yellow]"
        )
        loader = TextLoader(
            currentFile_str,
        )  # open, parse csv documents
        docs = loader.load()
        myArticles_list.append(docs)
    return myArticles_list


# end of load_documents_CSV()


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


def enhanced_split_documents(documents: list[Document], use_enhanced: bool = False) -> list[Document]:
    """Split documents using enhanced or traditional methods."""
    if use_enhanced and ENHANCED_PROCESSING_AVAILABLE:
        console.print("[cyan]Using enhanced document processing with multiple chunking strategies[/cyan]")
        processor = EnhancedDocumentProcessor()
        
        # Use multiple strategies for better coverage
        chunks = processor.process_documents_with_multiple_strategies(
            documents, 
            strategies=["semantic", "hierarchical", "adaptive"]
        )
        
        # Deduplicate similar chunks
        chunks = processor.deduplicate_chunks(chunks)
        
        return chunks
    else:
        if use_enhanced and not ENHANCED_PROCESSING_AVAILABLE:
            console.print("[yellow]Enhanced processing not available, using standard chunking[/yellow]")
        return split_documents(documents)


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
            console.print(
                "\t []bright_green]✅ No new documents to add.[/bright_green]"
            )

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
