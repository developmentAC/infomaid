#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Infomaid Main CLI Module

This module provides the command-line interface for Infomaid, an AI-powered tool
that combines generative AI with Retrieval Augmented Generation (RAG) capabilities.

Key Features:
- Generative AI chat using Ollama models
- RAG functionality with local document databases
- Enhanced RAG with TF-IDF, BM25, and hybrid search methods
- Support for multiple document formats (PDF, XML, TXT, CSV)
- Rich console output with color formatting

Author: Oliver Bonham-Carter
Project: https://github.com/developmentAC/infomaid
License: Open source
"""

import ollama
from rich.console import Console
import typer
import os
from pathlib import Path
import random
import sys

# Import internal modules
from infomaid import populate_database
from infomaid import query_data

# External reference acknowledgments
# Some code associated with reading PDF documents taken from;
# ref: https://github.com/pixegami/rag-tutorial-v2/tree/main
# ref: https://github.com/ollama/ollama-python

# Project constants
INFOMAID_WEB = "https://github.com/developmentAC/infomaid"
OUTPUTDIR = "0_out/"  # Directory for storing generated output files

# Initialize Rich console for formatted output
console = Console()

def show_help():
    """
    Display comprehensive help information for the CLI.
    
    This function provides a manual help implementation to work around
    compatibility issues with Typer's Rich formatting in certain environments.
    It displays all available command-line options with descriptions and defaults.
    
    Note: This is a workaround for Typer Rich formatting issues that cause
    TypeError in the standard --help implementation.
    """
    help_text = """
Usage: infomaid [OPTIONS]

Infomaid: AI prompt-based solution with built-in RAG support!

Options:
  --bighelp                     Get commonly used commands
  --count INTEGER              Number of results to get from the prompt [default: 1]
  --promptfile PATH            Give your prompt as a file
  --model TEXT                 General model. First time install --> ollama pull mistral [default: mistral]
  --pdfmodel TEXT              PDF model. First time install --> ollama pull nomic-embed-text [default: nomic-embed-text]
  --prompt TEXT                Give your prompt as a line [default: ]
  --useowndata / --no-useowndata  Chat with nxml or pdfs data that was used to build the project's database [default: no-useowndata]
  --resetdb / --no-resetdb     Reset the database for use with new data? [default: no-resetdb]
  --usepdf / --no-usepdf       Use PDF documents to populate the database? [default: no-usepdf]
  --usexml / --no-usexml       Use XML documents to populate the database? [default: no-usexml]
  --usetxt / --no-usetxt       Use TXT documents to populate the database? [default: no-usetxt]
  --usecsv / --no-usecsv       Use CSV documents to populate the database? [default: no-usecsv]
  --enhancedrag / --no-enhancedrag  Use enhanced RAG with TF-IDF, BM25, and hybrid search methods? [default: no-enhancedrag]
  --retrievalmethod TEXT       Retrieval method: vector, tfidf, hybrid, or bm25 [default: hybrid]
  --help                       Show this message and exit
"""
    console.print(help_text)

# Pre-process help arguments to avoid Typer Rich compatibility issues
# This check intercepts --help/-h before Typer can process them
if len(sys.argv) > 1 and ('--help' in sys.argv or '-h' in sys.argv):
    show_help()
    sys.exit(0)

# Initialize Typer application with basic configuration
# Rich formatting is avoided to prevent compatibility issues
app = typer.Typer(help="Infomaid: AI prompt-based solution with built-in RAG support!")

@app.command()
def main(
    # Core functionality options
    bighelp: bool = typer.Option(default=False, help="Get Commonly used commands."),
    count: int = typer.Option(
        default=1, help="Number of results to get from the prompt."
    ),
    promptfile: Path = typer.Option(default=None, help="Give your prompt as a file."),
    
    # Model configuration options
    model: str = typer.Option(
        default="mistral",
        help="General model. First time install --> ollama pull mistral",
    ),
    pdfmodel: str = typer.Option(
        default="nomic-embed-text",
        help="PDF model. First time install --> ollama pull nomic-embed-text",
    ),
    prompt: str = typer.Option(default="", help="Give your prompt as a line."),
    
    # Data source and operation mode options
    useOwnData: bool = typer.Option(
        default=False,
        help="Chat with nxml or pdfs data that was used to build the project's database (not generic generative AI)?",
    ),
    resetdb: bool = typer.Option(
        default=False, help="Reset the database for use with new data?"
    ),
    
    # Document format options for database population
    usePDF: bool = typer.Option(
        default=False, help="Use PDF documents to populate the database?"
    ),
    useXML: bool = typer.Option(
        default=False, help="Use XML documents to populate the database?"
    ),
    useTXT: bool = typer.Option(
        default=False, help="Use TXT documents to populate the database?"
    ),
    useCSV: bool = typer.Option(
        default=False, help="Use CSV documents to populate the database?"
    ),
    
    # Enhanced RAG options
    enhancedRAG: bool = typer.Option(
        default=False, help="Use enhanced RAG with TF-IDF, BM25, and hybrid search methods?"
    ),
    retrievalMethod: str = typer.Option(
        default="hybrid", help="Retrieval method: vector, tfidf, hybrid, or bm25"
    ),
) -> None:
    """
    Main CLI function that orchestrates all Infomaid operations.
    
    This function serves as the central dispatch point for all CLI operations,
    handling everything from simple AI chat to complex RAG queries with
    multiple retrieval strategies.
    
    Args:
        bighelp: Show extended help with examples
        count: Number of AI responses to generate
        promptfile: File containing the prompt text
        model: Ollama model for general text generation
        pdfmodel: Ollama model for document embeddings
        prompt: Direct prompt text input
        useOwnData: Enable RAG mode with local document database
        resetdb: Reset and rebuild the document database
        usePDF/useXML/useTXT/useCSV: Document format flags for database population
        enhancedRAG: Enable advanced RAG with multiple retrieval methods
        retrievalMethod: Specific retrieval algorithm to use
        
    Returns:
        None: Function handles all output and file operations internally
    """

    # Handle extended help display
    if bighelp:
        getBigHelp()
        exit()

    # Handle database reset and population
    # This rebuilds the vector database with new documents
    if resetdb == True:
        populate_database.main(resetdb, pdfmodel, usePDF, useXML, useTXT, useCSV)
        exit()
    
    # Input handling: Determine the prompt source
    # Priority: promptfile > direct prompt > interactive input
    seed = None
    if promptfile is None:
        # No file specified, check for direct prompt or request input
        if prompt:
            # Direct prompt provided via --prompt option
            seed = prompt
        else:
            # Interactive mode: request prompt from user
            seed = console.input(
                "\t[bright_green] What kind of AI help do you need? [/bright_green] :"
            )
            if len(seed) == 0:  # User entered nothing
                console.print(
                    "\t[bright_red] Nothing entered. Exiting ...[/bright_red]"
                )
                raise typer.Abort()
            prompt = seed
    else:
        # Handle file-based prompt input
        if promptfile.is_file():
            seed = promptfile.read_text()
            console.print(
                f"\t [bright_green]The data file that contains the input is: {seed}[bright_green]"
            )
            prompt = seed
        else:
            console.print(f"\t:scream: Bad filename entered")
            raise typer.Abort()
    
    # Display operation summary to user
    console.print(
        f"\t [bright_cyan] Code prompt:\n\t[bright_yellow]  {prompt}[/bright_yellow]"
    )
    console.print(f"\t [bright_cyan] Model: {model}[/bright_cyan]")
    console.print(
        f"\t [bright_cyan] Number of stories to create: {count}[/bright_cyan]"
    )
    
    # Execute main operations based on configuration
    if len(prompt) > 0:
        if not useOwnData:
            # Standard generative AI mode (no RAG)
            # Generate responses using pure AI without document context
            for i in range(int(count)):
                myStory = tellStory(prompt, model)  # Generate AI response
                saveFile(myStory, model)  # Save to output file

        if useOwnData:  # RAG mode with local document database
            # Process queries using document context for enhanced responses
            for i in range(int(count)):
                if enhancedRAG:
                    # Use advanced RAG with multiple retrieval strategies
                    console.print(f"\t [cyan] Using enhanced RAG with {retrievalMethod} retrieval[/cyan]")
                    myStory = query_data.main(prompt, pdfmodel, use_enhanced=True, retrieval_method=retrievalMethod)
                else:
                    # Use standard vector-based RAG
                    myStory = query_data.main(prompt, pdfmodel)
                myStory = formatOutput(prompt, myStory)  # Format for markdown output
                saveFile(myStory, model)  # Save formatted response

# End of main() function


def getBigHelp() -> None:
    """
    Display comprehensive help with practical examples and ASCII banner.
    
    This function provides an extended help interface that shows users
    how to use various Infomaid features with real command examples.
    It includes an ASCII art banner and categorized usage examples
    for different operation modes.
    
    Returns:
        str: Returns "getBigHelp" for testing purposes
    """
    # Display project description and branding
    console.print(
        "\t :sparkles: _Infomaid_ is a simple AI prompt-based solution with built in Retrieval augmented generation (RAG) support!"
    )
    
    # ASCII art banner for visual appeal
    # Banner art source: https://manytools.org/hacker-tools/ascii-banner/
    banner = """\n
\t██╗███╗   ██╗███████╗ ██████╗ ███╗   ███╗ █████╗ ██╗██████╗ 
\t██║████╗  ██║██╔════╝██╔═══██╗████╗ ████║██╔══██╗██║██╔══██╗
\t██║██╔██╗ ██║█████╗  ██║   ██║██╔████╔██║███████║██║██║  ██║
\t██║██║╚██╗██║██╔══╝  ██║   ██║██║╚██╔╝██║██╔══██║██║██║  ██║
\t██║██║ ╚████║██║     ╚██████╔╝██║ ╚═╝ ██║██║  ██║██║██████╔╝
\t╚═╝╚═╝  ╚═══╝╚═╝      ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═════╝ 

"""
    # Randomize banner color for visual variety
    randomColour = random.choice(
        ["bright_green", "bright_blue", "bright_red", "bright_yellow", "bright_cyan"]
    )
    console.print(f"\t[{randomColour}]{banner}\t{INFOMAID_WEB}\n[/{randomColour}]")

    # Basic AI chat examples
    console.print("\t + Ask a silly question of generative AI!")
    console.print(
        '\t\t-> [bright_green] poetry run infomaid --prompt "name four shapes" [/bright_green]'
    )

    console.print(
        "\t + Use general chat, give me two results to consider, not using pdf data"
    )
    console.print(
        '\t\t-> [bright_green] poetry run infomaid --count 2 --prompt "describe four breeds of dogs" [/bright_green]'
    )

    # Database setup and management examples
    console.print(
        "\t + Reset and build own model trained model with local data,\n\t\t use --usepdf or ---usexml options."
    )
    console.print("\t\t-> [bright_green] poetry run infomaid --resetdb [/bright_green]")

    console.print("\t   * Reset and build own model trained model with PDF files.")
    console.print(
        "\t\t-> [bright_green] poetry run infomaid --resetdb --usepdf [/bright_green]"
    )

    console.print("\t   * Reset and build own model trained model with XML files.")
    console.print(
        "\t\t-> [bright_green] poetry run infomaid --resetdb --usexml [/bright_green]"
    )

    console.print("\t   * Reset and build own model trained model with TXT files.")
    console.print(
        "\t\t-> [bright_green] poetry run infomaid --resetdb --usetxt [/bright_green]"
    )

    console.print("\t   * Reset and build own model trained model with CSV files.")
    console.print(
        "\t\t-> [bright_green] poetry run infomaid --resetdb --usecsv [/bright_green]"
    )

    # RAG query examples
    console.print(
        "\t + Use own model trained model as data source. Ask me for the prompt."
    )
    console.print(
        "\t\t-> [bright_green] poetry run infomaid --useowndata [/bright_green]"
    )

    console.print(
        "\t + Query own model trained model with supplied prompt and provide output."
    )
    console.print(
        '\t\t-> [bright_green] poetry run infomaid --useowndata --prompt "Whose name is on the included CV?" [/bright_green]'
    )
    
    # Enhanced RAG examples with multiple retrieval methods
    console.print(
        "\t + Use enhanced RAG with TF-IDF, BM25, and hybrid retrieval methods."
    )
    console.print(
        '\t\t-> [bright_green] poetry run infomaid --useowndata --enhancedrag --retrievalmethod hybrid --prompt "Your question here" [/bright_green]'
    )
    
    console.print(
        "\t + Available retrieval methods: vector, tfidf, hybrid, bm25"
    )
    console.print(
        '\t\t-> [bright_green] poetry run infomaid --useowndata --enhancedrag --retrievalmethod bm25 [/bright_green]'
    )
    
    # File-based prompt examples
    console.print(
        "\t + Use the prompt details of the supplied file for generative AI results"
    )
    console.print(
        "\t\t-> [bright_green] poetry run infomaid --promptfile promptFiles/tell_me_a_joke.txt [/bright_green]"
    )
    return "getBigHelp"  # Return value for testing purposes


# End of getBigHelp() function


def getNumber(myModel) -> int:
    """
    Manage sequential numbering for output files.
    
    This function maintains a persistent counter for each model to ensure
    unique filenames for generated content. It creates or updates an index
    file that tracks the current count for each model type.
    
    Args:
        myModel (str): The model name used to create model-specific counter files
        
    Returns:
        int: The current sequential number for this model
        
    File Format:
        Creates files like: "0_out/.mistral_currentStoryIndex.txt"
        Contains a single integer representing the current count
    """
    # Construct model-specific index filename
    filename = OUTPUTDIR + "." + myModel + "_currentStoryIndex.txt"
    number = 0

    # Check if index file already exists
    if os.path.exists(filename):
        # Read existing count and increment
        with open(filename, "r") as f:
            content = f.read()
            
            # Increment and update the counter
            with open(filename, "w") as f:
                try:
                    newNumber = int(content) + 1
                    f.write(str(newNumber))
                    console.print(
                        f"\t [green] Updated story index file [/green] {newNumber}"
                    )
                except ValueError:
                    # Handle corrupted counter files
                    console.print(
                        "\t [red] Problem with number in file. Resetting counter.[/red]"
                    )
                    with open(filename, "w") as f:
                        f.write(str(0))

    else:
        # Create new index file starting at 0
        with open(filename, "w") as f:
            f.write(str(number))
            console.print(
                f"\t [yellow] Number {number} written to the newly created file.[/yellow]"
            )
    
    # Read and return the current number
    with open(filename, "r") as f:
        content = f.read()
        console.print(
            f"\t [purple] The current number by the file is :{content}[/purple]"
        )
        return int(content)

# End of getNumber() function


def tellStory(storySeed: str, myModel: str) -> str:
    """
    Generate AI-powered content using Ollama.
    
    This function interfaces with the Ollama API to generate text responses
    based on user prompts. It handles connection errors and formats the
    output as structured Markdown for consistent presentation.
    
    Args:
        storySeed (str): The user's prompt/question for the AI
        myModel (str): The Ollama model name to use for generation
        
    Returns:
        str: Formatted Markdown string containing the AI response
        
    Raises:
        SystemExit: If Ollama connection fails or model is unavailable
        
    Output Format:
        - Markdown document with headers
        - Project branding and links
        - User prompt and AI response sections
    """
    try:
        # Make API call to Ollama
        response = ollama.chat(
            model=myModel,
            messages=[
                {
                    "role": "user",
                    "content": storySeed,
                },
            ],
        )
    except Exception:
        # Handle connection or model errors gracefully
        console.print(
            "\t  :scream: [red]There appears to be no connection to Ollama or to a model. Is Ollama client running? Has the model been loaded?\n\t  Try this command: ollama pull name-your-model \n\t  Exiting... [/red]"
        )
        exit()
    
    # Format response as structured Markdown
    myStory = f"# Infomaid\n\n{INFOMAID_WEB}\n\n## Prompt\n\n{storySeed}\n\n## Story\n\n {response['message']['content']}\n\n"
    return myStory

# End of tellStory() function


def formatOutput(storySeed: str, response: str) -> str:
    """
    Format RAG query results for consistent Markdown output.
    
    This function takes RAG query results and formats them into a standardized
    Markdown structure that matches the format used for direct AI queries.
    Used specifically for responses that include document context.
    
    Args:
        storySeed (str): The original user prompt/question
        response (str): The RAG-enhanced AI response
        
    Returns:
        str: Formatted Markdown string with branding and structured content
        
    Side Effects:
        Prints formatted output to console for immediate user feedback
    """
    # Display formatted output to user
    console.print(
        f"[green]\t  STORYSEED: {storySeed}\n\t  RESPONSE: {response}[/green]"
    )
    # Create formatted Markdown output matching tellStory format
    myStory = f"# Infomaid\n\n{INFOMAID_WEB}\n\n## Prompt\n\n {storySeed} \n\n## Story\n\n {response}\n\n"
    return myStory

# End of formatOutput() function


def checkDataDir(dir_str: str) -> int:
    """
    Verify output directory exists and create if necessary.
    
    This utility function ensures that the output directory structure
    is available before attempting to save files. It creates directories
    as needed and provides user feedback about the operation.
    
    Args:
        dir_str (str): Path to the directory that should exist
        
    Returns:
        int: 1 if directory was created, 0 if it already existed
        
    Side Effects:
        Creates directory structure if it doesn't exist
        Prints status messages to console
    """
    try:
        # Attempt to create directory (will fail if it exists)
        os.makedirs(dir_str)
        console.print(f"\t[red]\t Creating :{dir_str}[/red]")
        return 1
    except OSError:
        # Directory already exists, which is fine
        return 0

# End of checkDataDir() function


def saveFile(inText: str, myModel: str) -> None:
    """
    Save generated content to uniquely named markdown files.
    
    This function handles the complete file saving workflow:
    - Ensures output directory exists
    - Generates unique filenames using sequential numbering
    - Saves content with error handling
    - Provides user feedback about file operations
    
    Args:
        inText (str): The formatted content to save
        myModel (str): Model name used for filename generation
        
    Side Effects:
        Creates/updates directory structure
        Writes files to disk
        Updates sequence counter files
        Prints status messages to console
        
    File Naming:
        Format: "./0_out/myAI_{model_name}_{sequence_number}.md"
        Example: "./0_out/myAI_mistral_5.md"
    """
    # Ensure output directory exists
    checkDataDir(OUTPUTDIR)
    
    # Get next sequential number for this model
    currentStoryNumber = getNumber(myModel)
    
    # Construct unique filename
    fname_str = "./" + OUTPUTDIR + f"myAI_{myModel}_" + str(currentStoryNumber) + ".md"

    console.print(f"\t [bold yellow] Saving the story:[/bold yellow] --> {fname_str}\n")
    try:
        # Write content to file
        with open(fname_str, "w") as f:
            f.write(inText)
    except Exception:
        # Handle file I/O errors gracefully
        console.print(
            "[red]\t There was a problem in saving the file...File may not have been saved. [/red]"
        )
        exit()

# End of saveFile() function


# CLI Application Entry Point
if __name__ == "__main__":
    """
    Application entry point for direct script execution.
    
    (The program starts here!)

    This conditional block ensures that the Typer CLI app is only
    started when the script is run directly, not when imported as a module.
    Essential for proper CLI functionality and testing.
    """
    app()
