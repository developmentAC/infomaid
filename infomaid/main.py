#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ollama
from rich.console import Console
import typer
import os
from pathlib import Path
import random

from infomaid import populate_database
from infomaid import query_data

# Some code associated with reading PDF documents taken from;
# ref: https://github.com/pixegami/rag-tutorial-v2/tree/main
# ref: https://github.com/ollama/ollama-python
INFOMAID_WEB = "https://github.com/developmentAC/infomaid"

# globals
OUTPUTDIR = "0_out/"  # output directory

console = Console()
cli = typer.Typer()


@cli.command()
def main(
    bighelp: bool = typer.Option(default=False, help="Get Commonly used commands."),
    count: int = typer.Option(
        default=1, help="Number of results to get from the prompt."
    ),
    promptfile: Path = typer.Option(default=None, help="Give your prompt as a file."),
    model: str = typer.Option(
        default="mistral",
        help="General model. First time install --> ollama pull mistral",
    ),
    pdfmodel: str = typer.Option(
        default="nomic-embed-text",
        help="PDF model. First time install --> ollama pull nomic-embed-text",
    ),
    prompt: str = typer.Option(default="", help="Give your prompt as a line."),
    useOwnData: bool = typer.Option(
        default=False,
        help="Chat with nxml or pdfs data that was used to build the project's database (not generic generative AI)?",
    ),
    resetdb: bool = typer.Option(
        default=False, help="Reset the database for use with new data?"
    ),
    usePDF: bool = typer.Option(
        default=False, help="Use PDF documents to populate the database?"
    ),
    useXML: bool = typer.Option(
        default=False, help="Use XML documents to populate the database?"
    ),
    useTXT: bool = typer.Option(
        default=False, help="Use TXT documents to populate the database?"
    ),
) -> None:
    # """Driver of the program."""

    # bighelp?
    if bighelp:
        getBigHelp()
        exit()

    # reset database
    if resetdb == True:  # command to populate database with pdf
        populate_database.main(resetdb, pdfmodel, usePDF, useXML, useTXT)
        exit()
    #
    seed = None
    if promptfile is None:
        # console.print("No data file specified!")
        if prompt:
            # console.print(f"\t[cyan] Prompt entered : {prompt}[/cyan]")
            seed = prompt
        else:
            seed = console.input(
                "\t[bright_green] What kind of AI help do you need? [/bright_green] :"
            )
            if len(seed) == 0:  # nothing entered
                console.print(
                    "\t[bright_red] Nothing entered. Exiting ...[/bright_red]"
                )
                raise typer.Abort()
            prompt = seed
    else:
        if promptfile.is_file():
            seed = promptfile.read_text()
            console.print(
                f"[bright_green]The data file that contains the input is: {seed}[bright_green]"
            )
            prompt = seed
        else:
            console.print(f"\t:scream: Bad filename entered")
            raise typer.Abort()
    console.print(
        f"\t [bright_cyan] Code prompt:\n\t[bright_yellow]  {prompt}[/bright_yellow]"
    )
    console.print(f"\t [bright_cyan] Model: {model}[/bright_cyan]")
    console.print(
        f"\t [bright_cyan] Number of stories to create: {count}[/bright_cyan]"
    )
    if len(prompt) > 0:
        if not useOwnData:
            for i in range(int(count)):
                myStory = tellStory(prompt, model)  # use ollama to generate the story.
                saveFile(myStory, model)  # Save the story as a file

        if useOwnData:  # chat with pdfs
            for i in range(int(count)):
                myStory = query_data.main(prompt, pdfmodel)  # do a query
                myStory = formatOutput(prompt, myStory)  # format output
                saveFile(myStory, model)  # Save the story as a file


# end of main()


def getBigHelp() -> None:
    """Function to provide extra online help in the form of commands."""
    console.print(
        "\t :sparkles: _Infomaid_ is a simple AI prompt-based solution with built in Retrieval augmented generation (RAG) support!"
    )
    banner = """\n
\t██╗███╗   ██╗███████╗ ██████╗ ███╗   ███╗ █████╗ ██╗██████╗ 
\t██║████╗  ██║██╔════╝██╔═══██╗████╗ ████║██╔══██╗██║██╔══██╗
\t██║██╔██╗ ██║█████╗  ██║   ██║██╔████╔██║███████║██║██║  ██║
\t██║██║╚██╗██║██╔══╝  ██║   ██║██║╚██╔╝██║██╔══██║██║██║  ██║
\t██║██║ ╚████║██║     ╚██████╔╝██║ ╚═╝ ██║██║  ██║██║██████╔╝
\t╚═╝╚═╝  ╚═══╝╚═╝      ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═════╝ 

"""
    randomColour = random.choice(
        ["bright_green", "bright_blue", "bright_red", "bright_yellow", "bright_cyan"]
    )
    console.print(f"\t[{randomColour}]{banner}\t{INFOMAID_WEB}\n[/{randomColour}]")

    # Banner art: https://manytools.org/hacker-tools/ascii-banner/

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
    console.print(
        "\t + Use the prompt details of the supplied file for generative AI results"
    )
    console.print(
        "\t\t-> [bright_green] poetry run infomaid --promptfile promptFiles/tell_me_a_joke.txt [/bright_green]"
    )
    return "getBigHelp"  # for testing


# end of getBigHelp()


def getNumber(myModel) -> int:
    """function to load a number file to get a number to name the current file. The function also checks to see whether the current index file exists, and increments it if it does."""

    # Define the filename and number to be written to the file
    filename = OUTPUTDIR + "." + myModel + "_currentStoryIndex.txt"
    number = 0

    # Check if the file already exists
    if os.path.exists(filename):
        # If it does, open the file in read-only mode and print its content
        with open(filename, "r") as f:
            content = f.read()
            # console.print(f"\t [purple]Current content of the file:{content}, {type(content)}[\purple]")
            # Now, increment the number and save the file.
            with open(filename, "w") as f:
                try:
                    newNumber = int(content) + 1
                    # console.print(f"\t [green]Setting new number : {newNumber}, type(newNumber)[/green]")
                    f.write(str(newNumber))
                    console.print(
                        f"\t [green] Updated story index file [/green] {newNumber}]"
                    )
                except ValueError:
                    print(
                        "\t [red] Problem with number in file. Resetting counter.[/red]"
                    )
                    with open(filename, "w") as f:
                        f.write(str(0))
                        # return 0
        # return content

    else:
        # If it doesn't exist, create a new file and write the number to it
        with open(filename, "w") as f:
            f.write(str(number))
            console.print(
                f"\t [yellow] Number {number} written to the newly created file.[/yellow]"
            )
            # return 0
    # read the file, get the number
    with open(filename, "r") as f:
        content = f.read()
        console.print(
            f"\t [purple] The current number by the file is :{content}[/purple]"
        )
        return int(content)


# end of getNumber()


def tellStory(storySeed: str, myModel) -> str:
    """Generate the story by submitting the seed to the ollama AI app. Format output as Markdown."""

    try:
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
        console.print(
            "\t  :scream: [red]There appears to be no connection to Ollama or to a model. Is Ollama client running? Has the model been loaded?\n\t  Try this command: ollama pull name-your-model \n\t  Exiting... [/red]"
        )
        exit()
    myStory = f"# Infomaid\n\n{INFOMAID_WEB}\n\n## Prompt\n\n{storySeed}\n\n## Story\n\n {response['message']['content']}\n\n"
    return myStory


# end of tellStory()


def formatOutput(storySeed: str, response) -> None:
    """For use with output from own data. This function places output in Markdown formatting."""
    # console.print(f"[yellow]formatOutput()[/yellow]") # for debugging
    console.print(
        f"[green]\t  STORYSEED: {storySeed}\n\t  RESPONSE: {response}[/green]"
    )
    myStory = f"# Infomaid\n\n{INFOMAID_WEB}\n\n## Prompt\n\n {storySeed} \n\n## Story\n\n {response}\n\n"
    return myStory


# end of formatOutput()


def checkDataDir(dir_str):
    """function to determine whether a data output directory exists."""
    # if the directory doesn't exist, then it is created

    try:
        os.makedirs(dir_str)
        # print("  PROBLEM: MYOUTPUT_DIR doesn't exist")
        console.print(f"\t[red]\t Creating :{dir_str}[/red]")
        return 1

    except OSError:
        return 0


# end of checkDataDir()


def saveFile(inText: str, myModel: str) -> None:
    """Function to save the story to a file."""
    checkDataDir(OUTPUTDIR)  # does the output dir exist? if not, create it.
    currentStoryNumber = getNumber(myModel)
    # console.print(f"\t[green] Current story = {currentStoryNumber}[/green]")
    # fname_str = "./" + OUTPUTDIR + "/myCode_"+"mistral_"+str(currentStoryNumber)+".md"
    fname_str = "./" + OUTPUTDIR + f"myAI_{myModel}_" + str(currentStoryNumber) + ".md"

    console.print(f"\t [bold yellow] Saving the story:[/bold yellow] --> {fname_str}\n")
    try:
        f = open(fname_str, "w")
        f.write(inText)
        f.close()
    except Exception:
        console.print(
            "[red]\t There was a problem in saving the file...File may not have been saved. [\red]"
        )
        exit()


# end of saveFile()
