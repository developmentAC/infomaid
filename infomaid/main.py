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


# globals
OUTPUTDIR = "0_out/"  # output directory

console = Console()
cli = typer.Typer()


@cli.command()
def main(
    bighelp: bool = typer.Option(default=False, help="Get Commonly used commands."),
    count: int = 1,
    promptfile: Path = typer.Option(default=None, help="Give your prompt as a file."),
    model: str = typer.Option(
        default="mistral",
        help="General model. First time install --> ollama pull mistral",
    ),
    pdfmodel: str = typer.Option(
        default="nomic-embed-text",
        help="PDF model. First time install -->  ollama pull nomic-embed-text",
    ),
    prompt: str = typer.Option(default="", help="Give your prompt as a line."),
    usepdfdata: bool = typer.Option(
        default=False, help="Chat with pdfs data (not general generative AI)?"
    ),
    resetdb: bool = typer.Option(
        default=False, help="Reset the database for use with new data?"
    ),
) -> None:
    # """Driver of the program."""

    # bighelp?
    if bighelp:
        getBigHelp()
        exit()

    # reset database
    if resetdb == True:  # command to populate database with pdf
        # console.print("\t :sparkles: Resetting database ...")
        populate_database.main(resetdb, pdfmodel)
        console.print("\t :smiley: Populating complete")
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
                "\t[green] What kind of AI help do you need? [/green] :"
            )
            if len(seed) == 0:  # nothing entered
                console.print("\t[red] Nothing entered. Exiting ...[/red]")
                raise typer.Abort()
            prompt = seed
    else:
        if promptfile.is_file():
            seed = promptfile.read_text()
            console.print(f"The data file that contains the input is: {seed}")
            prompt = seed
        else:
            console.print(f"\t:scream: Bad filename entered")
            raise typer.Abort()
    console.print(f"\t [cyan] Code prompt:\n\t[/cyan] [yellow]  {prompt}[/yellow]")
    console.print(f"\t [cyan] Model: {model}[/cyan]")
    console.print(f"\t [cyan] Number of stories to create: {count}[/cyan]")
    if len(prompt) > 0:
        if not usepdfdata:
            for i in range(int(count)):
                myStory = tellStory(prompt, model)  # us ollama to generate the story.
                saveFile(myStory, model)  # Save the story as a file

        if usepdfdata:  # chat with pdfs
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
    randomColour = random.choice(["green", "blue", "red", "yellow", "cyan"])
    console.print(f"\t[{randomColour}]{banner}[/{randomColour}]")

    # Banner art: https://manytools.org/hacker-tools/ascii-banner/

    console.print("\t + Reset and build the internal data of pdf data to use.")
    console.print("\t -> [green] poetry run infomaid --resetdb [/green]")

    console.print("\t + Use pdfs as data source. Ask me for the prompt.")
    console.print("\t -> [green] poetry run infomaid --usepdfdata [/green]")

    console.print(
        "\t + Use general chat, give me two results to consider, not using pdf data"
    )
    console.print(
        '\t -> [green] poetry run infomaid --count 2 --prompt "describe four breeds of dogs" [/green]'
    )

    console.print("\t + Query pdfs with supplied prompt and provide two outputs")
    console.print(
        '\t -> [green] poetry run infomaid --count 2 --usepdfdata --prompt "what is the main idea of the article?" [/green]'
    )
    console.print("\t + Ask a silly question!")
    console.print(
        '\t -> [green] poetry run infomaid --count 2 --prompt "name four shapes" [/green]'
    )
    console.print(
        "\t + Use the prompt details of the supplied file for generative AI results"
    )
    console.print(
        "\t -> [green] infomaid poetry run infomaid --promptfile promptFiles/tell_me_a_joke.txt [/green]"
    )


# end of bighelp()


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
    """Generate the story by submitting the seed to the ollama AI app."""
    # storySeed = "Please,  " + storySeed # if we need to add details to prompt...

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
            "\t  :scream: [red]There appears to be no connection to Ollama. Is Ollama client running?[/red]\n\t  Exiting... "
        )
        exit()
    # response = formatOutput(storySeed, response)
    myStory = f"# infomaid\n\n## Prompt: {storySeed}\n\n## Story\n {response['message']['content']}\n\n"
    return myStory


# end of tellStory()


def formatOutput(storySeed: str, response) -> None:
    # console.print(f"[yellow]formatOutput()[/yellow]") # for debugging
    console.print(
        f"[green]\t  STORYSEED: {storySeed}\n\t  RESPONSE: {response}[/green]"
    )
    myStory = f"# infomaid\n\n## Prompt: {storySeed} \n\n## Story\n {response}\n\n"
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
