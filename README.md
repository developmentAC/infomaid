# Infomaid: An AI and RAG-Enabled Learning Application

![logo](graphics/infomaid_logo_yellow.png)

Date: 4 June 2024

[Oliver Bonham-Carter](https://www.oliverbonhamcarter.com/)

Email: obonhamcarter at allegheny.edu

[![MIT Licence](https://img.shields.io/bower/l/bootstrap)](https://opensource.org/licenses/MIT)

## Contents

+ [Overview](#overview)
+ [Prerequisites](#Prerequisites)
+ [Set Up Local Models for Ollama](#set-up-local-models-for-ollama)
+ [Setting Up The Project](#Setting-up-the-project)
+ [Execution](#Execution)
+ [Parameters](#Parameters)
+ [Generation](#generation)
+ [Output](#output)
+ [Working with PDF Data](#working-with-pdf-data)
+ [Sample Project](#sample-project)
+ [Testing the Code](#testing-the-code)
+ [Ethical Note](#ethical-note)
+ [A Work In Progress](#a-work-in-progress)

## Overview

_Infomaid_ is a simple AI prompt-based solution with built in Retrieval augmented generation (RAG) support!

Welcome to this simple AI application! _Infomaid_ is an experimental AI prompt-driven solution (i.e., each "chat" involves writing a separate prompt to use with a new execution of _Infomaid_) to help complete text-based work with information.

The software runs locally, without the need to send information for processing to another machine online. This application requires [Ollama](https://www.ollama.com/), for service. Parts of this project code for working with PDFs were borrowed from Pixegami's RAG tutorial at [Reference](https://github.com/pixegami/rag-tutorial-v2). Much thanks!

## Prerequisites

Before you start, make sure you have the following softwares installed.

+ Special note: This project was developed on the MacOS (Sonoma). Other OSs such as Windows and Linux have not been tested.

+ Python
  + [All about Python](https://www.python.org)
  + Note: This project was created with Python V3.11. Earlier versions may also be compatible with this project. If you are using an earlier version of Python, then you will have to modify the `pyproject.toml` file from the project.

  + To change from Python v3.10 through v3.13, to just Python v3.9 we make the following change in `pyproject.toml`. Note: This may cause other errors.

  Originally:

  ``` toml
  python = ">=3.10,<3.13" 
  ```

  After editing:

  ``` toml
  python = "^3.9"
  ```

  + Check to see if you already have Python installed on your machine before installing a new version.

  + Poetry: Used to manage the virtual environment and to _frame_ the project as a command line app.

  + [Instructions to install Poetry](https://python-poetry.org/docs/#installation)

+ Ollama: The AI server.
  + [Instructions to install Ollama](https://www.ollama.com/)

### Set Up Local Models for Ollama

The below commands will install the models that Ollama will require to perform its functions. Note, a typical model for Ollama is about 4 GB in size. As there are two models to install, this project will take about 8 GB of space.

``` bash
ollama pull mistral
ollama pull nomic-embed-text
```

## Setting Up the Project

We will use Poetry to manage the virtual environment for the project. Use `install` to download all the necessary packages for your project.

``` bash
poetry install
```

Check that the software is working on your system.

``` bash
poetry run infomaid --help
```

Use online help to help you to remember how to use the parameters. Sample commands are privided to copy and paste with editing.

``` bash
poetry run infomaid --bighelp
```

Some of the types of parameters are the following.

+ Use general chat, give me one result, ask me for a prompt

``` bash
poetry run infomaid --count 1
```

+ Ask a silly question from the command line! Provide two outputs.

``` bash
poetry run infomaid --count 2 --prompt "name four shapes"
```

+ Reset and build the internal data of pdf data.

``` bash
poetry run infomaid --resetdb
```

+ Use pdfs as data source with a provided prompt file (`myPrompt.txt`) in which instructions are provided about what the results should look like.

  + Note: if no prompt is supplied, then the program will stop to ask for one during execution.

``` bash
poetry run infomaid --usepdfdata --promptfile "promptFiles/myPrompt.txt"
```

+ Query pdfs with prompt. Provide two outputs.

``` bash
poetry run infomaid --count 2 --usepdfdata --prompt "what is the article's main idea?"
```

## Execution
### Parameters


+ __bighelp__ - Provides CLI commands to use the project.

+ __count__ - The number of results to give. It is sometimes a good idea to have several results from which to choose as not all output is the same.

+ __promptfile__ - The parameter to load a text file in which a complicated prompt is provided.  The text file can be used to describe the prompt for the outputs using PDF data, in addition to the regular generative output.

+ __model__ - Used to delare a model if different from the default, `mistral` for general usage (i.e. not for RAG using the PDF data).
  + Note: the model must first be `pull`ed to the local machine for usage. The command to pull a specific model is `ollama pull [myModel]`.

+ __pdfmodel__ - Use a specific model other than the default `nomic-embed-text` to work with the PDF data.

+ __prompt__ - The initial set of instructions about what kind of information work to complete using Ollama.

+ __usepdfdata__ - A toggle to specify using RAG support (i.e., PDF data sources). In absence of this parameter at the command line, the _Infomaid_ will not use data from PDF documents.

+ __resetdb__ - PDF documents are placed in the `data/` directory to provide the content to build a dataset with which the user may interacte. To prepare this dataset after documents have been placd in `data/`, the parameter `--resetdb` is used to clear out the former PDF content and to update the dataset with the new PDFs. 
  + Note: Always use this parameter when changing PDF content to avoid informational contamination of the results.

### Generation

With _Infomaid_, users may ask the AI to prepare information from prompts such as outlines, emails, and other types of information. Requests can be made with a prompt that may be entered at the command line, inputted after execution, or entered as a text file. The text file may contain large prompts where there are lots of details to consider. In addition, the text file may help to automate jobs where the prompt is created automatically by another task.

#### Output

All results are placed in the `0_out/` directory. The output files are listed according to the Ollama model that was used to create their content. Automatically created will be a hidden file called  `.mistral_currentStoryIndex.txt` that will keep track of the file indexing system. If the file is removed, then _Infomaid_ may overwrite the existing files. For new jobs on _Infomaid_, it is recommended that `0_out/` be removed, or moved somewhere outside of the project directory, in efforts to organize output by prompt.

### Working with PDF Data

_Infomaid_ also allows the user to interact with PDF documents to search for ideas which are contained (somewhere) in the documents. _Retrieval Augmented Generation_, or (RAG), is a natural language processing (NLP) technique that harnesses information retrieval from documents for the delivery of generated information through the use of generative-based artificial intelligence (AI) models.

#### Sample Project

For instance, imagine that the user wishes to create a draft of a recommendation letter for someone (i.e., a student) who has supplied a current curriculum vitae (CV) as a PDF document. Using _Infomaid_, a draft of the letter may be written that has been informed by the CV. To use this RAG functionality in this project, the command line parameter, `--usepdfdata`, must be utilized to execute the program. See the project's online help for a sample bash command line scrip to engage the RAG feature.

A prompt for such a task would be the following;

``` text
Write a letter of recommendation for MIT graduate school for AstroBill. Use the details from the data to complete the draft.
```

To set up the project, the PDF of the CV must first be copied into the `data/` directory of the project. It is important to note that other non-related PDF documents ought to be removed from this directory to prevent interference with the letter-writing task. The following command is necessary to update the working dataset involving PDFs.

``` bash
poetry run infomaid --resetdb
```

The below output will confirm that the database has been updated with the new PDF-derived information

Output:
``` bash
Resetting database: {resetDB}
Clearing Database
Number of existing documents in DB: 0
Adding new documents: 113
```

Next, the prompt may be introduced with the following command. Note, this command will return three potential letters that may differ in quality.

``` bash
poetry run infomaid --count 3 --usepdfdata --prompt "Write a letter of recommendation for MIT graduate school for AstroBill. Use the details from the data to complete the draft."
```

Same command using the `--promptfile FILE.TXT` parameter

``` bash
poetry run infomaid --count 3 --usepdfdata --promptfile promptFiles/mit.txt
```

Command Output:
``` bash
Code prompt: Write a letter of recommendation for MIT graduate school for AstroBill. Use the details from the data to complete the draft. Model: mistral Number of stories to create: 3
```

The results are Markdown files that will appear in the `0_out/` directory.

``` text
Dear Admissions Committee,

I am writing this letter in the highest regard and with great enthusiasm to recommend a remarkable individual, AstroBill, for your esteemed graduate program at MIT. AstroBill, also known as Bill on planet Zirconia, is a multifaceted genius with an insatiable thirst for knowledge and innovation that sets him apart from his peers.

...

In summary, AstroBill's unique blend of intellect, creativity, adaptability, determination, and compassion make him an outstanding candidate for MIT's graduate program. His diverse skills and experiences will enrich any academic or professional environment, and I have no doubt that he will continue to excel in your esteemed institution.
```

(Or whatever!! Now the letter can be edited to add a _human-touch_ and extra value.)

## Testing the Code

The code may be tested to determine functionality. At present, there are two tests; (1) general execution and (2) to determine whether the querying system if working for pdf data.

To run tests using `pytest` which is already installed, use the below commands.

``` python
poetry install # initialize project
poetry run infomaid --resetdb --usepdf # populate pdf db
poetry run pytest # run tests with the pdf database.
```

## Ethical Note

While there is a lot of convenience in using AI to prepare drafts of letters and other communications, in all this automation, it is important to have a human presence to preside over the generated textual (or graphical work). While AI systems excel at processing vast amounts of data and executing tasks with remarkable efficiency, they lack the nuanced understanding and ethical judgment inherent to human cognition, in addition to the sense of ethics that ought to come from the human world.

Involving ethics in decisions where machines have made the choices (as strange as that may seem) is essential in domains involving communication. Human oversight ensures that communications, whether they involve customer interactions, inter-office correspondence, or public statements, adhere to ethical standards, tone, and context sensitivity. In addition, decisions influenced by AI algorithms must be subjected to human judgment before implementation. Human evaluators can consider broader implications, ethical ramifications, and potential biases that AI systems might overlook. This "human-touch" can therefore help to safeguard against the potential and unintended consequences which may occur at the intersection of data and decision-making, to name one such area.

With this in mind, the _Infomaid_ project must be used responsibly. The project is to serve educational purposes -- it is to instruct on the uses of AI, allow for discovery and to entertain (in a way!). Please use _Infomaid_ responsibly.

---

## A Work In Progress

Check back often to see the evolution of the project!! _Infomaid_ is a work-in-progress. Updates will come periodically.

If you would like to contribute to this project, __then please do!__ For instance, if you see some low-hanging fruit or task that you could easily complete, that could add value to the project, then I would love to have your insight.

Otherwise, please create an Issue for bugs or errors. Since I am a teaching faculty member at Allegheny College, I may not have all the time necessary to quickly fix the bugs. I welcome the OpenSource Community to further the development of this project. Much thanks in advance. 

If you appreciate this project, please consider clicking the project's _Star_ button. :-)
