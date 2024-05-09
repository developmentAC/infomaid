# Infomaid: An AI and RAG-Enabled Learning Application

![logo](graphics/infomaid_logo_yellow.png)

Date: 7 May 2024

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
+ [Working with PDF Data](#working-with-pdf-data)
+ [Sample Project](#sample-project)
+ [Ethical Note](#ethical-note)
+ [A Work In Progress](#a-work-in-progress)


## Overview

_Infomaid_ is a simple AI prompt-based solution with built in Retrieval augmented generation (RAG) support!


Welcome to this simple AI application! _Infomaid_ is an experimental AI prompt-driven solution to help complete work with information. The software runs locally, without the need to send information to another machine online. This application requires [Ollama](https://www.ollama.com/), for service. Parts of this project code for working with PDFs were borrowed from Pixegami's RAG tutorial at [Reference](https://github.com/pixegami/rag-tutorial-v2). Much thanks!

## Prerequisites

Before you start, make sure you have the following softwares have been installed:

+ Special note: This project was developed on the MacOS and Linux but has not been tested on Windows. 

+ Python
+ [All about Python](https://www.python.org)
+ Note: This project was created with Python V3.11. Earlier versions may also be compatible with this project. If you are using an earlier version of Python, then you will have to modify the `pyproject.toml` file from the project.

+ To change from Python v3.11 to v3.10, we make the following change in `pyproject.toml`. version change,

Originally:
``` toml
python = "^3.11"
```

After editing:
``` toml
python = "^3.10"
```

+ Check to see if you already have Python installed on your machine before installing a new version.

+ Poetry

+ [Instructions to install Poetry](https://python-poetry.org/docs/#installation)

+ Ollama
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

Use online help to help you to remember how to use the parameters.

``` bash
poetry run infomaid --bighelp
```

The output is a list of commands that the user may edit to to work with the project.

Output:

+ Reset and build the internal data of pdf data.

``` bash
poetry run infomaid --resetdb
```

+ Use pdfs as data source.

``` bash
poetry run infomaid --usepdfdata
```

+ Use general chat, no pdf data

``` bash
poetry run infomaid --usepdfdata
```

+ Query pdfs with prompt. provide two outputs

``` bash
poetry run infomaid --count 2 --usepdfdata --prompt "what is the article's main idea?"
```

+ Ask a silly question!

``` bash
poetry run infomaid --count 2 --prompt "name four shapes"
```

## Execution
### Parameters

+ bighelp - Provides CLI commands to use the project.
+ count - The number of results to give. It is sometimes a good idea to have several results from which to choose as not all output is the same.
+ promptfile - The parameter to load a text file in which a complicated prompt is provided. The text file can be used to describe the prompt for the outputs using PDF data, in addition to the regular generative output.
+ model - If the user would like to use a model different from the default, `mistral`, then this parameter may be used to load that model. Note: the model must first be `pull`ed to the local machine. The command to pull a specific model is `ollama pull [myModel]`.
+ pdfmodel - Use a specific model other than the default `nomic-embed-text` to work with the PDF data.
+ prompt - the initial piece of information to instruct the model.
+ usepdfdata - Use data gained from the PDF documents.
+ resetdb - Each time new PDF documents are placed in the `data/` directory, their content must be used to build a local dataset to query them. Use this parameter to clear out the former PDF content and to update the dataset with the new PDFs.

### Generation

With _Infomaid_, users may ask the AI to prepare information from prompts such as outlines, emails, and other types of information. Requests can be made with a prompt that may be entered at the command line, inputted after execution, or entered as a text file. The text file may contain large prompts where there are lots of details to consider. In addition, the text file may help to automate jobs where the prompt is created automatically by another task.

### Working with PDF Data

_Infomaid_ also allows the user to interact with PDF documents to search for ideas which are contained (somewhere) in the documents. _Retrieval Augmented Generation_, or (RAG), is a natural language processing (NLP) technique that harnesses information retrieval from documents for the delivery of generated information through the use of generative-based artificial intelligence (AI) models.

#### Sample Project

For instance, imagine that the user wishes to create a draft of a recommendation letter for someone (i.e., a student) who has supplied a current curriculum vitae (CV) as a PDF document. Using _Infomaid_, a draft of the letter may be written that has been informed by the CV. To use this RAG functionality in this project, the command line parameter, `--usepdfdata`, must be utilized to execute the program. See the project's online help for a sample bash command line scrip to engage the RAG feature.

A prompt for such a task would be the following;

``` text
Write a letter of recommendation for MIT graduate school for Oliver Bonham-Carter. Use the details from the data to complete the draft.
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
poetry run infomaid --count 3 --usepdfdata --prompt "Write a letter of recommendation for MIT graduate school for Oliver Bonham-Carter. Use the details from the data to complete the draft."
```

Same command using the `--promptfile FILE.TXT` parameter

``` bash
poetry run infomaid --count 3 --usepdfdata --promptfile input/mit.txt
```


Command Output:
``` bash
Code prompt: Write a letter of recommendation for MIT graduate school for
Oliver Bonham-Carter. Use the details from the data to complete the draft.
Model: mistral
Number of stories to create: 3
```

The results are Markdown files that will appear in the `0_out/` directory.

``` text
Dear Admissions Committee,

I am writing this letter in strong support of Oliver Bonham-Carter's application to your esteemed graduate program at Massachusetts Institute of Technology (MIT). I have had the pleasure of working with Oliver on various research projects and collaborations over the past decade. His exceptional academic achievements, dedication, and innovative spirit make him an ideal candidate for your program.

...

In conclusion, I wholeheartedly recommend Oliver Bonham-Carter for your graduate program at MIT. His exceptional academic achievements, diverse research expertise, dedication to education, and innovative spirit make him an excellent candidate for your esteemed institution. Should you require any further information, please do not hesitate to contact me.
```

(Or whatever!!)

## Ethical Note

While there is a lot of convenience in using AI to prepare drafts of letters and other communications. In the realm of AI-driven automation, having a human presence to preside over the textual (or graphical work) generated by artificial intelligence is extremely important. While AI systems excel at processing vast amounts of data and executing tasks with remarkable efficiency, they lack the nuanced understanding and ethical judgment inherent to human cognition, in addition to the sense of ethics that ought to come from the human world).

Involving ethics in decisions where machines have made the choices (as strange as that may seem) is essential in domains involving communication. Human oversight ensures that communications, whether they involve customer interactions, inter-office correspondence, or public statements, adhere to ethical standards, tone, and context sensitivity. Moreover, decisions influenced by AI algorithms must be subjected to human judgment before implementation. Human evaluators can consider broader implications, ethical ramifications, and potential biases that AI might overlook, thus safeguarding against unintended consequences and ensuring alignment with organizational values and societal norms. Ultimately, the fusion of AI automation with human oversight represents a symbiotic relationship, harnessing the strengths of both to navigate complex challenges and foster responsible innovation in the digital age.

With this in mind, the _Infomaid_ project must be used responsibly. The project is to serve educational purposes -- it is to instruct on the uses of AI, allow for discovery and to entertain (in a way!). Please use _Infomaid_ responsibly.

---

## A Work In Progress

Check back often to see the evolution of the project!! _Infomaid_ is a work-in-progress. Updates will come periodically.

If you would like to contribute to this project, __then please do!__ For instance, if you see some low-hanging fruit or task that you could easily complete, that could add value to the project, then I would love to have your insight.

Otherwise, please create an Issue for bugs or errors. Since I am a teaching faculty member at Allegheny College, I may not have all the time necessary to quickly fix the bugs. I welcome the OpenSource Community to further the development of this project. Much thanks in advance. :-)