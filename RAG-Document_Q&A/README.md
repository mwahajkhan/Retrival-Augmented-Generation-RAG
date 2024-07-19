## PDF Question Answering Project

# Overview

This project leverages Retrieval-Augmented Generation (RAG) to enable specific question answering from any given PDF document. By integrating three different models—Mixtral, Llama2, and GPT-3 Turbo—the project contrasts their outputs to highlight their performance differences. The document processing and embedding are handled using LangChain's DocumentArray, which splits the PDF into chunks for efficient querying.

## Features

* Multi-Model Support: Utilizes Mixtral, Llama2, and GPT-3 Turbo for diverse model outputs.
* Document Embedding: Uses DocumentArray to embed documents divided into chunks.
* Question Answering: Allows asking specific questions about the content of a PDF.
* Local Vector Store: Embeds and stores document chunks locally using DocArrayInMemorySearch.
* Installation
* bash

* Copy code

pip install langchain langchain_openai langchain_community pypdf docarray
