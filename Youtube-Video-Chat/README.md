## YouTube Video Transcription and Contextual Retrieval using RAG

This project transcribes a YouTube video, processes the transcription using LangChain loaders and splitters, stores the resulting document embeddings in a Pinecone vector store, and employs Retrieval-Augmented Generation (RAG) to provide contextual answers based on the video content.


> Environment Setup:

  Configured environment variables and OpenAI API key.
  
> Model Setup:

  Used OpenAI's GPT-3.5-turbo for text processing and generation.
  
Transcription:

Downloaded and transcribed the YouTube video using OpenAI's Whisper model.
Document Processing:

Split the transcription into manageable chunks using LangChain tools.
Embedding Generation:

Generated embeddings for the text chunks to enable similarity searches.
Vector Store Setup:

Stored embeddings in an in-memory vector store for efficient retrieval.
Retrieval-Augmented Generation (RAG):

Created a system that combines retrieval and generation to answer questions based on the video content.
Pinecone Integration:

Used Pinecone for scalable vector storage and efficient similarity searches.


# Initial requirements to setup 

## Setup

1. Create a virtual environment and install the required packages:

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

2. Create a free Pinecone account and get your API key from [here](https://www.pinecone.io/).

3. Create a `.env` file with the following variables:

```bash
OPENAI_API_KEY = [ENTER YOUR OPENAI API KEY HERE]
PINECONE_API_KEY = [ENTER YOUR PINECONE API KEY HERE]
PINECONE_API_ENV = [ENTER YOUR PINECONE API ENVIRONMENT HERE]
```
