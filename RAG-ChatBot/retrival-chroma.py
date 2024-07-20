# Ensure the necessary packages are installed
# !pip install langchain langchain_openai langchain_community chromadb openai python-dotenv pypdf

from langchain.chains import RetrievalQA
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setting up OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
generation_model = "gpt-3.5-turbo-instruct"  # Use a valid OpenAI completion model

# Setting up OpenAI LLM
llm = OpenAI(
    model=generation_model,
    openai_api_key=OPENAI_API_KEY,
    max_tokens=2000
)

# Connecting to the ChromaDB server
client = chromadb.HttpClient(host="127.0.0.1", port=8000)

# Setting up the OpenAI embedding model
embedding_model = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(model=embedding_model)

# Setting up retriever to fetch relevant documents
db = Chroma(client=client, embedding_function=embeddings)
retv = db.as_retriever(search_type='similarity', search_kwargs={"k": 5})

# Retrieve relevant documents
docs = retv.invoke('Tell me what is numpy role in data analysis?')

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

pretty_print_docs(docs)

for doc in docs:
    print(doc.metadata)

# Creating retrieval chain that takes LLM, retriever, and invokes it to get response to query
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retv, return_source_documents=True)

chain = llm | retv | parser

response = chain.invoke("Author of the book?")

print(response)
pretty_print_docs(docs)

