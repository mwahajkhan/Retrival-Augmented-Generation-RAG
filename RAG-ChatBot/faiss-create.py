from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# from langchain.document_loaders import PyPDFDocumentLoader

from langchain_community.embeddings import GPT4AllEmbeddings


# Function to clean text data
def clean_text(text):
    return text.encode('utf-8', 'ignore').decode('utf-8')

#Load documents folder
pdf_loader = PyPDFDirectoryLoader("./Documents")

loaders = [pdf_loader]

documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2500, chunk_overlap=100)
all_documents = text_splitter.split_documents(documents)

print(f"Total number of documents: {len(all_documents)}")

#Setting up OpenAI gpt 4o- mini model
import openai

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

MODEL = "text-embedding-ada-002"

embeddings = OpenAIEmbeddings(
    model=MODEL,
    #model_kwargs={"truncate":True}
)

#Setting batch size
batch_size = 96

#Calculate number of batches
num_batches = len(all_documents)//batch_size + (len(all_documents) % batch_size > 0)

#creating FAISS db
texts = ["FAISS is an important librarary", "LangChain supports FAISS"]
db = FAISS.from_texts(texts, embeddings)
retv = db.as_retriever()

#Iterate over batches
for batch_num in range(num_batches):
    #Calculate start and end indicies for the current batch
    start_index = batch_num * batch_size
    end_index = min((batch_num + 1) * batch_size, len(all_documents))
    #Extract documents for the current batch
    batch_documents = all_documents[start_index:end_index]
    #Code to process each document
    retv.add_documents(batch_documents)
    print(start_index, end_index)

#persist the collection
# db.persist()
