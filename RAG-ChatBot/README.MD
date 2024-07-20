# RAG-Chatbot

> streamlit 
> chromadb
> langchain
> OpenAI

This project implements a Retrieval-Augmented Generative (RAG) chatbot that leverages Chromadb for document retrieval and OpenAI for generating responses grounded in retrieved information. 

## Setting Up the Environment

1. Create a Virtual Environment:

     > python3 -m venv .venv


2. Activate the virtual environment to use its packages:

                 source .venv/bin/activate  # Linux/macOS
                .venv\Scripts\activate.bat  # Windows
  
3. Install Required Packages:

      *Install the necessary libraries using a requirements.txt file to ensure consistent dependencies:*

                 pip install -r requirements.txt


# Obtain API Keys:

Retrieve your API keys:
OpenAI API key: https://beta.openai.com/account/api-keys
Create a .env File:

Create a .env file to securely store your API keys:
OPENAI_API_KEY=[YOUR_OPENAI_API_KEY]

## Project Files

The project contains the following components:

1. Prompts.py
* ChatOpenAI from langchain_community.llms is used for chat interactions.
ChatPromptTemplate combines system and user messages to guide the conversation flow.
ConversationBufferMemory stores the complete chat history for reference.
ConversationSummaryMemory uses the LLM to create a concise summary of the conversation.
llm chain piping components of this retrieval mechanism.

2. sessions.py

* StreamlitChatMessageHistory (history).
* ConversationBufferMemory (memory): This memory component from langchain.memory serves as a wrapper around history.
* Streamlit User Interface (UI)
* Streamlit Integration: The code utilizes the Streamlit framework (imported as st) to create a user-friendly chat interface.
* Message History Display: It iterates through the history object and displays each message (human or AI) using st.chat_message and st.write.
* User Input and Response: It uses st.chat_input to capture user questions and then runs the llm_chain to generate a response. Finally, it displays both the user's question and the LLM's response using st.chat_message and st.write.

* To create streamlit session:

     terminal:
          streamlit run sessions.py
* In browser:
     http://localhost:8501


https://github.com/mwahajkhan/Retrival-Augmented-Generation-RAG/blob/e32b383edefbaac7c62e5c349730c429b82076c0/RAG-ChatBot/Screenshot.png


3. chroma-create

* Document Loading and Cleaning.
Text Splitting.
Chroma DB creation to store the chunked documents via embeddings.

4. faiss-create

* Document Loading and Cleaning.
* Text Splitting.
* Faiss DB creation to store the chunked documents via embeddings.

5. retrieval-chroma.py

* Data Retrieval

* Document Exploration: Defines a function pretty_print_docs to display retrieved documents in a user-friendly format (content and title/metadata).
Uses the function to print details of the retrieved documents.

* Retrieval-Augmented QA Chain:
> Creates a RetrievalQA chain object.
> Creates a chain that combines the LLM, retriever, and parser in sequence. 
> This chain retrieves documents based on the user query and then uses the LLM to generate a response considering the retrieved documents.

