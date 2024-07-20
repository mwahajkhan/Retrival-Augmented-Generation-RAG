from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import openai

import openai

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"

from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model=MODEL,
    temperature=0.5,
    max_tokens=100)

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()


#StreamlitChatMessageHistory() will store messages in Streamlit session state at specified key
history = StreamlitChatMessageHistory(key="chat_messages")

#creating a memory object
memory = ConversationBufferMemory(chat_memory=history)

#creating template and prompt to accept a question
template = ("""You are an AI chatbot having a conversation with a human. 
Summarize your response in a concise manner within 100 tokens. Make test readable by good formatting.
Human: {human_input}
AI: """)
prompt = PromptTemplate(input_variables=["human_input"], template=template)

#creating a chain object
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

#Creating UI using Streamlit
#Using streamlit to print all messages in memory, create text input, run chain
#the question and response is automatically passed in the StreamlitChatMessageHistory

import streamlit as st

st.title("Welcome to RAG AI chatbot")
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)  #writing content of messages on ui

if x := st.chat_input():
    st.chat_message("human").write(x)

    #new messages will be added to StreamlitChatMessageHistory when the chain is called
    response = llm_chain.run(x)
    st.chat_message("ai").write(response) #writing response on ui


