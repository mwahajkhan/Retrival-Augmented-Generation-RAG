from langchain.memory. buffer import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.chains import LLMChain
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain_community.llms import OpenAI
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

#Creating a prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot who explain in steps"
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

#Creating memory to remeber chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

summary_memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history")

conversation = LLMChain(llm=llm, prompt=prompt, output_parser=parser, memory=summary_memory)

# Formatting the input correctly
response = conversation.invoke({"question": "What do you know about Seattle?"})
# Invoking the chain to populate variables
response

#printing all messages in the memory
print(memory.chat_memory.messages)
print(summary_memory.chat_memory.messages)
print("Summary of the conversation is----->"+ summary_memory.buffer)

#asking another question
conversation.invoke("What do you know about Mankato?")

#printing all messages in memory again to see the chat populate with previous response
print(memory.chat_memory.messages)
print(summary_memory.chat_memory.messages)
print("Summary of the conversation is----->"+ spromptsummary_memory.buffer)