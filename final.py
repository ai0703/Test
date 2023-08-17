!pip install streamlit langchain
import streamlit as st
from langchain.memory import StreamlitChatMessageHistory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Initialize StreamlitChatMessageHistory
msgs = StreamlitChatMessageHistory(key="chat_messages")

# Initialize ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# Initialize LangChain components
llm_chain = LLMChain(llm=OpenAI(), prompt=prompt, memory=memory)
template = """You are an AI chatbot having a conversation with a human.\n\n{history}\nHuman: {human_input}\nAI: """

# Streamlit UI
st.title("AI Personalized Training Tutor")

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input("You: "):
    st.chat_message("human").write(prompt)

    # Run LangChain to get AI response
    response = llm_chain.run(prompt)
    st.chat_message("ai").write(response)
    msgs.add_user_message(prompt)
    msgs.add_ai_message(response)
