import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


os.environ["GOOGLE_API_KEY"] = os.getenv('api')


system_template = "You are a helpful assistant. Please response to the user queries"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', 'Question:{question}')
])


model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
parser = StrOutputParser()
chain = prompt_template | model | parser

st.title("LangChain Chatbot Demo")
st.markdown("""
    Welcome to the LangChain Chatbot Demo! 
    Type your query below and get responses powered by Google's Generative AI.
    """)
st.sidebar.header("Chatbot Settings")
st.sidebar.text("You can enter your query in the main section below.")

input_text = st.text_input("Enter your question:", "")


if input_text:
    try:
        with st.spinner('Generating response...'):
            response = chain.invoke({"question": input_text})
            st.write("**Chatbot Response:**")
            st.write(response)
            
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({"role": "user", "text": input_text})
            st.session_state.history.append({"role": "chatbot", "text": response})
    
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.sidebar.header("Conversation History")
if "history" in st.session_state and st.session_state.history:
    for message in st.session_state.history:
        role = "You" if message["role"] == "user" else "Chatbot"
        st.sidebar.write(f"**{role}:** {message['text']}")
