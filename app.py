import getpass
import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Set up environment variable for API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = 'AIzaSyBpVjgos0JZ0yLIsbv_jl3zvHJlKEp11-Y'

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
parser = StrOutputParser()
chain = prompt_template | model | parser

# Streamlit app setup
st.title("LangChain Chatbot Demo")
st.markdown("""
    Welcome to the LangChain Chatbot Demo! 
    Type your query below and get responses powered by Google's Generative AI.
    """)
st.sidebar.header("Chatbot Settings")
st.sidebar.text("You can enter your query in the main section below.")

# Input box for user query
input_text = st.text_input("Enter your question:", "")

# Process the user input and generate the response
if input_text:
    with st.spinner('Generating response...'):
        response = chain.invoke({"question": input_text})
        st.write("**Chatbot Response:**")
        st.write(response)

# Display chat history if needed
if "history" not in st.session_state:
    st.session_state.history = []

if input_text:
    st.session_state.history.append({"role": "user", "text": input_text})
    st.session_state.history.append({"role": "chatbot", "text": response})

st.sidebar.header("Conversation History")
if st.session_state.history:
    for message in st.session_state.history:
        role = "You" if message["role"] == "user" else "Chatbot"
        st.sidebar.write(f"**{role}:** {message['text']}")
