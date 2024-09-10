import getpass
import os
import asyncio
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import nest_asyncio

# Allow nested async loops in Streamlit
nest_asyncio.apply()

# Set up environment variable for API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('api')

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
parser = StrOutputParser()
chain = ChatPromptTemplate() | model | parser

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

async def get_response(input_text):
    return await chain.acall({"question": input_text})

# Process the user input and generate the response
if input_text:
    with st.spinner('Generating response...'):
        loop = asyncio.get_event_loop()  # Ensure there's a running event loop
        response = loop.run_until_complete(get_response(input_text))
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
