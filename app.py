import os
import asyncio
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("api")
if not api_key:
    raise ValueError("API key not found. Please set it as an environment variable or secret.")
os.environ["GOOGLE_API_KEY"] = api_key

# Create or set the asyncio event loop for the current thread
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Define the system and user prompts
system_template = "You are a helpful assistant. Please respond to the user queries."
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_template),
    HumanMessage(content="Question:{question}")
])

async def initialize_model():
    """Initialize the ChatGoogleGenerativeAI model."""
    return ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

# Run the model initialization
model = asyncio.run(initialize_model())
parser = StrOutputParser()
chain = prompt_template | model | parser

# Streamlit app interface
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
            
            # Manage conversation history
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({"role": "user", "text": input_text})
            st.session_state.history.append({"role": "chatbot", "text": response})
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display conversation history
st.sidebar.header("Conversation History")
if "history" in st.session_state and st.session_state.history:
    for message in st.session_state.history:
        role = "You" if message["role"] == "user" else "Chatbot"
        st.sidebar.write(f"**{role}:** {message['text']}")
