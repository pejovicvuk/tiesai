import streamlit as st
import os
from dotenv import load_dotenv
# Import functions from aiAssistant
from aiAssistant import ask_question

# Load environment variables
load_dotenv()

# Set page title and favicon
st.set_page_config(
    page_title="TIES.Connect Documentation Assistant",
    page_icon="📚",
    layout="wide"
)

# Get API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to get conversation history as text
def get_conversation_history():
    history = ""
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n\n"
    return history

# Create header
st.title("TIES.Connect Documentation Assistant")
st.markdown("Ask questions about TIES.Connect documentation")

# Create the sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This assistant uses RAG (Retrieval Augmented Generation) to answer questions about TIES.Connect documentation.
    
    It searches through the documentation to find relevant information and uses GPT to generate helpful responses.
    """)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("Sources"):
                st.markdown("\n".join([f"- {source}" for source in message["sources"]]))

# Get user input
if prompt := st.chat_input("Ask a question about TIES.Connect"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get conversation history
            conversation_history = get_conversation_history()
            
            # Get answer from RAG chain with conversation history
            answer, unique_sources = ask_question(prompt, conversation_history)
            
            # Display answer
            st.markdown(answer)
            
            # Display sources
            if unique_sources:
                with st.expander("Sources"):
                    st.markdown("\n".join([f"- {source}" for source in unique_sources]))
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": unique_sources
            })