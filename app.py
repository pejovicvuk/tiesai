import streamlit as st
import os
import re
from dotenv import load_dotenv
# Import functions from aiAssistant
from aiAssistant import ask_question
# Import image handler
from image_handler import get_image_base64

# Load environment variables
load_dotenv()

# Set page title and favicon
st.set_page_config(
    page_title="TIES Documentation Assistant",
    page_icon="🔷",
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

# Function to process AI response and replace image references
def process_response(response):
    # Regular expression to find image references
    image_pattern = r'\[IMAGE: (\d+) - (.*?)\]'
    
    # Find all image references
    image_refs = re.findall(image_pattern, response)
    
    # Replace each image reference with an actual image
    for image_id, description in image_refs:
        image_base64 = get_image_base64(image_id)
        if image_base64:
            # Create HTML for the image with the description as alt text
            img_html = f'<img src="{image_base64}" alt="{description}" style="max-width: 100%; margin: 10px 0;">'
            # Replace the reference with the actual image
            response = response.replace(f'[IMAGE: {image_id} - {description}]', img_html)
    
    return response

# Create header
st.title("TIES Documentation Assistant")
st.markdown("Ask questions about TIES documentation")

# Create the sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This assistant uses RAG (Retrieval Augmented Generation) to answer questions about TIES documentation.
    
    It searches through the documentation to find relevant information and uses GPT to generate helpful responses.
    """)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Process assistant messages to display images
            st.markdown(process_response(message["content"]), unsafe_allow_html=True)
        else:
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
            
            # Process answer to handle image references
            processed_answer = process_response(answer)
            
            # Display answer with images
            st.markdown(processed_answer, unsafe_allow_html=True)
            
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