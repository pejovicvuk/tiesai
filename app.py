import streamlit as st
import os
import re
from dotenv import load_dotenv
# Import functions from aiAssistant
from aiAssistant import ask_question, find_image_path

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

# Function to display response with images
def display_response_with_images(response):
    # Regular expression to find image references
    image_pattern = r'\[image_id: (\d+)\]'
    parts = re.split(image_pattern, response)
    
    if len(parts) == 1:
        # No image references found
        st.markdown(response)
        return
    
    # Display text and images alternately
    for i in range(0, len(parts)):
        if i == 0 or i % 2 == 0:
            # Even indices are text
            st.markdown(parts[i])
        else:
            # Odd indices are image IDs
            image_id = parts[i]
            try:
                # Find the image path
                image_path = find_image_path(image_id)
                if image_path:
                    # Get absolute path
                    if image_path.startswith("./"):
                        image_path = image_path[2:]  # Remove leading ./
                    
                    # Construct absolute path
                    base_dir = os.path.dirname(__file__)
                    absolute_path = os.path.join(base_dir, image_path)
                    
                    if os.path.exists(absolute_path):
                        st.image(absolute_path, caption=f"Image {image_id}")
                    else:
                        st.warning(f"Image file not found: {absolute_path}")
                else:
                    st.warning(f"Image {image_id} not found in documentation")
            except Exception as e:
                st.error(f"Could not load image {image_id}: {str(e)}")

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
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            display_response_with_images(message["content"])
            
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
            answer, unique_sources, image_references = ask_question(prompt, conversation_history)
            
            # Display answer with images
            display_response_with_images(answer)
            
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