import streamlit as st
import os
from dotenv import load_dotenv
# Import functions from aiAssistant
from aiAssistant import ask_question

# Load environment variables
load_dotenv()

# Set page title and favicon
st.set_page_config(
    page_title="TIES Documentation Assistant",
    page_icon="🔷",  # Blue diamond emoji to match Trilogy's blue branding
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to style the app with Trilogy's color scheme
st.markdown("""
<style>
    /* Main area background and text colors */
    .stApp {
        background-color: #f8f9fa;
        color: #1e1e1e;
    }
    
    /* Header styling */
    .main .block-container h1 {
        color: #0e2b3d;  /* Dark blue from Trilogy's branding */
        font-weight: 600;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* User message styling - using Trilogy's blue */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #e6f2ff;
        border-left: 4px solid #1e88e5;
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #ffffff;
        border-left: 4px solid #0e2b3d;
    }
    
    /* Sidebar styling with Trilogy's blue */
    .css-1d391kg, .css-1wrcr25, .css-6qob1r {
        background-color: #0e2b3d;
    }
    
    .sidebar .sidebar-content {
        background-color: #0e2b3d;
        color: white;
    }
    
    /* Button styling using Trilogy's blue */
    .stButton>button {
        background-color: #1e88e5;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #0d47a1;
    }
    
    /* Input box styling */
    .stTextInput>div>div>input {
        border-radius: 4px;
        border: 1px solid #ddd;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f1f8fe;
        border-radius: 4px;
    }
    
    /* Link color */
    a {
        color: #1e88e5;
    }
    
    /* Title styling to match Trilogy's branding */
    h1, h2, h3 {
        color: #0e2b3d;
    }
</style>
""", unsafe_allow_html=True)

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