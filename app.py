import streamlit as st
import os
import re
import requests
import base64
from io import BytesIO
import certifi
from dotenv import load_dotenv
# Import functions from aiAssistant
from aiAssistant import ask_question

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

zendesk_api_token = os.getenv("ZENDESK_API_TOKEN")
zendesk_subdomain = os.getenv("ZENDESK_SUBDOMAIN", "trilogyeffective")


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for image cache
if "image_cache" not in st.session_state:
    st.session_state.image_cache = {}

# Function to get conversation history as text
def get_conversation_history():
    history = ""
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n\n"
    return history

# Function to fetch image from Zendesk and convert to base64
def get_image_base64(image_id):
    # Check if image is already in cache
    if image_id in st.session_state.image_cache:
        return st.session_state.image_cache[image_id]
    
    try:
        # Configuration
        zendesk_subdomain = os.getenv("ZENDESK_SUBDOMAIN", "trilogyeffective")
        zendesk_user = os.getenv("ZENDESK_EMAIL", "TIESConnectHelpCenterBot@trilogyes.com")
        encoded_token = os.getenv("ZENDESK_API_TOKEN")
        
        # Decode the token if needed
        if encoded_token and encoded_token.endswith('=='):
            try:
                api_token = base64.b64decode(encoded_token).decode('utf-8')
            except Exception as e:
                print(f"Error decoding token: {e}")
                api_token = encoded_token
        else:
            api_token = encoded_token
        
        # Set up authentication
        auth = (f"{zendesk_user}/token", api_token)
        
        # Try direct access to the attachment first (this might work for public attachments)
        direct_url = f"https://{zendesk_subdomain}.zendesk.com/hc/article_attachments/{image_id}"
        direct_response = requests.get(direct_url)
        
        if direct_response.status_code == 200:
            # If direct access works, use that
            content_type = direct_response.headers.get('Content-Type', 'image/png')
            encoded = base64.b64encode(direct_response.content).decode()
            data_url = f"data:{content_type};base64,{encoded}"
            st.session_state.image_cache[image_id] = data_url
            return data_url
        
        # If direct access fails, try the API approach
        # First, get the attachment metadata
        metadata_url = f"https://{zendesk_subdomain}.zendesk.com/api/v2/help_center/articles/attachments/{image_id}.json"
        
        # Get the attachment metadata
        metadata_response = requests.get(metadata_url, auth=auth)
        metadata_response.raise_for_status()
        
        attachment_data = metadata_response.json()
        attachment = attachment_data.get('article_attachment', {})
        
        # Get the content URL from the attachment data
        content_url = attachment.get('content_url')
        if not content_url:
            print(f"No content URL found for attachment {image_id}")
            return None
        
        # Get the content type
        content_type = attachment.get('content_type', 'image/png')
        
        # Download the actual image with authentication
        image_response = requests.get(content_url, auth=auth)
        image_response.raise_for_status()
        
        # Convert image to base64
        encoded = base64.b64encode(image_response.content).decode()
        
        # Create data URL
        data_url = f"data:{content_type};base64,{encoded}"
        
        # Cache the image
        st.session_state.image_cache[image_id] = data_url
        
        return data_url
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error fetching image {image_id}: {e}")
        if hasattr(e, 'response'):
            if e.response.status_code == 404:
                print(f"Attachment with ID {image_id} not found.")
            elif e.response.status_code == 401 or e.response.status_code == 403:
                print(f"Authentication failed or access forbidden. Status code: {e.response.status_code}")
                print(f"Response: {e.response.text[:200]}...")  # Print first 200 chars of response
        return None
    except Exception as e:
        print(f"Error fetching image {image_id}: {str(e)}")
        return None

# Function to process AI response and replace image references
def process_response(response):
    # Regular expression to find image references with ID
    image_pattern = r'\[IMAGE: (\d+).*?\]'
    
    # Find all image references
    image_refs = re.findall(image_pattern, response)
    
    # Replace each image reference with the actual image if it exists
    for image_id in image_refs:
        image_base64 = get_image_base64(image_id)
        if image_base64:
            # Create HTML for the image
            img_html = f'<img src="{image_base64}" alt="Image {image_id}" style="max-width: 100%; margin: 10px 0;">'
            # Replace the entire reference with the actual image
            response = re.sub(r'\[IMAGE: ' + image_id + r'.*?\]', img_html, response)
        else:
            # If image doesn't exist, remove the reference completely
            response = re.sub(r'\[IMAGE: ' + image_id + r'.*?\]', '', response)
    
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
                for source in message["sources"]:
                    title = source.get("title", "Unknown")
                    url = source.get("url", "")
                    if url:
                        st.markdown(f"- [{title}]({url})")
                    else:
                        st.markdown(f"- {title}")

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
                    for source in unique_sources:
                        title = source.get("title", "Unknown")
                        url = source.get("url", "")
                        if url:
                            st.markdown(f"- [{title}]({url})")
                        else:
                            st.markdown(f"- {title}")
            
            # Add assistant message to chat history - STORE THE ORIGINAL ANSWER
            # This ensures image references are preserved for future processing
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,  # Store the original unprocessed answer
                "sources": unique_sources
            })