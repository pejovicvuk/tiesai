import streamlit as st
import re
import os
import base64
import requests
import traceback
from aiAssistant import ask_question, get_vectorstore

# Set page configuration
st.set_page_config(
    page_title="TIES AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "attachment_ids" not in st.session_state:
    st.session_state.attachment_ids = []

# Function to fetch image from Zendesk
def get_image_base64(image_id):
    print(f"Attempting to fetch image with ID: {image_id}")
    
    # Check if image is already in cache
    if image_id in st.session_state.image_cache:
        print(f"Image {image_id} found in cache")
        return st.session_state.image_cache[image_id]
    
    # Configuration
    zendesk_subdomain = os.getenv("ZENDESK_SUBDOMAIN", "trilogyeffective")
    zendesk_user = os.getenv("ZENDESK_EMAIL", "TIESConnectHelpCenterBot@trilogyes.com")
    encoded_token = os.getenv("ZENDESK_API_TOKEN")
    
    # Decode the token if needed
    if encoded_token and encoded_token.endswith('=='):
        try:
            api_token = base64.b64decode(encoded_token).decode('utf-8')
            print("Token was base64 encoded, decoded successfully")
        except Exception as e:
            print(f"Error decoding token: {e}")
            api_token = encoded_token
    else:
        api_token = encoded_token
        print("Using token as-is (not base64 encoded)")
    
    # Set up authentication
    auth = (f"{zendesk_user}/token", api_token)
    
    # First, get the attachment metadata
    url = f"https://{zendesk_subdomain}.zendesk.com/api/v2/help_center/articles/attachments/{image_id}.json"
    
    print(f"Fetching attachment information from {url}...")
    
    try:
        # Get the attachment metadata
        response = requests.get(url, auth=auth)
        response.raise_for_status()
        
        attachment_data = response.json()
        attachment = attachment_data.get('article_attachment', {})
        
        # Get the content URL from the attachment data
        content_url = attachment.get('content_url')
        if not content_url:
            print("No content URL found in the attachment data")
            return None
        
        content_type = attachment.get('content_type', 'image/png')
        print(f"Found attachment: {attachment.get('name')} ({content_type})")
        print(f"Downloading from: {content_url}")
        
        # Create a session with authentication
        session = requests.Session()
        session.auth = auth
        
        # Download the actual image with authentication
        image_response = session.get(content_url)
        image_response.raise_for_status()
        
        # Convert image to base64
        encoded = base64.b64encode(image_response.content).decode()
        
        # Create data URL
        data_url = f"data:{content_type};base64,{encoded}"
        
        # Cache the image
        st.session_state.image_cache[image_id] = data_url
        print(f"Successfully fetched image {image_id}")
        
        return data_url
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error fetching image {image_id}: {e}")
        
        # Try alternative approach for restricted content
        if hasattr(e, 'response') and e.response.status_code == 403:
            print("Attempting alternative approach for restricted content...")
            return fetch_restricted_image(image_id, auth, zendesk_subdomain)
            
        return None
    except Exception as e:
        print(f"Error fetching image: {e}")
        traceback.print_exc()
        return None

def fetch_restricted_image(image_id, auth, zendesk_subdomain):
    """Alternative approach for fetching restricted images"""
    
    # Try direct API access to the attachment content
    url = f"https://{zendesk_subdomain}.zendesk.com/api/v2/help_center/articles/attachments/{image_id}/inline"
    
    print(f"Trying direct API access: {url}")
    
    try:
        session = requests.Session()
        session.auth = auth
        
        response = session.get(url)
        response.raise_for_status()
        
        # Determine content type
        content_type = response.headers.get('Content-Type', 'image/png')
        
        # Convert to base64
        encoded = base64.b64encode(response.content).decode()
        data_url = f"data:{content_type};base64,{encoded}"
        
        # Cache the image
        st.session_state.image_cache[image_id] = data_url
        print(f"Successfully downloaded restricted image")
        return data_url
        
    except requests.exceptions.HTTPError as e:
        print(f"Failed to fetch restricted image: {e}")
        
        # One more attempt with a different endpoint
        try:
            url = f"https://{zendesk_subdomain}.zendesk.com/hc/article_attachments/{image_id}"
            print(f"Trying final approach: {url}")
            
            headers = {
                'Authorization': f'Basic {base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()}'
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Try to determine content type from headers
            content_type = response.headers.get('Content-Type', 'image/png')
            
            # Convert to base64
            encoded = base64.b64encode(response.content).decode()
            data_url = f"data:{content_type};base64,{encoded}"
            
            # Cache the image
            st.session_state.image_cache[image_id] = data_url
            print(f"Successfully downloaded restricted image with final approach")
            return data_url
            
        except Exception as e:
            print(f"All attempts failed: {e}")
            traceback.print_exc()
            return None
    
    except Exception as e:
        print(f"Error in alternative approach: {e}")
        traceback.print_exc()
        return None

# Function to process AI response and replace image references
def process_response(response, attachment_ids=None):
    # Regular expression to find image references in the new format
    image_pattern = r'!\[Image\]\(IMAGE_ID:(\d+)\)'
    image_matches = re.findall(image_pattern, response)
    
    # Process images referenced in the text
    processed_response = response
    zendesk_subdomain = os.getenv("ZENDESK_SUBDOMAIN", "trilogyeffective")
    
    # Track which images were already processed to avoid duplicates
    processed_images = set()
    
    for image_id in image_matches:
        # Skip if we've already processed this image
        if image_id in processed_images:
            # Remove duplicate references
            processed_response = processed_response.replace(f'![Image](IMAGE_ID:{image_id})', '')
            continue
        
        # Try multiple URL formats
        image_urls = [
            f"https://{zendesk_subdomain}.zendesk.com/hc/article_attachments/{image_id}",
            f"https://{zendesk_subdomain}.zendesk.com/hc/en-us/article_attachments/{image_id}",
            f"https://{zendesk_subdomain}.zendesk.com/api/v2/help_center/articles/attachments/{image_id}/inline"
        ]
        
        # Use the first URL format for now
        image_url = image_urls[0]
        
        # Replace the reference with an img tag using the direct URL
        img_html = f'<img src="{image_url}" alt="Image {image_id}" style="max-width: 100%; display: block; margin: 20px auto;" onerror="this.onerror=null; this.src=\'{image_urls[1]}\'; this.onerror=function(){{this.src=\'{image_urls[2]}\';}};">'
        processed_response = processed_response.replace(f'![Image](IMAGE_ID:{image_id})', img_html)
        
        # Mark this image as processed
        processed_images.add(image_id)
        
        # Log successful image processing
        print(f"Processed image reference for ID: {image_id}")
    
    # If there are attachment_ids that weren't referenced in the text, don't add them
    # We're removing the "Additional Images" section as requested
    
    return processed_response

# Function to display chat messages
def display_chat_messages():
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"], unsafe_allow_html=True)

# Function to display sources
def display_sources():
    if st.session_state.sources:
        with st.expander("Sources", expanded=True):
            for source in st.session_state.sources:
                st.markdown(f"[{source['title']}]({source['url']})")

# Main function
def main():
    # Custom CSS for styling
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stChatMessage.user {
        background-color: #f0f2f6;
    }
    .stChatMessage.assistant {
        background-color: #e6f7ff;
    }
    .sources-section {
        margin-top: 2rem;
        padding: 1rem;
        background-color: #f9f9f9;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("TIES AI Assistant")
    st.markdown("Ask me anything about TIES software and features.")

    # Initialize vectorstore
    with st.spinner("Loading knowledge base..."):
        vectorstore = get_vectorstore()

    # Display chat messages
    display_chat_messages()

    # Display sources
    display_sources()

    # Chat input
    prompt = st.chat_input("Ask a question about TIES...")
    
    if prompt:
        # Add user message to conversation history
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.spinner("Thinking..."):
            answer, unique_sources, unique_attachment_ids = ask_question(prompt, st.session_state.conversation_history, vectorstore)
            
            # Process the response to handle image references
            processed_answer = process_response(answer, unique_attachment_ids)
            
            # Update session state
            st.session_state.conversation_history.append({"role": "assistant", "content": processed_answer})
            st.session_state.sources = unique_sources
            st.session_state.attachment_ids = unique_attachment_ids
        
        # Display AI response
        with st.chat_message("assistant"):
            st.markdown(processed_answer, unsafe_allow_html=True)
        
        # Display sources
        display_sources()

if __name__ == "__main__":
    main()