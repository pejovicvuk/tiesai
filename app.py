import streamlit as st
import re
import os
import base64
import requests
import traceback
from aiAssistant import ask_question, get_vectorstore

# Set page configuration
st.set_page_config(
    page_title="TIES.Connect Help Center Assistant",
    page_icon="🤖",
    layout="wide" 
)

# Initialize session state variables if they don't exist
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'image_cache' not in st.session_state:
    st.session_state.image_cache = {}

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
        
        session = requests.Session()
        session.auth = auth
        
        image_response = session.get(content_url)
        image_response.raise_for_status()

        encoded = base64.b64encode(image_response.content).decode()
        
        data_url = f"data:{content_type};base64,{encoded}"
        
        st.session_state.image_cache[image_id] = data_url
        print(f"Successfully fetched image {image_id}")
        
        return data_url
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error fetching image {image_id}: {e}")
        
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
    
    url = f"https://{zendesk_subdomain}.zendesk.com/api/v2/help_center/articles/attachments/{image_id}/inline"
    
    print(f"Trying direct API access: {url}")
    
    try:
        session = requests.Session()
        session.auth = auth
        
        response = session.get(url)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', 'image/png')
        
        encoded = base64.b64encode(response.content).decode()
        data_url = f"data:{content_type};base64,{encoded}"
        
        st.session_state.image_cache[image_id] = data_url
        print(f"Successfully downloaded restricted image")
        return data_url
        
    except requests.exceptions.HTTPError as e:
        print(f"Failed to fetch restricted image: {e}")
        
        try:
            url = f"https://{zendesk_subdomain}.zendesk.com/hc/article_attachments/{image_id}"
            print(f"Trying final approach: {url}")
            
            headers = {
                'Authorization': f'Basic {base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()}'
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', 'image/png')
            
            encoded = base64.b64encode(response.content).decode()
            data_url = f"data:{content_type};base64,{encoded}"
            
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
def process_response(response):
    image_pattern = r'\[IMAGE: (\d+) - (.*?)\]'
    
    # Find all image references in the response
    image_matches = re.findall(image_pattern, response)
    print(f"Found {len(image_matches)} image references in response")
    
    # Replace each image reference with an HTML img tag
    processed_response = response
    for image_id, description in image_matches:
        
        zendesk_subdomain = os.getenv("ZENDESK_SUBDOMAIN", "trilogyeffective")
        image_url = f"https://{zendesk_subdomain}.zendesk.com/hc/article_attachments/{image_id}"
        
        # Create HTML for the image with the description as alt text and caption
        image_html = f"""
        <div style="text-align: center; margin: 20px 0;">
            <img src="{image_url}" alt="{description}" style="max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
        </div>
        """
        
        # Replace the image reference with the HTML
        processed_response = processed_response.replace(f'[IMAGE: {image_id} - {description}]', image_html)
        print(f"Successfully replaced image reference for {image_id}")
    
    return processed_response

# Main function to handle the chat interface
def main():
    st.title("TIES Help Center Assistant")
    
    # Initialize the vector store
    vectorstore = get_vectorstore()
    
    # Chat input
    prompt = st.chat_input("Ask a question about TIES.Connect...")
    
    # Display conversation history
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"], unsafe_allow_html=True)
    
    # Process new user input
    if prompt:
        # Add user message to chat history
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Display assistant response with a spinner
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                # Get response from AI
                answer, unique_sources, unique_attachment_ids = ask_question(prompt, st.session_state.conversation_history, vectorstore)
                
                # Process the response to replace image references with actual images
                processed_answer = process_response(answer)
                
                # Add assistant message to chat history
                st.session_state.conversation_history.append({"role": "assistant", "content": processed_answer})
                
                # Display the processed answer
                message_placeholder.write(processed_answer, unsafe_allow_html=True)
                
                # Display sources if available
                if unique_sources:
                    with st.expander("Sources"):
                        for source in unique_sources:
                            st.markdown(f"[{source['title']}]({source['url']})")

# Run the main function
if __name__ == "__main__":
    main()