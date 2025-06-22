import os
import json
import traceback
import openai
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import re

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ties-docs")

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=2048
)

def get_vectorstore():
    """Connect to existing Pinecone vector store"""
    
    try:
        print("Connecting to existing Pinecone vector store...")
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            text_key="content"
        )
        
        # Test the connection by doing a simple search
        test_docs = vectorstore.similarity_search("test", k=1)
        print("Vector store connection tested successfully")
        return vectorstore
        
    except Exception as e:
        print(f"Error connecting to Pinecone vector store: {e}")
        traceback.print_exc()
        raise e

def get_attachment_ids_for_articles(article_ids):
    attachment_ids = []
    try:
        with open("processed_zendesk_docs_v2.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"Looking for images in articles with IDs: {article_ids}")
        
        for article in data.get("documents", []):
            if article.get("id") in article_ids:
                print(f"Found matching article: {article.get('id')} - {article.get('title')}")
                
                # Method 1: Check traditional attachments field
                if article.get("attachments"):
                    print(f"Article has {len(article.get('attachments'))} attachments in attachments field")
                    for attachment in article.get("attachments", []):
                        if attachment.get("id"):
                            print(f"Adding image ID from attachments: {attachment.get('id')}")
                            attachment_ids.append(attachment.get("id"))
                
                # Method 2: Extract from document_structure if available
                if article.get("document_structure"):
                    print(f"Article has document_structure, checking for images")
                    structure_images = extract_images_from_structure(article.get("document_structure"))
                    if structure_images:
                        print(f"Found {len(structure_images)} images in document_structure")
                        attachment_ids.extend(structure_images)
                
                # Method 3: Extract from full_content using regex
                if article.get("full_content"):
                    print(f"Checking full_content for image references")
                    # Find all image references in the format ![Image](IMAGE_ID:id)
                    image_pattern = r'!\[Image\]\(IMAGE_ID:(\d+)\)'
                    content_image_ids = re.findall(image_pattern, article.get("full_content", ""))
                    if content_image_ids:
                        print(f"Found {len(content_image_ids)} image references in full_content")
                        attachment_ids.extend(content_image_ids)
                
                if not attachment_ids:
                    print(f"Article {article.get('id')} has no images found by any method")
                else:
                    print(f"Total images found for article {article.get('id')}: {len(set(attachment_ids))}")
    except Exception as e:
        print(f"Error extracting image IDs from JSON: {e}")
        traceback.print_exc()
    
    # Remove duplicates
    unique_ids = list(set(attachment_ids))
    print(f"Found {len(unique_ids)} unique attachment IDs: {unique_ids}")
    return unique_ids

def extract_images_from_structure(structure):
    """Recursively extract image IDs from document structure"""
    image_ids = []
    
    if isinstance(structure, list):
        for item in structure:
            image_ids.extend(extract_images_from_structure(item))
    elif isinstance(structure, dict):
        if structure.get('type') == 'image' and 'id' in structure:
            image_ids.append(structure['id'])
        for key, value in structure.items():
            if key != 'type':  # Avoid processing the type field again
                image_ids.extend(extract_images_from_structure(value))
    
    return image_ids

def ask_question(question, chat_history=None, vectorstore=None):
    if vectorstore is None:
        vectorstore = get_vectorstore()

    # 1. Retrieve documents with their similarity scores to filter by relevance.
    # We fetch a larger number of documents (k=5) to have a pool to select from.
    docs_and_scores = vectorstore.similarity_search_with_score(
        question,
        k=5
    )

    # 2. Filter documents by a relevance threshold to ensure source quality.
    # The threshold is tunable (0.0 to 1.0 for cosine similarity).
    # A score of 0.75 is a good starting point.
    RELEVANCE_THRESHOLD = 0.40
    relevant_docs = []
    print("--- Document Relevance Scores ---")
    for doc, score in docs_and_scores:
        # The score from similarity_search_with_score is typically distance (lower is better),
        # but LangChain normalizes some to be similarity scores. Assuming higher is better.
        # We will treat it as similarity where > threshold is good.
        print(f"  - Score: {score:.4f}, Title: {doc.metadata.get('title', 'N/A')}")
        if score > RELEVANCE_THRESHOLD:
            relevant_docs.append(doc)

    # Use only the relevant documents for the rest of the process
    docs = relevant_docs
    
    if not docs:
        print("No relevant documents found above the threshold. Proceeding without sources.")
    else:
        print(f"Found {len(docs)} relevant documents above threshold {RELEVANCE_THRESHOLD}.")

    # Extract source information with URLs for linking
    sources = []
    article_ids = []
    for doc in docs:
        # Debug: print the metadata to see what's available
        print(f"Document metadata: {doc.metadata}")
        
        title = doc.metadata.get("title", "Unknown")
        article_id = doc.metadata.get("article_id", "")
        url = doc.metadata.get("url", "")
        
        if article_id:
            article_ids.append(article_id)
        
        # Ensure we have a valid URL
        if not url or not url.startswith("http"):
            url = f"http://localhost:51744/#/help/{article_id}" #change to the actual TIES url
        
        sources.append({
            "title": title,
            "article_id": article_id,
            "url": url
        })
    
    # Get image IDs for the retrieved articles
    attachment_ids = get_attachment_ids_for_articles(article_ids)
    
    # Remove duplicates while preserving order
    unique_attachment_ids = []
    for img_id in attachment_ids:
        if img_id not in unique_attachment_ids:
            unique_attachment_ids.append(img_id)
    
    print(f"Found {len(unique_attachment_ids)} unique image IDs: {unique_attachment_ids}")
    
    # Get unique sources by article_id
    unique_sources = []
    seen_ids = set()
    for source in sources:
        if source["article_id"] not in seen_ids and source["article_id"]:
            seen_ids.add(source["article_id"])
            unique_sources.append(source)
    
    # Create a prompt that includes image information
    image_context = ""
    if unique_attachment_ids:
        image_context = "\n\nThe following images are available for reference:\n"
        for img_id in unique_attachment_ids[:5]:  # Limit to 5 images
            image_context += f"![Image](IMAGE_ID:{img_id})\n"
    
    # Prepare the context from the retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create the system message with guidelines
    system_message = f"""You are an AI assistant for TIES (Trilogy Integrated Energy Solutions) software. Your role is to provide accurate and helpful information about TIES software features, functionality, and processes.

## About TIES Software
TIES (The Integrated Energy System) is a modern, cloud-native solution that centralizes trading, risk, and operational workflows. It is purpose-built for producers, gatherers, pipeline & storage operators, plant processors, and traders, combining ETRM functionality with deep operational capabilities.

## Core Directives & Boundaries
- Your single purpose is to answer questions about TIES software using the provided documentation.
- If a user asks about something unrelated to TIES (e.g., general knowledge, news, other software), politely state that you can only answer questions about TIES.
- **Strictly avoid:**
  - Expressing personal or political opinions.
  - Recommending or comparing TIES with competitor products.
  - Engaging in off-topic conversations.
- You are an informational assistant. Do not pretend to perform actions, take user orders, or make changes to any system. Your role is to explain how a user can do something, not to do it for them.
- Maintain a helpful, natural, and conversational tone.

Context:
{context}

{image_context}

## Answer Guidelines
- ALWAYS maintain context from previous questions in the conversation.
- If you don't immediately know the answer, look for related concepts in the context that might help.
- NEVER just say "I don't know" without suggesting related topics or asking clarifying questions.
- Keep your answers concise and focused on the documentation provided.
- Use bullet points or numbered lists for step-by-step instructions.
- Format your response with markdown for better readability (headers, bold, lists).
- If the user asks about configuration, include specific field names, options, and default values.
- When explaining processes, clearly indicate the sequence of steps and any dependencies.
- If multiple approaches exist for a task, briefly outline each option with its use case.
- For technical terms specific to TIES, provide brief definitions when first mentioned.
- If a feature has limitations or requirements, clearly state them.
- When appropriate, include examples to illustrate concepts.
- If you can't provide complete information on a topic, offer to explain what you do know and ask if the user would like more details.
- Whenever you make a reference to TIES.Connect, just refer to it as TIES.

## Source Guidelines
- **IMPORTANT:** Never include URLs or web links in your main response. Sources are handled automatically.
- If users ask for sources or where to find information, tell them to check the "Sources" section that appears with your answer.
- You can mention article titles when relevant, but do not include URLs.

## Documentation Update Guidance
- When you recognize that a user wants to update documentation, prioritize helping them find the right article to update.
- Focus on guiding the user to the correct article rather than explaining how to perform the task they want to document.
- Provide the exact title of the article that needs updating based on the user's description.
- If multiple articles might be relevant, list their titles in order of relevance.
- If no existing article seems to match what the user wants to update, suggest the most closely related articles as potential starting points.
- Ask clarifying questions if needed to better understand which documentation the user is trying to update.
- Remember that finding the right documentation to update is often the user's biggest challenge, not explaining the content itself.

## Handling Unanswerable Questions
- Consider a question unanswerable if:
- The retrieved documents don't mention the specific topic or process being asked about
- The documents mention the topic but don't provide clear instructions or details
- The retrieved information is tangential or only vaguely related to the query
- Before stating you don't have information, check if the question might be using terminology different from the documentation (e.g., "master storage deal" vs "primary storage transaction")
- If the documents provide partial information, acknowledge this limitation while still sharing what's available
- If you cannot find specific information about a user's question in the provided context, do not make up information
- Instead, acknowledge the limitation by saying: "I don't have detailed information about [specific topic] in my knowledge base"
- Then offer related information: "However, I can provide information on related topics such as [list 2-3 related topics from the context]"
- Always provide an option to contact support: "Would you like me to share what I know about these related topics, or would you prefer to contact our support team for specific assistance?"
- If the user chooses support, provide: "You can reach our support team by submitting a ticket through the TIES support portal or by emailing support@trilogyenergysolutions.com"
"""
    
    # Create the chat history for the conversation
    messages = [{"role": "system", "content": system_message}]
    
    # Add the chat history if provided
    if chat_history:
        # Ensure chat history is in the correct format
        for message in chat_history:
            if isinstance(message, dict) and "role" in message and "content" in message:
                # Only include user and assistant messages, not system messages
                if message["role"] in ["user", "assistant"]:
                    messages.append(message)
            else:
                print(f"Warning: Skipping invalid message format in chat history: {message}")
    
    # Add the user's question
    messages.append({"role": "user", "content": question})
    
    # Debug: Print the messages being sent to the API
    print(f"Sending {len(messages)} messages to the API")
    for i, msg in enumerate(messages):
        print(f"Message {i}: role={msg.get('role')}, content_length={len(msg.get('content', ''))}")
    
    # Get the response from the model
    response = openai.chat.completions.create(
        model="ft:gpt-4.1-2025-04-14:bridgeiq:trilogyai:BQiiHK25",
        messages=messages,
        temperature=0.7,
    )
    
    # Extract the assistant's response
    answer = response.choices[0].message.content
    print(unique_sources)
    return answer, unique_sources, []