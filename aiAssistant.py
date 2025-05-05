import os
import shutil
import json
import openai
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re
import traceback

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define paths
index_path = "./faiss_index"

#if os.path.exists(index_path):
#    shutil.rmtree(index_path)
#    print(f"Deleted existing vector store at {index_path}")

def get_vectorstore():
    # Force rebuild the index if there's an error
    rebuild_index = False
    
    # Check if the FAISS index already exists
    if os.path.exists(index_path) and not rebuild_index:
        try:
            print("Loading existing vector store...")
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print("Vector store loaded successfully")
            
            # Test the vectorstore with a simple query to ensure it works
            try:
                test_docs = vectorstore.similarity_search("test", k=1)
                print("Vector store tested successfully")
            except Exception as e:
                print(f"Error testing vector store: {e}")
                print("Rebuilding index...")
                rebuild_index = True
                
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Rebuilding index...")
            rebuild_index = True
    else:
        rebuild_index = True
    
    if rebuild_index:
        print("Creating new vector store...")
        
        # Remove the old index if it exists
        if os.path.exists(index_path):
            import shutil
            try:
                shutil.rmtree(index_path)
                print(f"Removed old index at {index_path}")
            except Exception as e:
                print(f"Error removing old index: {e}")
        
        # Load the JSON file
        with open("processed_zendesk_docs_v2.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents = []
        
        # Process each document in the JSON
        for doc in data.get("documents", []):
            # Create a document with the content and metadata
            content = doc.get("full_content", "")
            
            # If no content, use a simple title + id format
            if not content:
                content = f"# {doc.get('title', 'Untitled')}\n\n"
            
            # Create metadata
            metadata = {
                "title": doc.get("title", "Unknown"),
                "article_id": doc.get("id", ""),
                "last_updated": doc.get("last_updated", ""),
                "url": doc.get("url", "")
            }
            
            # Create document
            document = Document(page_content=content, metadata=metadata)
            documents.append(document)
        
        print(f"Loaded {len(documents)} documents")

        # 2. Create a text splitter for chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Smaller chunk size
            chunk_overlap=100,  # Smaller overlap
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # 3. Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # 4. Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # 5. Create vector store in batches
        batch_size = 100  # Process 100 chunks at a time
        vectorstore = None
        
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            print(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} (chunks {i} to {end_idx-1})")
            
            batch = chunks[i:end_idx]
            
            if vectorstore is None:
                # Create new vectorstore with first batch
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                # Add subsequent batches to existing vectorstore
                vectorstore.add_documents(batch)
            
            print(f"Processed batch {i//batch_size + 1}")
        
        print("Vector store created successfully")
        
        # 6. Save the vector store
        vectorstore.save_local(index_path)
        print(f"Vector store saved to {index_path}")
    
    return vectorstore

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
    
    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Get sources and images
    docs = retriever.get_relevant_documents(question)
    
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
            url = f"https://trilogyeffective.zendesk.com/hc/en-us/articles/{article_id}"
        
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
    system_message = f"""You are an AI assistant for Trilogy Energy Solutions, a company that has been partnering with oil & gas companies since 1992 to help them modernize operations, streamline workflows, and gain visibility across the energy value chain.

## About TIES Software
TIES (The Integrated Energy System) is a modern, cloud-native solution that centralizes trading, risk, and operational workflows. It is purpose-built for producers, gatherers, pipeline & storage operators, plant processors, and traders, combining ETRM functionality with deep operational capabilities.

Key components of TIES include:
- Plant & Production Accounting
- Reporting & Forecasting
- Financial Management
- Compliance & Regulatory Reporting
- Settlements & Balancing
- Data & Systems Management

Your role is to provide expert support to users navigating this comprehensive platform, helping them understand features, workflows, and solutions to their technical challenges with the software.

Context:
{context}

{image_context}

Guidelines:
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
- DO NOT include URLs or links in your main response - the sources will be automatically displayed in a separate section.

SOURCE GUIDELINES:
- DO NOT include URLs or links in your main response text.
- If users ask for sources or where to find information, tell them to check the Sources section below your answer.
- You can mention article titles when relevant, but do not include the URLs.
- The sources will be automatically displayed in the "Sources" section below your response.
- When users ask "where can I find more information", direct them to check the Sources section rather than providing links.

HANDLING UNANSWERABLE QUESTIONS:
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
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
    )
    
    # Extract the assistant's response
    answer = response.choices[0].message.content
    
    return answer, unique_sources, []
    