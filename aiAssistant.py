import os
#import shutil
import json
import openai
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

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
    # Define the embedding model consistently
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=3072
    )
    
    if os.path.exists(index_path):
        try:
            print("Loading existing vector store...")
            vectorstore = FAISS.load_local(
                index_path, 
                embeddings,  # Use the same embedding model definition
                allow_dangerous_deserialization=True
            )
            print("Vector store loaded successfully")
            return vectorstore
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Rebuilding vector store due to embedding model change...")
            # Fall through to rebuild
    else:
        print("Creating new vector store...")
    
    with open("processed_zendesk_docs_v2.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    documents = []
    
    # Process documents in smaller batches to avoid memory issues
    batch_size = 10  # Adjust based on your document sizes
    total_docs = len(data.get("documents", []))
    
    for i in range(0, total_docs, batch_size):
        batch_docs = data.get("documents", [])[i:i+batch_size]
        batch_documents = []
        
        for doc in batch_docs:
            content = ""
            
            for section in doc.get("structured_content", []):
                heading = section.get("heading", "")
                if heading:
                    content += f"# {heading}\n\n"
                
                for item in section.get("content", []):
                    content += f"{item}\n\n"
            
            if not content:
                content = f"# {doc.get('title', 'Untitled')}\n\n"
            
            if doc.get("attachments"):
                content += "\n\n## Images\n\n"
                for attachment in doc.get("attachments", []):
                    attachment_id = attachment.get("id", "")
                    if attachment_id:
                        content += f"[IMAGE: {attachment_id}]\n\n"
                        content += f"Context: {attachment.get('context_before', '')} [IMAGE] {attachment.get('context_after', '')}\n\n"
            
            metadata = {
                "title": doc.get("title", "Unknown"),
                "article_id": doc.get("id", ""),
                "last_updated": doc.get("last_updated", ""),
                "url": doc.get("url", "")
            }
            
            document = Document(page_content=content, metadata=metadata)
            batch_documents.append(document)
        
        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(batch_documents)
        documents.extend(chunks)
        print(f"Processed batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}: {len(chunks)} chunks")
    
    print(f"Total: {len(documents)} chunks from {total_docs} documents")
    
    # Create vector store in batches
    vectorstore = None
    batch_size = 100  # Adjust based on your system's memory
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"Creating embeddings for batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size} ({len(batch)} chunks)")
        
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            batch_vectorstore = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(batch_vectorstore)
    
    print("Vector store created successfully")
    
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
                if article.get("attachments"):
                    print(f"Article has {len(article.get('attachments'))} attachments")
                    for attachment in article.get("attachments", []):
                        if attachment.get("id"):
                            print(f"Adding image ID: {attachment.get('id')}")
                            attachment_ids.append(attachment.get("id"))
                else:
                    print(f"Article {article.get('id')} has no attachments")
    except Exception as e:
        print(f"Error extracting image IDs from JSON: {e}")
    
    return attachment_ids

def ask_question(question, chat_history=None, vectorstore=None):
    if vectorstore is None:
        vectorstore = get_vectorstore()
    
    retriever = vectorstore.as_retriever(
        search_type="similarity", #testiraj i menjaj
        search_kwargs={"k": 5}
    )
    
    docs = retriever.get_relevant_documents(question)
    
    sources = []
    article_ids = []
    for doc in docs:
        print(f"Document metadata: {doc.metadata}")
        
        title = doc.metadata.get("title", "Unknown")
        article_id = doc.metadata.get("article_id", "")
        url = doc.metadata.get("url", "")
        
        if article_id:
            article_ids.append(article_id)
        
        if not url or not url.startswith("http"):
            url = f"https://trilogyeffective.zendesk.com/hc/en-us/articles/{article_id}"
        
        sources.append({
            "title": title,
            "article_id": article_id,
            "url": url
        })
    
    attachment_ids = get_attachment_ids_for_articles(article_ids)
    
    unique_attachment_ids = []
    for attachment_id in attachment_ids:
        if attachment_id not in unique_attachment_ids:
            unique_attachment_ids.append(attachment_id)
    
    print(f"Found {len(unique_attachment_ids)} unique attachment IDs: {unique_attachment_ids}")
    
    unique_sources = []
    seen_ids = set()
    for source in sources:
        if source["article_id"] not in seen_ids and source["article_id"]:
            seen_ids.add(source["article_id"])
            unique_sources.append(source)
    
    image_context = ""
    if unique_attachment_ids:
        image_context = "\n\nThe following images are available for reference:\n"
        for attachment_id in unique_attachment_ids[:10]:  # Limit to 10 images
            image_context += f"[IMAGE: {attachment_id}]\n"
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
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
    
    IMPORTANT IMAGE GUIDELINES:
    - When discussing a feature that has an associated image, place an image reference EXACTLY where it belongs in your response.
    - Insert image references at the appropriate point in your text, not just at the end of your response.
    - If a section has multiple images, include ALL of them at their correct positions in your text.
    - Use this format for image references: [IMAGE: image_id]
    - Example: [IMAGE: 33996955212813]
    - The image should appear immediately after the text that describes what the image shows.
    - NEVER say "image not available" - if you can't find a relevant image, simply don't mention images.
    - Use the image IDs provided in the context when referencing images.
    
    SOURCE GUIDELINES:
    - DO NOT include URLs or links in your main response text.
    - If users ask for sources or where to find information, tell them to check the Sources section below your answer.
    - You can mention article titles when relevant, but do not include the URLs.
    - The sources will be automatically displayed in the "Sources" section below your response.
    - When users ask "where can I find more information", direct them to check the Sources section rather than providing links.

    IMPORTANT DOMAIN KNOWLEDGE:
    - Facility Types: TIES recognizes several types of facilities including gathering, pipeline, processing plant, ISO, storage, producer field, and refinery
    - Pipelines are classified as a type of facility in the TIES system
    - Station Types: Meters, pools, and hubs are all types of stations
    - Every facility is associated with a Business Associate as its operator
    - Stations represent points on a facility and include meters, pools, hubs, and interconnects
    - Each station belongs to a facility and has a location code, meter number (DRN), and type (e.g., receipt, delivery)
    - Business Associates are used for all counterparties, operators, pipelines, and trading partners
    - TIES separates workflows between Operator Scenarios (pipeline, plant, scheduling) and Trading Scenarios (deal capture, MtM, credit, position, risk)
    - A Book is a financial container for organizing transactions by purpose, strategy, or region
    - Physical trades and Financial trades have different properties and cannot be both simultaneously
    - Basis Points link stations or locations to reference prices used for risk reporting and valuation

    GUIDANCE ON DOMAIN KNOWLEDGE:
    - When you see content from documents marked as "domain_knowledge", use this to inform your understanding of relationships and concepts
    - Do not directly quote from domain knowledge documents in your answers
    - Use domain knowledge to verify your understanding of TIES concepts before providing answers
    - Domain knowledge documents provide context for how different parts of TIES relate to each other
    
    ## Determining Answer Availability
    - Consider a question FULLY answerable if:
      - The retrieved context contains explicit instructions or explanations that directly address the specific question
      - Key terminology from the question appears in similar contexts within the retrieved documents
      - The information is detailed enough to provide step-by-step guidance if the user is asking for a process

    - Consider a question PARTIALLY answerable if:
      - The retrieved context mentions the topic but lacks complete details
      - Related concepts are explained but the specific question isn't directly addressed
      - General principles are available that can be applied to the specific question

    - Consider a question UNANSWERABLE if:
      - None of the key terms or concepts from the question appear in the retrieved context
      - The retrieved information is about entirely different topics or processes
      - The context contains only tangential information that wouldn't help the user accomplish their goal

    - When a question is partially answerable, clearly state the limitations of your knowledge before providing the partial information
    
    - When a question is unanswerable, identify the most closely related topics from your context before suggesting support options
    
    - Before stating you don't have information, check if the question might be using terminology different from the documentation (e.g., "master storage deal" vs "primary storage transaction")
    - If the documents provide partial information, acknowledge this limitation while still sharing what's available
    - If you cannot find specific information about a user's question in the provided context, do not make up information
    - Instead, acknowledge the limitation by saying: "I don't have detailed information about [specific topic] in my knowledge base"
    - Then offer related information: "However, I can provide information on related topics such as [list 2-3 related topics from the context]"
    - Always provide an option to contact support: "Would you like me to share what I know about these related topics, or would you prefer to contact our support team for specific assistance?"
    - If the user chooses support, provide: "You can reach our support team by submitting a ticket through the TIES support portal or by emailing support@trilogyenergysolutions.com"
    """
    
    messages = [{"role": "system", "content": system_message}]
    
    if chat_history:
        for message in chat_history:
            if isinstance(message, dict) and "role" in message and "content" in message:
                if message["role"] in ["user", "assistant"]:
                    messages.append(message)
            else:
                print(f"Warning: Skipping invalid message format in chat history: {message}")
    
    messages.append({"role": "user", "content": question})
    
    print(f"Sending {len(messages)} messages to the API")
    for i, msg in enumerate(messages):
        print(f"Message {i}: role={msg.get('role')}, content_length={len(msg.get('content', ''))}")
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
    )
    
    answer = response.choices[0].message.content
    
    return answer, unique_sources, unique_attachment_ids
    