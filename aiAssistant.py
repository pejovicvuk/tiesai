import os
import os.path
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit(1)

# Define metadata extraction function
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title", "")
    metadata["article_id"] = record.get("article_id", "")
    metadata["last_updated"] = record.get("last_updated", "")
    return metadata

# Path to save/load the FAISS index
index_path = "./faiss_index"

# Function to load or create the vector store
def get_vectorstore():
    # Check if the FAISS index already exists
    if os.path.exists(index_path):
        print("Loading existing vector store...")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully")
    else:
        print("Creating new vector store...")
        # 1. Load documents directly from your JSON file
        loader = JSONLoader(
            file_path="zendesk documentation/processed_zendesk_docs_enhanced.json",
            jq_schema='.documents[]',
            content_key="markdown_content",
            metadata_func=metadata_func
        )

        documents = loader.load()
        print(f"Loaded {len(documents)} documents")

        # 2. Create a text splitter for chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n#### ", "\n##### ", "\n", " ", ""]
        )

        # 3. Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")

        # 4. Create embeddings and store in vector database
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        vectorstore.save_local(index_path)
        print("Vector store created and persisted")
    
    return vectorstore

# Function to query the system with conversation history
# Function to query the system with conversation history
def ask_question(question, conversation_history=""):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    template = """You are an assistant for TIES.Connect software documentation. Use the following pieces of context to answer the question at the end.
    
    Guidelines:
    - ALWAYS maintain context from previous questions in the conversation.
    - If you don't immediately know the answer, look for related concepts in the context that might help.
    - NEVER just say "I don't know" without suggesting related topics or asking clarifying questions.
    - If you recognize keywords (like "book", "job", "order") but need clarification, ask specific follow-up questions.
    - NEVER refer users to external documentation, guides, or articles. This is critical - do not tell users to "refer to", "see", or "check" any documentation.
    - Instead of referring to documentation, always provide complete answers directly in your response.
    - Keep your answers concise and focused on the documentation provided.
    - Use bullet points or numbered lists for step-by-step instructions.
    - Format your response with markdown for better readability (headers, bold, lists).
    - If the user asks about configuration, include specific field names, options, and default values.
    - When explaining processes, clearly indicate the sequence of steps and any dependencies.
    - If multiple approaches exist for a task, briefly outline each option with its use case.
    - For technical terms specific to TIES.Connect, provide brief definitions when first mentioned.
    - If a feature has limitations or requirements, clearly state them.
    - When appropriate, include examples to illustrate concepts.
    - If you can't provide complete information on a topic, offer to explain what you do know and ask if the user would like more details.
    - Whenever you make a reference to TIES.Connect, just refer to it as TIES.
    
    IMPORTANT IMAGE GUIDELINES:
    - When discussing a feature that has an associated image, place an image reference EXACTLY where it belongs in your response.
    - Insert image references at the appropriate point in your text, not just at the end of your response.
    - If a section has multiple images, include ALL of them at their correct positions in your text.
    - Use this format for image references: [IMAGE: image_id - brief description]
    - Example: [IMAGE: 33996955310861 - Term Supply Planning screen]
    - The image should appear immediately after the text that describes what the image shows.
    
    Previous conversation:
    {conversation_history}
    
    Context:
    {context}
    
    Question: {question}
    Answer:"""

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt.partial(conversation_history=conversation_history)
        | llm
        | StrOutputParser()
    )
    
    # Get answer
    answer = rag_chain.invoke(question)
    
    # Get sources and images
    docs = retriever.get_relevant_documents(question)
    sources = [doc.metadata.get("title", "Unknown") for doc in docs]
    unique_sources = list(set(sources))
    
    return answer, unique_sources
    