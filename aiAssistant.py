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

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit(1)

# Define metadata extraction function
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title", "")
    metadata["article_id"] = record.get("article_id", "")
    metadata["last_updated"] = record.get("last_updated", "")
    # Add image information
    metadata["images"] = record.get("images", [])
    metadata["image_folder"] = record.get("image_folder", "")
    return metadata

# Path to save/load the FAISS index
index_path = "./faiss_index"

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

# 5. Create a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 6. Create a prompt template
template = """You are an assistant for TIES.Connect software documentation. Use the following pieces of context to answer the question at the end.
    
Guidelines:
- ALWAYS maintain context from previous questions in the conversation.
- If you don't immediately know the answer, look for related concepts in the context that might help.
- NEVER just say "I don't know" without suggesting related topics or asking clarifying questions.
- If you recognize keywords (like "book", "job", "order") but need clarification, ask specific follow-up questions.
- ALWAYS provide the actual information from documentation rather than telling users to "refer to documentation."
- When relevant images are available in the context, REFERENCE them using the exact format: [image_id: 12345]
- Keep your answers concise and focused on the documentation provided.
- Use bullet points or numbered lists for step-by-step instructions.
- When explaining features, mention their business benefits.
- If relevant, suggest related features or settings that might be helpful.
- Format your response with markdown for better readability.
- If the user asks about configuration, include specific field names and options.

Context:
{context}

Question: {question}
Answer:"""

prompt = PromptTemplate.from_template(template)

# 7. Create the LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 8. Create the RAG chain using the new LangChain expression language
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 9. Function to query the system
def ask_question(question):
    answer = rag_chain.invoke(question)
    docs = retriever.get_relevant_documents(question)
    sources = [doc.metadata.get("title", "Unknown") for doc in docs]
    unique_sources = list(set(sources))
    
    # Extract image information
    image_references = []
    for doc in docs:
        if "images" in doc.metadata and doc.metadata["images"]:
            image_folder = doc.metadata.get("image_folder", "")
            for img in doc.metadata["images"]:
                image_path = os.path.join("zendesk documentation", image_folder, img)
                if os.path.exists(image_path):
                    image_references.append({
                        "path": image_path, 
                        "title": doc.metadata.get("title", "Unknown")
                    })
    
    print(f"Answer: {answer}")
    print(f"Sources: {', '.join(unique_sources)}")
    if image_references:
        print(f"Found {len(image_references)} relevant images")
    
    return answer, unique_sources, image_references

# Example usage
if __name__ == "__main__":
    question = "How do I configure Natural Gas Intelligence in TIES.Connect?"
    answer, sources, images = ask_question(question)