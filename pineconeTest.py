import os
import json
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trilogyai-docs")

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=3072
)

print("Starting Pinecone index population script")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{PINECONE_INDEX_NAME}' does not exist. Creating...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_ENVIRONMENT
        )
    )
    print(f"Created index: {PINECONE_INDEX_NAME}")
    # Wait for index to initialize
    time.sleep(10)
else:
    print(f"Using existing index: {PINECONE_INDEX_NAME}")

# Get index statistics first to check if it has any vectors
index = pc.Index(PINECONE_INDEX_NAME)
stats = index.describe_index_stats()
vector_count = stats.get('total_vector_count', 0)

# Only try to delete vectors if the index actually has some
if vector_count > 0:
    try:
        print(f"Found {vector_count} existing vectors. Deleting them...")
        index.delete(delete_all=True)
        print("All vectors deleted. Waiting for operation to complete...")
        time.sleep(5)
    except Exception as e:
        print(f"Note: Could not delete vectors. This is normal for an empty index: {e}")
else:
    print("Index is already empty. Proceeding with population.")

# Load data from JSON file
print("Loading documents from JSON file...")
try:
    with open("processed_zendesk_docs_v2.json", "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print("ERROR: processed_zendesk_docs_v2.json file not found!")
    print(f"Current working directory: {os.getcwd()}")
    print("Please make sure the file exists in this directory.")
    exit(1)

# Process documents
documents = []
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

# Create a text splitter for chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Split documents into chunks
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Process in batches
batch_size = 50  # Smaller batch size for more reliable loading
total_batches = (len(chunks) - 1) // batch_size + 1

print(f"Will process in {total_batches} batches")

# Initialize the vector store
print("Initializing PineconeVectorStore...")
vectorstore = PineconeVectorStore.from_documents(
    documents=[chunks[0]],  # Start with just one document to initialize
    embedding=embeddings,
    index_name=PINECONE_INDEX_NAME,
    text_key="text"
)
print("Vector store initialized with first document")

# Now add the rest of the documents
if len(chunks) > 1:
    remaining_chunks = chunks[1:]
    for i in range(0, len(remaining_chunks), batch_size):
        batch_num = (i // batch_size) + 1
        end_idx = min(i + batch_size, len(remaining_chunks))
        
        print(f"Processing batch {batch_num}/{total_batches-1} (chunks {i+1} to {end_idx})")
        
        try:
            # Get the current batch
            batch = remaining_chunks[i:end_idx]
            
            # Add documents to the vector store
            vectorstore.add_documents(batch)
            
            print(f"Successfully processed batch {batch_num}")
            
            # Add a delay between batches to avoid rate limiting
            if i + batch_size < len(remaining_chunks):
                print("Waiting a moment before next batch...")
                time.sleep(2)
        except Exception as e:
            print(f"ERROR processing batch {batch_num}: {e}")
            print("Continuing with next batch...")

# Verify the index now has vectors
final_stats = index.describe_index_stats()
final_vector_count = final_stats.get('total_vector_count', 0)
print(f"\nFinal index stats: {final_stats}")
print(f"Total vectors: {final_vector_count}")

if final_vector_count > 0:
    print("\nSUCCESS: Your Pinecone index has been populated with vectors!")
else:
    print("\nWARNING: Your Pinecone index still appears to be empty after population attempt.")
    print("Check for errors above and try again.")

print("\nPopulation complete!")