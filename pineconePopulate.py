import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# Get environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trilogyai-docs")

print("Verifying Pinecone index status...")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Error: Index '{PINECONE_INDEX_NAME}' not found!")
    exit(1)

# Get index statistics
index = pc.Index(PINECONE_INDEX_NAME)
stats = index.describe_index_stats()
vector_count = stats.get('total_vector_count', 0)

print(f"Index stats: {stats}")
print(f"Total vectors: {vector_count}")

if vector_count > 0:
    print("\nSUCCESS: Your Pinecone index is populated with vectors!")
    
    # Test retrieval with a simple query
    print("\nTesting retrieval with a sample query...")
    
    # Initialize the embeddings model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=3072
    )
    
    # Connect to the vector store
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        text_key="text"
    )
    
    # Run a test query
    test_query = "TIES software features"
    results = vectorstore.similarity_search(test_query, k=2)
    
    print(f"Retrieved {len(results)} documents for query: '{test_query}'")
    
    # Display the results
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Title: {doc.metadata.get('title', 'No title')}")
        print(f"Article ID: {doc.metadata.get('article_id', 'No ID')}")
        print(f"Content (first 150 chars): {doc.page_content[:150]}...")
else:
    print("\nWARNING: Your Pinecone index appears to be empty!")
    print("Please run the population script first.")

print("\nVerification complete!")