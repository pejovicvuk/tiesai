import os
import json
import time
from typing import List, Dict, Any
from pinecone import Pinecone
from tqdm import tqdm

# LangChain imports
from langchain_openai import OpenAIEmbeddings

def initialize_pinecone():
    """Initialize Pinecone client"""
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        raise ValueError("❌ Please set PINECONE_API_KEY environment variable")
    
    pc = Pinecone(api_key=api_key)
    return pc

def initialize_openai_embeddings():
    """Initialize OpenAI embeddings via LangChain"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("❌ Please set OPENAI_API_KEY environment variable")
    
    # Use text-embedding-3-large with 1024 dimensions to match your Pinecone index
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1024,  # Match your Pinecone index dimension
        openai_api_key=api_key
    )
    
    print("✅ Initialized OpenAI text-embedding-3-large model")
    return embeddings

def connect_to_index(pc: Pinecone, index_name: str = "ties-docs"):
    """Connect to Pinecone index"""
    try:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        
        print(f"✅ Connected to index: {index_name}")
        print(f"📊 Index dimension: {stats.get('dimension')}")
        print(f"📊 Current vector count: {stats.get('total_vector_count', 0)}")
        
        return index
    except Exception as e:
        print(f"❌ Error connecting to index {index_name}: {e}")
        raise

def load_chunks_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load chunks from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"📁 Loaded {len(chunks)} chunks from {file_path}")
        
        # Show sample
        if chunks:
            sample = chunks[0]
            print(f"📝 Sample chunk ID: {sample['chunk_id']}")
            print(f"📝 Sample version: {sample['metadata'].get('version', 'N/A')}")
            print(f"📝 Sample content: {sample['content'][:100]}...")
        
        return chunks
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {file_path}: {e}")
        raise

def generate_embeddings_batch(texts: List[str], embeddings_model: OpenAIEmbeddings, batch_size: int = 100) -> List[List[float]]:
    """Generate embeddings in batches to handle rate limits"""
    all_embeddings = []
    
    print(f"🔄 Generating embeddings for {len(texts)} texts in batches of {batch_size}")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[i:i + batch_size]
        
        try:
            # Use LangChain's embed_documents method
            batch_embeddings = embeddings_model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Error in embedding batch {i//batch_size + 1}: {e}")
            raise
    
    print(f"✅ Generated {len(all_embeddings)} embeddings")
    print(f"📏 Embedding dimension: {len(all_embeddings[0]) if all_embeddings else 'N/A'}")
    
    return all_embeddings

def prepare_chunks_for_pinecone(chunks: List[Dict[str, Any]], embeddings_model: OpenAIEmbeddings) -> List[Dict[str, Any]]:
    """Prepare chunks with OpenAI embeddings for Pinecone upload"""
    
    # Extract all content for embedding
    texts = [chunk['content'] for chunk in chunks]
    
    # Generate embeddings using LangChain + OpenAI
    embeddings = generate_embeddings_batch(texts, embeddings_model)
    
    # Prepare vectors for Pinecone
    prepared_vectors = []
    for i, chunk in enumerate(chunks):
        vector = {
            'id': chunk['chunk_id'],
            'values': embeddings[i],  # OpenAI embedding vector
            'metadata': {
                **chunk['metadata'],
                # Store original content for retrieval
                'content': chunk['content'],
                'content_length': len(chunk['content']),
                'chunk_type': 'ties_documentation',
                'embedding_model': 'openai-text-embedding-3-large',
                'embedding_dimension': len(embeddings[i])
            }
        }
        prepared_vectors.append(vector)
    
    print(f"✅ Prepared {len(prepared_vectors)} vectors for upload")
    return prepared_vectors

def upload_vectors_to_pinecone(index, vectors: List[Dict[str, Any]], batch_size: int = 50):
    """Upload vectors to Pinecone in batches"""
    print(f"🚀 Uploading {len(vectors)} vectors in batches of {batch_size}")
    
    successful_uploads = 0
    
    for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading batches"):
        batch = vectors[i:i + batch_size]
        
        try:
            response = index.upsert(vectors=batch)
            successful_uploads += len(batch)
            print(f"✅ Uploaded batch {i//batch_size + 1}: {len(batch)} vectors")
            
            # Small delay for rate limiting
            time.sleep(0.2)
            
        except Exception as e:
            print(f"❌ Error uploading batch {i//batch_size + 1}: {e}")
            # Show vector format for debugging
            if batch:
                sample_vector = batch[0]
                print(f"   Sample vector keys: {list(sample_vector.keys())}")
                print(f"   Values length: {len(sample_vector.get('values', []))}")
            continue
    
    print(f"📊 Successfully uploaded {successful_uploads}/{len(vectors)} vectors")
    return successful_uploads

def verify_upload(index, expected_count: int):
    """Verify upload and show sample results"""
    print("\n🔍 Verifying upload...")
    time.sleep(3)  # Wait for indexing
    
    try:
        stats = index.describe_index_stats()
        total_count = stats.get('total_vector_count', 0)
        
        print(f"📊 Total vectors in index: {total_count}")
        print(f"📊 Expected: {expected_count}")
        
        # Test query with dummy vector to check metadata
        dummy_vector = [0.01] * 1024
        results = index.query(
            vector=dummy_vector,
            top_k=5,
            include_metadata=True,
            filter={"chunk_type": "ties_documentation"}
        )
        
        uploaded_count = len(results.matches)
        print(f"✅ Found {uploaded_count} TIES documentation chunks")
        
        if results.matches:
            print("\n📝 Sample uploaded chunks:")
            for i, match in enumerate(results.matches[:3]):
                chunk_id = match.id
                version = match.metadata.get('version', 'N/A')
                feature_type = match.metadata.get('feature_type', 'N/A')
                content_preview = match.metadata.get('content', '')[:80] + '...'
                
                print(f"   {i+1}. ID: {chunk_id}")
                print(f"      Version: {version} | Type: {feature_type}")
                print(f"      Content: {content_preview}")
        
        return uploaded_count
        
    except Exception as e:
        print(f"⚠️ Verification failed: {e}")
        return 0

def test_semantic_search(index, embeddings_model: OpenAIEmbeddings):
    """Test semantic search with real queries"""
    print("\n🎯 Testing semantic search...")
    
    test_queries = [
        "TIES version 25 new features and enhancements",
        "trading improvements and order management",
        "unified launcher and navigation updates", 
        "Buy/Sell order scheduling and connections",
        "Azure Active Directory authentication and security",
        "hub nominations and Customer Activity Website",
        "dashboard improvements and UX changes"
    ]
    
    for query in test_queries:
        try:
            # Embed the query using the same OpenAI model
            query_embedding = embeddings_model.embed_query(query)
            
            # Search Pinecone
            results = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True,
                filter={"chunk_type": "ties_documentation"}
            )
            
            print(f"\n🔍 Query: '{query}'")
            print(f"   Found {len(results.matches)} matches")
            
            for j, match in enumerate(results.matches):
                score = match.score
                version = match.metadata.get('version', 'N/A')
                feature_type = match.metadata.get('feature_type', 'N/A')
                software_module = match.metadata.get('software_module', 'N/A')
                
                print(f"   {j+1}. Score: {score:.4f}")
                print(f"      Version: {version} | Module: {software_module} | Type: {feature_type}")
                print(f"      ID: {match.id}")
                
        except Exception as e:
            print(f"   ❌ Query '{query}' failed: {e}")

def main(chunks_file_path: str = "chunks/new_in_release.json"):
    """Main function to upload TIES documentation using LangChain + OpenAI"""
    print("🚀 Starting TIES documentation upload...")
    print("📊 Using: LangChain + OpenAI text-embedding-3-large")
    
    try:
        # Check environment variables
        if not os.getenv('PINECONE_API_KEY'):
            raise ValueError("❌ PINECONE_API_KEY environment variable not set")
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("❌ OPENAI_API_KEY environment variable not set")
        
        # Load data
        chunks = load_chunks_from_file(chunks_file_path)
        
        # Initialize services
        pc = initialize_pinecone()
        index = connect_to_index(pc)
        embeddings_model = initialize_openai_embeddings()
        
        # Prepare and upload
        vectors = prepare_chunks_for_pinecone(chunks, embeddings_model)
        successful_count = upload_vectors_to_pinecone(index, vectors)
        
        # Verify and test
        verified_count = verify_upload(index, len(chunks))
        test_semantic_search(index, embeddings_model)
        
        print(f"\n✅ Process completed successfully!")
        print(f"📊 Uploaded: {successful_count}/{len(chunks)} chunks")
        print(f"📊 Verified: {verified_count} chunks in index")
        print(f"💰 Estimated cost: ~${(len(chunks) * 1000 * 0.00013):.4f} USD")  # Rough estimate
        
    except Exception as e:
        print(f"❌ Process failed: {e}")
        print("\n💡 Troubleshooting checklist:")
        print("   1. Set PINECONE_API_KEY environment variable")
        print("   2. Set OPENAI_API_KEY environment variable") 
        print("   3. Verify chunks/new_in_release.json file exists")
        print("   4. Check Pinecone index 'ties-docs' exists")
        raise

if __name__ == "__main__":
    import sys
    
    # Allow custom file path
    file_path = sys.argv[1] if len(sys.argv) > 1 else "chunks/new_in_release.json"
    print(f"📄 Using file: {file_path}")
    
    main(file_path)