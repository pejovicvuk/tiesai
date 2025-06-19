import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

def test_specific_queries():
    """Test with more specific queries that should match your TIES content better"""
    
    # Setup
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index("ties-docs")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1024,
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    # More specific queries based on your actual content
    specific_queries = [
        # Should match version 25.0 content
        "TIES version 25 unified launcher navigation menu",
        
        # Should match trading enhancements
        "buy sell order scheduling connections flexibility",
        
        # Should match confirmations 
        "confirmation center email PDF signatures",
        
        # Should match security features
        "Azure Active Directory virtual users authentication",
        
        # Should match UX improvements
        "dashboard layout widgets drag drop resizing",
        
        # Should match hub nominations
        "Customer Activity Website hub nominations transfer",
        
        # Very specific to your chunks
        "Deal Margin Profit daily margin deals Credit Family Report"
    ]
    
    print("🎯 Testing with specific TIES-related queries...\n")
    
    for query in specific_queries:
        print(f"🔍 Query: '{query}'")
        
        # Embed and search
        query_embedding = embeddings.embed_query(query)
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            filter={"chunk_type": "ties_documentation"}
        )
        
        print(f"   Found {len(results.matches)} matches")
        
        for i, match in enumerate(results.matches):
            score = match.score
            version = match.metadata.get('version', 'N/A')
            feature_type = match.metadata.get('feature_type', 'N/A')
            functionality = match.metadata.get('functionality', 'N/A')
            
            # Get content preview - more relevant parts
            content = match.metadata.get('content', '')
            # Try to find the most relevant sentence
            sentences = content.split('. ')
            preview = sentences[0][:120] + '...' if sentences else content[:120] + '...'
            
            print(f"   {i+1}. Score: {score:.4f} | Version: {version} | Type: {feature_type}")
            print(f"      Functionality: {functionality}")
            print(f"      Preview: {preview}")
            print()
        
        print("-" * 80)

def analyze_score_quality():
    """Analyze what different score ranges mean for your use case"""
    print("\n📊 Score Quality Guide for TIES Documentation:")
    print("🟢 0.65+ : Excellent match - very relevant content")
    print("🟡 0.45-0.65 : Good match - relevant, usable content") 
    print("🟠 0.30-0.45 : Moderate match - somewhat relevant")
    print("🔴 0.00-0.30 : Poor match - probably not relevant")
    print()
    print("💡 For technical documentation, 0.4+ scores are often good results!")
    print("💡 Your 0.46, 0.40, 0.38 scores indicate the system is working correctly")

if __name__ == "__main__":
    test_specific_queries()
    analyze_score_quality()