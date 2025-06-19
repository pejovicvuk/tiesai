# pip install "pinecone[grpc]"
from pinecone.grpc import PineconeGRPC as Pinecone

pc = Pinecone(api_key="pcsk_4u38XC_4qEzBpioJP4s5rWgTRviYrrAcG7SBEMvQSPVcNETXxPP1PerQwSrkqGRNzZZkgD")

index = pc.Index(host="https://ties-docs-1a22if3.svc.aped-4627-b74a.pinecone.io")

# First, let's check what namespaces exist
print("Checking available namespaces...")
try:
    stats = index.describe_index_stats()
    namespaces = stats.get('namespaces', {})
    print(f"Available namespaces: {list(namespaces.keys())}")
    
    if 'ties-docs' in namespaces:
        print("Deleting all records from namespace 'ties-docs'...")
        index.delete(delete_all=True, namespace='ties-docs')
        print("Successfully deleted all records from 'ties-docs' namespace!")
    else:
        print("Namespace 'ties-docs' not found. Available namespaces:")
        for ns, info in namespaces.items():
            print(f"  - {ns}: {info.get('vector_count', 0)} vectors")
        
        # Option to delete from all namespaces
        if namespaces:
            print("\nDeleting all records from ALL namespaces...")
            index.delete(delete_all=True)
            print("Successfully deleted all records from all namespaces!")
        else:
            print("No namespaces found in the index.")
            
except Exception as e:
    print(f"Error: {e}")