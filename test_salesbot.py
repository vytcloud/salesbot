# ✅ test_database.py - Test if ChromaDB is working properly

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

print("🔄 Starting database test...")

# ✅ Initialize embeddings (same as your working ingest.py)
print("📊 Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"}
)
print("✅ Embeddings loaded successfully!")

# ✅ Connect to your existing ChromaDB
print("🗄️ Connecting to ChromaDB...")
db = Chroma(
    collection_name="salesbot_db",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
print("✅ ChromaDB connected successfully!")

# ✅ Test if data exists
print("📋 Checking if data exists in database...")
try:
    # Get total count of documents
    collection = db._collection
    count = collection.count()
    print(f"📊 Total documents in database: {count}")
    
    if count == 0:
        print("⚠️ WARNING: No documents found in database!")
        print("❌ You may need to run ingest.py again")
    else:
        print("✅ Database contains data!")
        
        # Test a simple search
        print("\n🔍 Testing search functionality...")
        test_query = "SKU 16500503"
        docs = db.similarity_search(test_query, k=3)
        
        print(f"📝 Found {len(docs)} documents for query: '{test_query}'")
        
        if docs:
            print("\n📄 First document preview:")
            print("-" * 50)
            print(docs[0].page_content[:200] + "...")
            print("-" * 50)
            print("✅ Database search is working perfectly!")
        else:
            print("⚠️ No documents found for test query")
            
except Exception as e:
    print(f"❌ Error testing database: {e}")

print("\n🎉 Database test complete!")