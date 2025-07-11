# ✅ debug_data.py - See exactly what data the AI receives

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("🔍 Debugging: What data does the AI actually see?")

# ✅ Initialize embeddings and database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"}
)

db = Chroma(
    collection_name="salesbot_db",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# ✅ Test the exact same query
user_query = "What is the stock of SKU 16500503?"
print(f"🔍 Searching for: {user_query}")

# Get the documents
docs = db.similarity_search(user_query, k=3)

print(f"\n📊 Found {len(docs)} documents")
print("\n" + "="*80)

# Show each document the AI receives
for i, doc in enumerate(docs, 1):
    print(f"📄 DOCUMENT {i}:")
    print("-" * 50)
    print(doc.page_content)
    print("-" * 50)
    print()

print("="*80)
print("🤔 ANALYSIS:")
print("- Does any document mention SKU 16500503?")
print("- Does any document show stock quantities?")
print("- What format is the data in?")
print("="*80)