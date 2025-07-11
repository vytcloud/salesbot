# ✅ test_salesbot_fixed.py - Working version with proper LM Studio connection

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import json

print("🔄 Starting AI Sales Assistant test...")

# ✅ Initialize embeddings (same as your working database test)
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

# ✅ Test LM Studio connection
print("🤖 Testing LM Studio connection...")
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

def ask_llm(prompt_text):
    """Send a prompt to LM Studio and get response"""
    try:
        payload = {
            "model": "qwen2.5-72b-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        print("📡 Sending request to LM Studio...")
        response = requests.post(LM_STUDIO_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Connection error: {str(e)}"

# ✅ Test the complete sales assistant workflow
print("\n🔍 Testing complete sales assistant workflow...")

# Test query
user_query = "What is the stock of SKU 16500503?"
print(f"❓ User Query: {user_query}")

# Step 1: Search database
print("📋 Searching database...")
docs = db.similarity_search(user_query, k=3)
context_text = "\n".join([doc.page_content for doc in docs])

print(f"📊 Found {len(docs)} relevant documents")

# Step 2: Create prompt
prompt = f"""You are VYT's AI Sales Assistant. Answer the user's question using only the context data below. Be helpful and specific.

Context:
{context_text}

Question: {user_query}

Answer:"""

print("🤖 Asking LM Studio for response...")

# Step 3: Get LLM response
response = ask_llm(prompt)

print("\n" + "="*60)
print("🎯 AI SALES ASSISTANT RESPONSE:")
print("="*60)
print(response)
print("="*60)

print("\n✅ Test complete!")