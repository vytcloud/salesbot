"""
AI Sales Assistant - LM Studio Integration Test
This script connects to your LM Studio server to test the sales bot
"""

import os
import requests
import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Configuration
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"  # Default LM Studio API endpoint
DATABASE_PATH = "./chroma_db_20250711_030207"  # Your successful database path

def initialize_vector_store():
    """Initialize the vector database"""
    print("🔄 Initializing vector store...")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load the vector store
    vectorstore = Chroma(
        persist_directory=DATABASE_PATH,
        embedding_function=embeddings
    )
    
    print(f"✅ Vector store loaded from: {DATABASE_PATH}")
    return vectorstore

def query_lmstudio(prompt, max_tokens=500):
    """Send query to LM Studio"""
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "qwen2.5-72b-instruct",  # Adjust model name as needed
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI Sales Assistant. Provide clear, concise answers based on the inventory data provided."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": False
        }
        
        response = requests.post(
            f"{LMSTUDIO_BASE_URL}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error connecting to LM Studio: {str(e)}"

def search_and_answer(vectorstore, question, top_k=3):
    """Search vector database and generate answer using LM Studio"""
    print(f"\n🔍 Searching for: {question}")
    
    # Search the vector database
    docs = vectorstore.similarity_search(question, k=top_k)
    
    if not docs:
        print("❌ No relevant documents found")
        return "No relevant information found in the database."
    
    print(f"✅ Found {len(docs)} relevant documents")
    
    # Prepare context from search results
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt for LM Studio
    prompt = f"""Based on the following inventory data, please answer the question:

INVENTORY DATA:
{context}

QUESTION: {question}

Please provide a clear, specific answer based only on the data provided above."""
    
    print("🤖 Generating answer using LM Studio...")
    
    # Get answer from LM Studio
    answer = query_lmstudio(prompt)
    
    return answer, docs

def test_specific_queries():
    """Test specific queries about your inventory"""
    vectorstore = initialize_vector_store()
    
    # Test queries
    test_queries = [
        "What is the stock quantity of SKU 10000010?",
        "What is the WAC of SKU 10000010?",
        "Who is the supplier for SKU 10000010?",
        "What is the lot number for SKU 10000010?",
        "Tell me about SKU 17004224",
        "What products do we have from supplier S31040?",
        "Show me all products expiring in 2026",
        "What is the total value of our inventory?"
    ]
    
    print("🧪 Testing Sales Bot with LM Studio...")
    print("=" * 60)
    
    for query in test_queries:
        try:
            answer, docs = search_and_answer(vectorstore, query)
            
            print(f"\n📋 QUERY: {query}")
            print(f"🤖 ANSWER: {answer}")
            print(f"📊 Based on {len(docs)} documents")
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Error processing query '{query}': {str(e)}")
    
    return vectorstore

def interactive_chat():
    """Interactive chat mode"""
    vectorstore = initialize_vector_store()
    
    print("\n🎯 Interactive Chat Mode - Ask anything about your inventory!")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n💬 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not question:
                continue
                
            print("🔍 Searching...")
            answer, docs = search_and_answer(vectorstore, question)
            
            print(f"\n🤖 Answer: {answer}")
            print(f"📊 (Based on {len(docs)} relevant documents)")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def check_lmstudio_connection():
    """Check if LM Studio is running and accessible"""
    try:
        response = requests.get(f"{LMSTUDIO_BASE_URL}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("✅ LM Studio connection successful!")
            print(f"📋 Available models: {[model['id'] for model in models['data']]}")
            return True
        else:
            print(f"❌ LM Studio responded with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to LM Studio: {str(e)}")
        print("💡 Make sure LM Studio is running on http://localhost:1234")
        return False

def main():
    """Main function"""
    print("🚀 AI Sales Assistant - LM Studio Test")
    print("=" * 50)
    
    # Check LM Studio connection
    if not check_lmstudio_connection():
        print("\n🔧 Please ensure:")
        print("1. LM Studio is running")
        print("2. A model is loaded (like Qwen2.5-72B)")
        print("3. API server is enabled on localhost:1234")
        return
    
    # Test specific queries
    print("\n🧪 Running automated tests...")
    vectorstore = test_specific_queries()
    
    # Interactive mode
    print("\n🎯 Starting interactive mode...")
    interactive_chat()

if __name__ == "__main__":
    main()