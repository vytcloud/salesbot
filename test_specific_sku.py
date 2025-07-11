import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import requests

def test_lm_studio_connection():
    """Test connection to LM Studio"""
    try:
        response = requests.get("http://localhost:1234/v1/models")
        if response.status_code == 200:
            print("✅ LM Studio connection successful!")
            return True
        else:
            print(f"❌ LM Studio connection failed. Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ LM Studio connection error: {e}")
        return False

def create_custom_llm():
    """Create LLM that connects to LM Studio"""
    class LMStudioLLM:
        def __init__(self):
            self.base_url = "http://localhost:1234/v1"
        
        def __call__(self, prompt):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": "local-model",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 500
                    }
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    return f"Error: {response.status_code}"
            except Exception as e:
                return f"Error: {str(e)}"
    
    return LMStudioLLM()

def main():
    print("🔍 Searching for SKU 17004224...")
    print("=" * 60)
    
    # Test LM Studio connection
    if not test_lm_studio_connection():
        print("❌ Please make sure LM Studio is running on localhost:1234")
        return
    
    # Initialize embeddings
    print("🧠 Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'}
    )
    
    # Load the latest vector database
    db_path = "./chroma_db_20250711_030207"  # Use the latest timestamp
    print(f"📁 Loading database from: {db_path}")
    
    try:
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        print("✅ Database loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return
    
    # Create custom LLM
    llm = create_custom_llm()
    
    # Search for the specific SKU
    sku_query = "17004224"
    print(f"\n🔍 Searching for SKU: {sku_query}")
    print("-" * 40)
    
    try:
        # Get relevant documents
        docs = vectorstore.similarity_search(sku_query, k=5)
        
        if docs:
            print(f"✅ Found {len(docs)} matching documents:")
            
            # Show all matching documents
            for i, doc in enumerate(docs, 1):
                print(f"\n📋 Document {i}:")
                print("-" * 30)
                print(doc.page_content)
                print("-" * 30)
                
                # Check if this is the exact SKU match
                if "17004224" in doc.page_content:
                    print("🎯 EXACT MATCH FOUND!")
                    
                    # Now ask the AI to extract specific information
                    prompt = f"""Based on this inventory data, please extract all information about SKU 17004224:

{doc.page_content}

Please provide:
1. Product name/description
2. Stock quantity
3. WAC (Weighted Average Cost)
4. Supplier
5. Lot number
6. Expiry date
7. Any other relevant details

Answer in a clear, organized format."""
                    
                    print("\n🤖 AI Analysis:")
                    print("=" * 40)
                    response = llm(prompt)
                    print(response)
                    print("=" * 40)
        else:
            print("❌ No documents found matching SKU 17004224")
            
            # Let's also try a broader search
            print("\n🔍 Trying broader search patterns...")
            search_patterns = [
                "SKU: 17004224",
                "17004224",
                "Item: 17004224"
            ]
            
            for pattern in search_patterns:
                docs = vectorstore.similarity_search(pattern, k=3)
                if docs:
                    print(f"✅ Found results for pattern: '{pattern}'")
                    for doc in docs:
                        if "17004224" in doc.page_content:
                            print("🎯 EXACT MATCH!")
                            print(doc.page_content)
                            break
                else:
                    print(f"❌ No results for pattern: '{pattern}'")
                    
    except Exception as e:
        print(f"❌ Error searching: {e}")
    
    print("\n" + "=" * 60)
    print("🔍 Search completed!")

if __name__ == "__main__":
    main()