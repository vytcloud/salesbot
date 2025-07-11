import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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
    print("🤖 Testing AI Sales Assistant with Real Data...")
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
    
    # Create custom prompt template
    prompt_template = """You are a helpful AI sales assistant with access to inventory data.
    
    Use the following context to answer the user's question about inventory, stock levels, pricing, or product information.
    
    Context: {context}
    
    Question: {question}
    
    Instructions:
    - Answer based on the provided context
    - If you find specific data (SKU, quantity, price, etc.), provide the exact numbers
    - If the information isn't in the context, say so clearly
    - Be helpful and concise
    - For pricing questions, WAC refers to Weighted Average Cost
    
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Test queries
    test_queries = [
        "What's the stock quantity of SKU 10000010?",
        "What's the WAC of SKU 10000010?",
        "Who is the supplier for SKU 10000010?",
        "What's the lot number for SKU 10000010?",
        "Show me information about Sima Flexback Cake Margarine"
    ]
    
    print("\n🧪 Testing with sample queries...")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Test Query {i}: {query}")
        print("-" * 40)
        
        try:
            # Get relevant documents
            docs = vectorstore.similarity_search(query, k=3)
            
            if docs:
                # Prepare context
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Create full prompt
                full_prompt = PROMPT.format(context=context, question=query)
                
                # Get LLM response
                response = llm(full_prompt)
                print(f"🤖 Answer: {response}")
            else:
                print("❌ No relevant documents found")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Testing completed!")
    print("\n💡 Your AI Sales Assistant is working with real data!")
    print("🚀 Ready to deploy as a Telegram bot or web interface!")
    
    # Interactive mode
    print("\n🔥 Interactive Mode - Ask your own questions!")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        user_query = input("\n❓ Your Question: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_query:
            continue
        
        try:
            # Get relevant documents
            docs = vectorstore.similarity_search(user_query, k=3)
            
            if docs:
                # Prepare context
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Create full prompt
                full_prompt = PROMPT.format(context=context, question=user_query)
                
                # Get LLM response
                response = llm(full_prompt)
                print(f"🤖 Answer: {response}")
            else:
                print("❌ No relevant information found for your query")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()