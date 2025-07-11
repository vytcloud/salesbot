#!/usr/bin/env python3
"""
Enhanced AI Sales Assistant Test Script for Current Stock PDF
Optimized for your existing setup with ChromaDB and LangChain
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# PDF Processing
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    print("Installing PDF processing libraries...")
    os.system("pip install PyPDF2 pdfplumber")
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True

# Vector Database & Embeddings
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    VECTOR_DB_AVAILABLE = True
except ImportError:
    print("Installing vector database libraries...")
    os.system("pip install chromadb sentence-transformers")
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    VECTOR_DB_AVAILABLE = True

# LangChain
try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("Installing LangChain...")
    os.system("pip install langchain")
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True

class EnhancedSalesbotTester:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.documents = []
        self.inventory_data = []
        self.db_path = f"chroma_db_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.collection_name = "inventory_test"
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
    def setup_embeddings(self):
        """Initialize the embedding model (same as your existing setup)"""
        print("🔧 Setting up embeddings...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Embeddings model loaded successfully")
        
    def setup_vector_database(self):
        """Initialize ChromaDB (matching your existing setup)"""
        print("🔧 Setting up ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✅ Created new collection: {self.collection_name}")
        except Exception as e:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"✅ Using existing collection: {self.collection_name}")
            
    def extract_pdf_data(self):
        """Extract inventory data from PDF with enhanced parsing"""
        print(f"📄 Extracting data from: {self.pdf_path}")
        
        if not os.path.exists(self.pdf_path):
            print(f"❌ PDF file not found: {self.pdf_path}")
            return False
            
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                all_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
                        
            # Parse inventory data from text
            self.parse_inventory_text(all_text)
            print(f"✅ Successfully extracted {len(self.inventory_data)} inventory items")
            return True
            
        except Exception as e:
            print(f"❌ Error extracting PDF: {str(e)}")
            return False
            
    def parse_inventory_text(self, text: str):
        """Parse inventory data from extracted text"""
        lines = text.split('\n')
        current_item = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for item codes (SKUs) - typically 8 digits
            sku_match = re.search(r'\b(\d{8})\b', line)
            if sku_match:
                # Save previous item if exists
                if current_item:
                    self.inventory_data.append(current_item.copy())
                    
                # Start new item
                current_item = {
                    'sku': sku_match.group(1),
                    'line_text': line,
                    'description': '',
                    'category': '',
                    'internal_qty': 0,
                    'available_qty': 0,
                    'wac': 0.0,
                    'total_value': 0.0,
                    'uom': ''
                }
                
                # Extract description (text after SKU)
                desc_part = line[sku_match.end():].strip()
                if desc_part:
                    current_item['description'] = desc_part.split()[0:5]  # First few words
                    current_item['description'] = ' '.join(current_item['description'])
                    
            # Look for category indicators
            if any(cat in line for cat in ['EGG', 'FIN', 'ING', 'MTI']):
                for cat in ['EGG', 'FIN', 'ING', 'MTI']:
                    if cat in line:
                        current_item['category'] = cat
                        break
                        
            # Look for quantities and values
            numbers = re.findall(r'\d+\.?\d*', line)
            if len(numbers) >= 2 and current_item:
                try:
                    current_item['internal_qty'] = float(numbers[0])
                    current_item['available_qty'] = float(numbers[1])
                    if len(numbers) >= 3:
                        current_item['wac'] = float(numbers[2])
                    if len(numbers) >= 4:
                        current_item['total_value'] = float(numbers[3])
                except ValueError:
                    pass
                    
            # Look for UOM
            uom_match = re.search(r'\b(CTN|KG|LTR|PCE|BOX)\b', line)
            if uom_match and current_item:
                current_item['uom'] = uom_match.group(1)
                
        # Don't forget the last item
        if current_item:
            self.inventory_data.append(current_item)
            
    def create_documents(self):
        """Create LangChain documents from inventory data"""
        print("📚 Creating documents for vector database...")
        
        valid_items = 0
        for item in self.inventory_data:
            # Skip items without essential data
            if not item.get('sku') and not item.get('line_text'):
                continue
                
            # Use safe get methods with defaults
            sku = item.get('sku', 'Unknown')
            description = item.get('description', 'No description')
            category = item.get('category', 'Unknown')
            internal_qty = item.get('internal_qty', 0)
            available_qty = item.get('available_qty', 0)
            wac = item.get('wac', 0.0)
            total_value = item.get('total_value', 0.0)
            uom = item.get('uom', 'Unknown')
            
            # Create comprehensive document text
            doc_text = f"""
            SKU: {sku}
            Description: {description}
            Category: {category}
            Internal Quantity: {internal_qty} {uom}
            Available Quantity: {available_qty} {uom}
            WAC (Weighted Average Cost): {wac}
            Total Value: {total_value}
            Unit of Measure: {uom}
            Raw Text: {item.get('line_text', '')}
            """
            
            # Create metadata (clean for ChromaDB)
            metadata = {
                'sku': str(sku),
                'description': str(description)[:50],  # Truncate for ChromaDB
                'category': str(category),
                'internal_qty': float(internal_qty),
                'available_qty': float(available_qty),
                'wac': float(wac),
                'total_value': float(total_value),
                'uom': str(uom),
                'doc_type': 'inventory_item'
            }
            
            # Create document
            document = Document(
                page_content=doc_text.strip(),
                metadata=metadata
            )
            self.documents.append(document)
            valid_items += 1
            
        print(f"✅ Created {valid_items} valid documents from {len(self.inventory_data)} extracted items")
        
    def store_in_vector_db(self):
        """Store documents in ChromaDB"""
        print("💾 Storing documents in vector database...")
        
        if not self.documents:
            print("❌ No documents to store")
            return False
            
        try:
            # Prepare data for ChromaDB
            texts = [doc.page_content for doc in self.documents]
            metadatas = [doc.metadata for doc in self.documents]
            ids = [f"doc_{i}" for i in range(len(self.documents))]
            
            # Generate embeddings
            print("🔢 Generating embeddings...")
            embeddings = self.embedding_model.encode(texts)
            
            # Store in ChromaDB
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings.tolist(),
                ids=ids
            )
            
            print(f"✅ Successfully stored {len(self.documents)} documents in vector database")
            return True
            
        except Exception as e:
            print(f"❌ Error storing documents: {str(e)}")
            return False
            
    def query_database(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query the vector database"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else 0
                }
                formatted_results.append(result)
                
            return formatted_results
            
        except Exception as e:
            print(f"❌ Query error: {str(e)}")
            return []
            
    def run_test_queries(self):
        """Run comprehensive test queries"""
        print("\n🧪 Running Test Queries...")
        print("=" * 50)
        
        test_queries = [
            "What is the stock quantity for SKU 16500503?",
            "Tell me about fresh eggs",
            "What items are in the EGG category?",
            "What is the WAC for chocolate items?",
            "Show me items with high stock levels",
            "What is the total value of inventory?",
            "List all CTN items",
            "What products have low availability?",
            "Show me FIN category items",
            "What is the description for SKU 17002311?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            print("-" * 30)
            
            results = self.query_database(query, n_results=3)
            
            if results:
                for j, result in enumerate(results, 1):
                    metadata = result['metadata']
                    print(f"📦 Result {j}:")
                    print(f"   SKU: {metadata.get('sku', 'N/A')}")
                    print(f"   Description: {metadata.get('description', 'N/A')}")
                    print(f"   Category: {metadata.get('category', 'N/A')}")
                    print(f"   Available: {metadata.get('available_qty', 'N/A')} {metadata.get('uom', '')}")
                    print(f"   WAC: {metadata.get('wac', 'N/A')}")
                    print(f"   Value: {metadata.get('total_value', 'N/A')}")
                    print(f"   Relevance: {1 - result['distance']:.3f}")
                    if j < len(results):
                        print()
            else:
                print("❌ No results found")
                
    def generate_inventory_summary(self):
        """Generate comprehensive inventory summary"""
        print("\n📊 Inventory Summary")
        print("=" * 50)
        
        if not self.inventory_data:
            print("❌ No inventory data available")
            return
            
        # Basic statistics
        total_items = len(self.inventory_data)
        total_value = sum(item['total_value'] for item in self.inventory_data)
        
        # Category breakdown
        categories = {}
        for item in self.inventory_data:
            cat = item['category'] or 'Unknown'
            if cat not in categories:
                categories[cat] = {'count': 0, 'total_value': 0}
            categories[cat]['count'] += 1
            categories[cat]['total_value'] += item['total_value']
            
        print(f"📦 Total Items: {total_items}")
        print(f"💰 Total Inventory Value: ${total_value:,.2f}")
        print(f"📂 Categories: {len(categories)}")
        
        print("\n📂 Category Breakdown:")
        for cat, data in categories.items():
            print(f"   {cat}: {data['count']} items, ${data['total_value']:,.2f}")
            
        # Top 5 highest value items
        top_items = sorted(self.inventory_data, key=lambda x: x['total_value'], reverse=True)[:5]
        print("\n🏆 Top 5 Highest Value Items:")
        for i, item in enumerate(top_items, 1):
            print(f"   {i}. SKU {item['sku']}: {item['description']}")
            print(f"      Value: ${item['total_value']:,.2f}")
            
    def interactive_mode(self):
        """Interactive query mode"""
        print("\n🎯 Interactive Query Mode")
        print("=" * 50)
        print("Ask questions about your inventory! (type 'quit' to exit)")
        
        while True:
            try:
                query = input("\n💬 Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                if not query:
                    continue
                    
                print(f"\n🔍 Searching for: {query}")
                results = self.query_database(query, n_results=3)
                
                if results:
                    print("\n📋 Results:")
                    for i, result in enumerate(results, 1):
                        metadata = result['metadata']
                        print(f"\n🎯 Result {i}:")
                        print(f"   SKU: {metadata.get('sku', 'N/A')}")
                        print(f"   Description: {metadata.get('description', 'N/A')}")
                        print(f"   Category: {metadata.get('category', 'N/A')}")
                        print(f"   Available Stock: {metadata.get('available_qty', 'N/A')} {metadata.get('uom', '')}")
                        print(f"   WAC: ${metadata.get('wac', 'N/A')}")
                        print(f"   Total Value: ${metadata.get('total_value', 'N/A')}")
                        print(f"   Relevance: {(1 - result['distance'])*100:.1f}%")
                else:
                    print("❌ No relevant results found. Try a different query.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                
    def run_full_test(self):
        """Run complete test suite"""
        print("🚀 Enhanced AI Sales Assistant Test Suite")
        print("=" * 50)
        
        # Setup
        self.setup_embeddings()
        self.setup_vector_database()
        
        # Data processing
        if not self.extract_pdf_data():
            print("❌ Failed to extract PDF data")
            return False
            
        self.create_documents()
        
        if not self.store_in_vector_db():
            print("❌ Failed to store documents in database")
            return False
            
        # Testing
        self.generate_inventory_summary()
        self.run_test_queries()
        
        # Interactive mode
        self.interactive_mode()
        
        return True

def main():
    """Main execution function"""
    # Your PDF file path
    pdf_path = r"C:\Users\vytcl\Downloads\Chatbot\CURRENTSTOCK - 2025-06-12T001611.055.pdf"
    
    print("🤖 Enhanced AI Sales Assistant (Salesbot) Test")
    print("=" * 50)
    print(f"📄 Target PDF: {pdf_path}")
    
    # Initialize and run test
    tester = EnhancedSalesbotTester(pdf_path)
    success = tester.run_full_test()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("🎉 Your AI Sales Assistant is ready for production!")
    else:
        print("\n❌ Test failed. Please check the error messages above.")

if __name__ == "__main__":
    main()