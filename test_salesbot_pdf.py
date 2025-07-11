#!/usr/bin/env python3
"""
AI Sales Assistant (Salesbot) Test Script for Current Stock PDF
Tests the chatbot's ability to query inventory data from the PDF file
"""

import os
import sys
import pandas as pd
import PyPDF2
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import re
import json

class SalesbotPDFTester:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.embeddings = None
        self.vectorstore = None
        self.db_path = None
        
    def setup_embeddings(self):
        """Initialize embeddings model"""
        print("🔄 Setting up embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Embeddings initialized successfully")
        
    def extract_pdf_data(self):
        """Extract and structure data from PDF"""
        print(f"📄 Processing PDF: {self.pdf_path}")
        
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        
        # Extract text content
        full_text = ""
        for page in pages:
            full_text += page.page_content + "\n"
        
        # Parse inventory data from text
        inventory_items = self.parse_inventory_data(full_text)
        
        # Create documents
        documents = []
        for item in inventory_items:
            # Create detailed document for each item
            doc_content = f"""
            Item Code: {item['item_code']}
            Description: {item['description']}
            Category: {item['category']}
            UOM: {item['uom']}
            Internal Quantity: {item['internal_qty']}
            Available Quantity: {item['available_qty']}
            WAC (Weighted Average Cost): {item['wac']}
            Total Value: {item['value']}
            Site: {item['site']}
            
            Full Item Details:
            - SKU: {item['item_code']}
            - Product Name: {item['description']}
            - Stock Level: {item['internal_qty']} {item['uom']}
            - Available Stock: {item['available_qty']} {item['uom']}
            - Unit Cost: {item['wac']}
            - Total Inventory Value: {item['value']}
            - Product Category: {item['category']}
            - Storage Location: Site {item['site']}
            """
            
            metadata = {
                'item_code': item['item_code'],
                'description': item['description'][:100],  # Truncate for metadata
                'category': item['category'],
                'uom': item['uom'],
                'internal_qty': float(item['internal_qty']),
                'available_qty': float(item['available_qty']),
                'wac': float(item['wac']),
                'value': float(item['value']),
                'site': item['site'],
                'source': 'current_stock_report'
            }
            
            documents.append(Document(
                page_content=doc_content,
                metadata=metadata
            ))
        
        print(f"✅ Extracted {len(documents)} inventory items from PDF")
        return documents
    
    def parse_inventory_data(self, text):
        """Parse inventory data from PDF text"""
        inventory_items = []
        
        # Split text into lines
        lines = text.split('\n')
        
        current_site = None
        current_category = None
        
        for line in lines:
            line = line.strip()
            
            # Check for site information
            if line.startswith('SITE'):
                current_site = line.split()[-1] if line.split() else 'M06'
            
            # Check for category information
            elif line.startswith('CATEGORY'):
                current_category = line.split()[-1] if line.split() else 'Unknown'
            
            # Check for item data (starts with item code)
            elif re.match(r'^\d{8}', line):
                try:
                    # Parse item line
                    parts = line.split()
                    if len(parts) >= 6:
                        item_code = parts[0]
                        
                        # Find description (everything between item code and UOM)
                        # UOM is typically 3 letters like CTN, PAL, BAG, etc.
                        uom_pattern = r'\b(CTN|PAL|BAG|PKT|TIN|CAN|TUB|BTL|PCS|UNI|EW)\b'
                        uom_match = re.search(uom_pattern, line)
                        
                        if uom_match:
                            uom = uom_match.group(1)
                            uom_index = line.find(uom)
                            
                            # Description is between item code and UOM
                            desc_start = len(item_code)
                            description = line[desc_start:uom_index].strip()
                            
                            # Extract numeric values after UOM
                            after_uom = line[uom_index + len(uom):].strip()
                            numbers = re.findall(r'[\d,]+\.?\d*', after_uom)
                            
                            # Clean numbers (remove commas)
                            numbers = [n.replace(',', '') for n in numbers]
                            
                            if len(numbers) >= 4:
                                internal_qty = float(numbers[0])
                                available_qty = float(numbers[1])
                                wac = float(numbers[2])
                                value = float(numbers[3])
                                
                                inventory_items.append({
                                    'item_code': item_code,
                                    'description': description,
                                    'category': current_category or 'Unknown',
                                    'uom': uom,
                                    'internal_qty': internal_qty,
                                    'available_qty': available_qty,
                                    'wac': wac,
                                    'value': value,
                                    'site': current_site or 'M06'
                                })
                        
                except Exception as e:
                    print(f"⚠️ Error parsing line: {line[:50]}... - {str(e)}")
                    continue
        
        return inventory_items
    
    def create_vector_database(self, documents):
        """Create vector database from documents"""
        print("🔄 Creating vector database...")
        
        # Create timestamped database directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.db_path = f"chroma_db_pdf_{timestamp}"
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        
        # Persist the database
        self.vectorstore.persist()
        print(f"✅ Vector database created at: {self.db_path}")
        
    def test_queries(self):
        """Test various queries on the inventory data"""
        print("\n🔍 Testing AI Sales Assistant Queries...")
        print("=" * 50)
        
        test_queries = [
            "What is the stock quantity for SKU 16500503?",
            "Tell me about Fresh Eggs inventory",
            "What items are in the EGG category?",
            "What is the WAC for item 17002311?",
            "Show me all chocolate items",
            "What is the total value of SKU 15100051?",
            "How many CTN of Ambra Rondo are available?",
            "List items with low stock levels",
            "What products are in the ING category?",
            "Show me the most expensive items by WAC"
        ]
        
        results = []
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("-" * 30)
            
            try:
                # Search for relevant documents
                docs = self.vectorstore.similarity_search(query, k=3)
                
                if docs:
                    print("🎯 Results found:")
                    for i, doc in enumerate(docs, 1):
                        print(f"\n{i}. {doc.metadata.get('description', 'No description')}")
                        print(f"   SKU: {doc.metadata.get('item_code', 'N/A')}")
                        print(f"   Category: {doc.metadata.get('category', 'N/A')}")
                        print(f"   Stock: {doc.metadata.get('internal_qty', 'N/A')} {doc.metadata.get('uom', '')}")
                        print(f"   Available: {doc.metadata.get('available_qty', 'N/A')} {doc.metadata.get('uom', '')}")
                        print(f"   WAC: {doc.metadata.get('wac', 'N/A')}")
                        print(f"   Value: {doc.metadata.get('value', 'N/A')}")
                        
                        # Store result
                        results.append({
                            'query': query,
                            'sku': doc.metadata.get('item_code'),
                            'description': doc.metadata.get('description'),
                            'stock': doc.metadata.get('internal_qty'),
                            'wac': doc.metadata.get('wac'),
                            'value': doc.metadata.get('value')
                        })
                else:
                    print("❌ No results found")
                    
            except Exception as e:
                print(f"❌ Error processing query: {str(e)}")
        
        return results
    
    def generate_inventory_summary(self):
        """Generate summary statistics from the inventory"""
        print("\n📊 Inventory Summary Statistics")
        print("=" * 50)
        
        try:
            # Get all documents
            all_docs = self.vectorstore.get()
            
            if all_docs and 'metadatas' in all_docs:
                metadatas = all_docs['metadatas']
                
                # Calculate statistics
                total_items = len(metadatas)
                categories = set()
                total_value = 0
                total_stock = 0
                
                for meta in metadatas:
                    if meta.get('category'):
                        categories.add(meta['category'])
                    if meta.get('value'):
                        total_value += meta['value']
                    if meta.get('internal_qty'):
                        total_stock += meta['internal_qty']
                
                print(f"📈 Total Items: {total_items}")
                print(f"📈 Categories: {len(categories)}")
                print(f"📈 Total Inventory Value: ${total_value:,.2f}")
                print(f"📈 Total Stock Units: {total_stock:,.0f}")
                print(f"📈 Categories Found: {', '.join(sorted(categories))}")
                
                # Top 5 most valuable items
                valuable_items = sorted(metadatas, key=lambda x: x.get('value', 0), reverse=True)[:5]
                print(f"\n💰 Top 5 Most Valuable Items:")
                for i, item in enumerate(valuable_items, 1):
                    print(f"{i}. {item.get('item_code')} - ${item.get('value', 0):,.2f}")
                
        except Exception as e:
            print(f"❌ Error generating summary: {str(e)}")
    
    def interactive_query_mode(self):
        """Interactive mode for testing queries"""
        print("\n🤖 Interactive Query Mode")
        print("=" * 50)
        print("Type your questions about inventory. Type 'quit' to exit.")
        
        while True:
            try:
                query = input("\n🔍 Your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Search for relevant documents
                docs = self.vectorstore.similarity_search(query, k=3)
                
                if docs:
                    print("\n🎯 Results:")
                    for i, doc in enumerate(docs, 1):
                        print(f"\n{i}. {doc.metadata.get('description', 'No description')}")
                        print(f"   SKU: {doc.metadata.get('item_code', 'N/A')}")
                        print(f"   Stock: {doc.metadata.get('internal_qty', 'N/A')} {doc.metadata.get('uom', '')}")
                        print(f"   WAC: ${doc.metadata.get('wac', 'N/A')}")
                        print(f"   Value: ${doc.metadata.get('value', 'N/A')}")
                else:
                    print("❌ No results found for your query.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def run_full_test(self):
        """Run complete test suite"""
        print("🚀 Starting AI Sales Assistant PDF Test")
        print("=" * 50)
        
        try:
            # Setup
            self.setup_embeddings()
            
            # Extract data
            documents = self.extract_pdf_data()
            
            # Create database
            self.create_vector_database(documents)
            
            # Test queries
            results = self.test_queries()
            
            # Generate summary
            self.generate_inventory_summary()
            
            print(f"\n✅ Test completed successfully!")
            print(f"📁 Database saved to: {self.db_path}")
            print(f"📊 Total queries tested: {len(results)}")
            
            # Option for interactive mode
            response = input("\n🤖 Would you like to try interactive query mode? (y/n): ")
            if response.lower().startswith('y'):
                self.interactive_query_mode()
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """Main function"""
    pdf_path = r"C:\Users\vytcl\Downloads\Chatbot\CURRENTSTOCK - 2025-06-12T001611.055.pdf"
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        print("Please make sure the file exists and the path is correct.")
        return
    
    # Create and run tester
    tester = SalesbotPDFTester(pdf_path)
    tester.run_full_test()

if __name__ == "__main__":
    main()