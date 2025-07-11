import pandas as pd
import pdfplumber
import os
import shutil
import time
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def safe_remove_directory(path, max_attempts=3):
    """Safely remove directory with retry logic for Windows"""
    for attempt in range(max_attempts):
        try:
            if os.path.exists(path):
                print(f"🗑️ Attempting to remove old database (attempt {attempt + 1}/{max_attempts})...")
                shutil.rmtree(path)
                print("✅ Old database removed successfully!")
                return True
        except PermissionError as e:
            print(f"⚠️ Permission error: {e}")
            if attempt < max_attempts - 1:
                print("⏳ Waiting 2 seconds before retry...")
                time.sleep(2)
            else:
                print("⚠️ Could not remove old database. Creating new one with different name...")
                return False
        except Exception as e:
            print(f"❌ Error removing database: {e}")
            return False
    return False

def clean_metadata(metadata):
    """Clean metadata to ensure only simple types (str, int, float, bool, None)"""
    cleaned = {}
    for key, value in metadata.items():
        if pd.isna(value) or value is pd.NaT:
            cleaned[key] = None
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, np.floating):
            cleaned[key] = float(value)
        elif isinstance(value, np.integer):
            cleaned[key] = int(value)
        else:
            cleaned[key] = str(value)
    return cleaned

def create_documents_from_excel(excel_path):
    """Create documents from Excel data with proper column mapping"""
    print("🔄 Starting fixed data ingestion...")
    
    # Read Excel file, skipping first 2 rows to get to actual headers
    df = pd.read_excel(excel_path, skiprows=2)
    print("✅ Excel file loaded successfully!")
    print(f"📊 Total rows: {len(df)}")
    
    documents = []
    
    for index, row in df.iterrows():
        # Extract relevant fields with proper error handling
        try:
            sku = str(row.get('Item', 'N/A')).strip()
            product = str(row.get('Description', 'N/A')).strip()
            site = str(row.get('Site', 'N/A')).strip()
            stock_qty = row.get('STK qty', 0)
            wac = row.get('Cur WAC', 0)
            cost = row.get('Cost', wac)  # Use WAC as cost if Cost column doesn't exist
            total_value = row.get('STK Value', 0)
            lot = str(row.get('Lot', 'N/A')).strip()
            supplier = str(row.get('Supplier', 'N/A')).strip()
            expiry_date = row.get('Exp date', 'N/A')
            production_date = row.get('Prd date', 'N/A')
            receipt_date = row.get('Receipt date', 'N/A')
            age = row.get('Age', 0)
            
            # Skip rows with missing critical data
            if sku == 'N/A' or sku == 'nan' or pd.isna(sku):
                continue
            
            # Format dates safely
            expiry_str = "N/A" if pd.isna(expiry_date) or expiry_date == 'N/A' else str(expiry_date)
            production_str = "N/A" if pd.isna(production_date) or production_date == 'N/A' else str(production_date)
            receipt_str = "N/A" if pd.isna(receipt_date) or receipt_date == 'N/A' else str(receipt_date)
                
            # Create formatted document content
            content = f"""SKU: {sku}
Product: {product}
Site: {site}
Stock Quantity: {stock_qty}
WAC (Weighted Average Cost): {wac}
Cost: {cost}
Total Value: {total_value}
Lot: {lot}
Supplier: {supplier}
Expiry Date: {expiry_str}
Production Date: {production_str}
Receipt Date: {receipt_str}
Age: {age} days"""
            
            # Create document with safe metadata (only simple types)
            metadata = {
                "source": "excel_inventory",
                "sku": str(sku),
                "product": str(product),
                "stock_qty": float(stock_qty) if not pd.isna(stock_qty) else 0.0,
                "wac": float(wac) if not pd.isna(wac) else 0.0,
                "supplier": str(supplier),
                "lot": str(lot)
            }
            
            doc = Document(
                page_content=content,
                metadata=clean_metadata(metadata)
            )
            documents.append(doc)
            
        except Exception as e:
            print(f"⚠️ Error processing row {index}: {e}")
            continue
    
    print(f"✅ Created {len(documents)} documents from Excel data")
    
    # Show sample document
    if documents:
        print("📄 SAMPLE DOCUMENT:")
        print("-" * 50)
        print(documents[0].page_content)
        print("-" * 50)
    
    return documents

def create_documents_from_pdf(pdf_path):
    """Create documents from PDF file"""
    documents = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": "pdf_inventory", 
                            "page": page_num + 1
                        }
                    )
                    documents.append(doc)
        
        print("✅ PDF file processed successfully")
        
    except Exception as e:
        print(f"⚠️ Error processing PDF: {e}")
    
    return documents

def main():
    # File paths
    excel_path = "C:/Users/vytcl/Downloads/Chatbot/CURSTKLOT (32).xls"
    pdf_path = "C:/Users/vytcl/Downloads/Chatbot/CURRENTSTOCK - 2025-06-12T001611.055.pdf"
    
    # Create documents from both sources
    excel_docs = create_documents_from_excel(excel_path)
    pdf_docs = create_documents_from_pdf(pdf_path)
    
    # Combine all documents
    all_documents = excel_docs + pdf_docs
    print(f"🗄️ Total documents to store: {len(all_documents)}")
    
    if not all_documents:
        print("❌ No documents were created. Please check your data files.")
        return
    
    # Initialize embeddings
    print("🧠 Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'}
    )
    
    # Try to remove old database safely
    db_path = "./chroma_db"
    if not safe_remove_directory(db_path):
        # If we can't remove it, use a timestamped name
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        db_path = f"./chroma_db_{timestamp}"
        print(f"📁 Using new database path: {db_path}")
    
    # Create new vector store
    print("🔄 Creating vector database...")
    try:
        # Clean all documents to ensure compatible metadata
        cleaned_documents = []
        for doc in all_documents:
            cleaned_doc = Document(
                page_content=doc.page_content,
                metadata=clean_metadata(doc.metadata)
            )
            cleaned_documents.append(cleaned_doc)
        
        vectorstore = Chroma.from_documents(
            documents=cleaned_documents,
            embedding=embeddings,
            persist_directory=db_path
        )
        print("✅ Vector database created successfully!")
        print(f"📊 Stored {len(cleaned_documents)} documents in database")
        
        # Test the database
        print("\n🧪 Testing database with sample query...")
        results = vectorstore.similarity_search("SKU 10000010", k=3)
        print(f"✅ Found {len(results)} matching documents")
        
        if results:
            print("\n📋 Sample result:")
            print(results[0].page_content[:200] + "...")
            
    except Exception as e:
        print(f"❌ Error creating vector database: {e}")
        return
    
    print("\n🎉 Data ingestion completed successfully!")
    print(f"📁 Database location: {db_path}")
    print("\n🚀 Ready to test your chatbot!")

if __name__ == "__main__":
    main()