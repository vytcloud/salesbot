# ✅ ingest_fixed.py - Properly reads your Excel data structure

import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import pdfplumber

print("🔄 Starting fixed data ingestion...")

# ✅ STEP 1: Read Excel file with correct structure
excel_file = "C:/Users/vytcl/Downloads/Chatbot/CURSTKLOT (32).xls"
documents = []

try:
    # Read Excel file, skipping the first row (header info)
    df = pd.read_excel(excel_file, skiprows=1)
    
    # Set proper column names based on row 2 (index 0 after skipping)
    df.columns = [
        'Site', 'Item', 'Description', 'Lot', 'Exp_date', 
        'Prd_Date', 'Rcpt_Date', 'Age', 'PKU', 'STK_qty', 
        'Cur_WAC', 'Value', 'Supplier', 'Extra'
    ]
    
    # Remove the header row that became row 0
    df = df.iloc[1:].reset_index(drop=True)
    
    print(f"✅ Excel file loaded successfully!")
    print(f"📊 Total rows: {len(df)}")
    
    # ✅ STEP 2: Convert each row to a meaningful document
    for index, row in df.iterrows():
        # Skip rows with empty or invalid Item numbers
        if pd.isna(row['Item']) or str(row['Item']).strip() == '':
            continue
            
        # Create detailed text for each product
        text = f"""
SKU: {row['Item']}
Product: {row['Description']}
Site: {row['Site']}
Stock Quantity: {row['STK_qty']}
WAC (Weighted Average Cost): {row['Cur_WAC']}
Cost: {row['Cur_WAC']}
Total Value: {row['Value']}
Lot: {row['Lot']}
Supplier: {row['Supplier']}
Expiry Date: {row['Exp_date']}
Production Date: {row['Prd_Date']}
Receipt Date: {row['Rcpt_Date']}
Age: {row['Age']} days
        """.strip()
        
        # Create Document object
        doc = Document(page_content=text)
        documents.append(doc)
    
    print(f"✅ Created {len(documents)} documents from Excel data")
    
    # Show sample of what we created
    if documents:
        print("\n📄 SAMPLE DOCUMENT:")
        print("-" * 50)
        print(documents[0].page_content)
        print("-" * 50)
    
except Exception as e:
    print(f"❌ Error loading Excel file: {e}")

# ✅ STEP 3: Read PDF file
pdf_file = "C:/Users/vytcl/Downloads/Chatbot/CURRENTSTOCK - 2025-06-12T001611.055.pdf"

try:
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                doc = Document(
                    page_content=f"PDF Page {page_num}:\n{text}",
                    metadata={"source": "PDF", "page": page_num}
                )
                documents.append(doc)
    
    print(f"✅ PDF file processed successfully")
    
except Exception as e:
    print(f"❌ Error loading PDF file: {e}")

# ✅ STEP 4: Clear old database and create new one
print(f"\n🗄️ Total documents to store: {len(documents)}")

if documents:
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"}
    )
    
    # Delete old database folder if it exists
    import shutil
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        print("🗑️ Removed old database")
    
    # Create new ChromaDB
    db = Chroma(
        collection_name="salesbot_db",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Add all documents
    db.add_documents(documents)
    
    print(f"✅ Successfully stored {len(documents)} documents in ChromaDB!")
    print("🎉 Data ingestion complete!")
    
else:
    print("❌ No documents to store")

print("\n🚀 Ready to test your AI Sales Assistant!")