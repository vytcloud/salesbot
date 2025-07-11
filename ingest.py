# ✅ ingest.py

# Import necessary libraries
import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain.docstore.document import Document
import pdfplumber

# ✅ STEP 8.2: Read Excel sales report

# Replace this with your actual Excel file path
excel_file = "C:/Users/vytcl/Downloads/Chatbot/CURSTKLOT (32).xls"

documents = []

# Read the Excel file using pandas
try:
    df = pd.read_excel(excel_file)
    print("✅ Excel file loaded successfully.")

    # ✅ STEP 8.3: Convert Excel rows to text Documents
    if not df.empty:
        for index, row in df.iterrows():
            # Create a simple text description for each row
            text = f"Date: {row.get('Date', 'N/A')}, Customer: {row.get('CustomerName', 'N/A')}, SKU: {row.get('ProductSKU', 'N/A')}, Quantity: {row.get('Quantity', 'N/A')}, Price: {row.get('SellingPrice', 'N/A')}."
            
            # Create a Document object with the text
            doc = Document(page_content=text)
            
            # Add to documents list
            documents.append(doc)

        print(f"✅ Created {len(documents)} documents from Excel data.")
    else:
        print("❌ Excel file is empty.")

except Exception as e:
    print(f"❌ Error loading Excel file: {e}")

# ✅ STEP 8.4: Read PDF report and convert to Documents

# Replace with your actual PDF file path
pdf_file = "C:/Users/vytcl/Downloads/Chatbot/CURRENTSTOCK - 2025-06-12T001611.055.pdf"

try:
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                doc = Document(page_content=text)
                documents.append(doc)

    print("✅ PDF file loaded and converted to documents successfully.")

except Exception as e:
    print(f"❌ Error loading PDF file: {e}")

# ✅ STEP 8.5: Create embeddings and store in ChromaDB

if documents:
    # Initialize embedding model (using a small, fast model)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"}
    )

    # Create or load ChromaDB
    db = Chroma(
        collection_name="salesbot_db",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    db.add_documents(documents)
    print(f"✅ Added {len(documents)} documents to ChromaDB successfully.")
else:
    print("❌ No documents to add to ChromaDB.")
