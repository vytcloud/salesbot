"""
🧠 AI Salesbot: Ingest Excel + PDF and test with LM Studio
"""

import os
import time
import shutil
import datetime
import pandas as pd
import pdfplumber
import numpy as np
import requests
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ================================
# CONFIGURATION (edit if needed)
# ================================
EXCEL_PATH = "C:/Users/vytcl/Downloads/Chatbot/CURSTKLOT (32).xls"
PDF_PATH = "C:/Users/vytcl/Downloads/Chatbot/CURRENTSTOCK - 2025-06-12T001611.055.pdf"
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
MODEL_NAME = "qwen2.5-72b-instruct"

# ================================
# FUNCTIONS
# ================================

def clean_sku(sku_raw):
    if pd.isna(sku_raw):
        return 'N/A'
    sku_str = str(sku_raw).strip()
    if sku_str.endswith('.0'):
        return sku_str[:-2]
    try:
        sku_float = float(sku_raw)
        sku_int = int(sku_float)
        return str(sku_int) if sku_float == sku_int else str(sku_float)
    except:
        return sku_str

def safe_remove_directory(path):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
            print("✅ Removed old database")
    except Exception as e:
        print(f"⚠️ Could not remove old database: {e}")

def clean_metadata(metadata):
    cleaned = {}
    for k,v in metadata.items():
        if pd.isna(v) or v is pd.NaT:
            cleaned[k] = None
        elif isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif isinstance(v, np.floating):
            cleaned[k] = float(v)
        elif isinstance(v, np.integer):
            cleaned[k] = int(v)
        else:
            cleaned[k] = str(v)
    return cleaned

def ingest_excel(excel_path):
    df = pd.read_excel(excel_path, skiprows=2)
    print(f"✅ Excel loaded: {len(df)} rows")
    docs = []
    for _, row in df.iterrows():
        sku = clean_sku(row.get('Item', 'N/A'))
        if sku == 'N/A': continue
        product = str(row.get('Description', 'N/A')).strip()
        content = f"""SKU: {sku}
Product: {product}"""
        metadata = {
            "sku": sku,
            "product": product
        }
        docs.append(Document(page_content=content, metadata=clean_metadata(metadata)))
    print(f"✅ Created {len(docs)} Excel documents")
    return docs

def ingest_pdf(pdf_path):
    docs = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    docs.append(Document(page_content=text, metadata={"source":"pdf","page":i+1}))
        print(f"✅ Created {len(docs)} PDF documents")
    except Exception as e:
        print(f"⚠️ PDF error: {e}")
    return docs

def create_vectorstore(all_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cuda'})
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = f"./chroma_db_{timestamp}"
    safe_remove_directory(db_path)
    vs = Chroma.from_documents(documents=all_docs, embedding=embeddings, persist_directory=db_path)
    print(f"✅ Vectorstore created: {db_path} with {len(all_docs)} docs")
    return vs

def query_lmstudio(prompt):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful AI Sales Assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.1,
        "stream": False
    }
    try:
        r = requests.post(f"{LMSTUDIO_BASE_URL}/chat/completions", headers=headers, json=data, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ LM Studio error {r.status_code}: {r.text}"
    except Exception as e:
        return f"❌ LM Studio connection error: {e}"

def test_queries(vs):
    docs = vs.similarity_search("SKU", k=5)
    if not docs:
        print("❌ No SKUs found for testing.")
        return
    for doc in docs:
        sku = doc.metadata.get('sku','N/A')
        question = f"What is the product name of SKU {sku}?"
        prompt = f"""Use the below data to answer:

{doc.page_content}

Question: {question}

Answer briefly."""
        print(f"\n❓ {question}")
        answer = query_lmstudio(prompt)
        print(f"🤖 {answer}")

# ================================
# MAIN
# ================================

if __name__ == "__main__":
    print("🚀 Starting AI Salesbot ingestion + test\n" + "="*50)
    excel_docs = ingest_excel(EXCEL_PATH)
    pdf_docs = ingest_pdf(PDF_PATH)
    all_docs = excel_docs + pdf_docs

    if not all_docs:
        print("❌ No documents created. Exiting.")
        exit()

    vs = create_vectorstore(all_docs)
    test_queries(vs)
    print("\n✅ All done. Ready for next steps.")
