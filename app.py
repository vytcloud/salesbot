"""
✅ **Final code snippet to display all matches in a clean table with formatted integers and decimals**
This integrates your verified formatting logic into your current Streamlit app structure.
"""

import streamlit as st
import pandas as pd
import requests

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
MODEL_NAME = "qwen2.5-72b-instruct"

def query_lmstudio(prompt):
    resp = requests.post(
        f"{LMSTUDIO_BASE_URL}/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful AI Sales Assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.1,
            "stream": False
        },
        timeout=60
    )
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"❌ Error {resp.status_code}: {resp.text}"

st.title("🧠 AI Sales Assistant")

uploaded_file = st.file_uploader("Upload your CURRENTSTOCK Excel file", type=["xls", "xlsx"])
if not uploaded_file:
    st.stop()

df_raw = pd.read_excel(uploaded_file, header=None)
header_row = next((i for i in range(len(df_raw)) if 'ITEM CODE' in df_raw.iloc[i].astype(str).str.upper().values), None)
if header_row is None:
    st.error("❌ 'ITEM CODE' header not found.")
    st.stop()

df = pd.read_excel(uploaded_file, skiprows=header_row)

def clean_sku(x):
    try:
        xf = float(x)
        return str(int(xf)) if xf.is_integer() else str(round(xf, 2))
    except:
        return str(x).strip()

df['SKU'] = df['ITEM CODE'].apply(clean_sku)
df = df[~df['SKU'].str.upper().isin(['SITE', 'CATEGORY'])]

def fmt_qty(x):
    try:
        xf = float(x)
        return int(xf) if xf.is_integer() else round(xf,2)
    except:
        return x

for c in ['AVAILABLE QTY','INTERNAL QTY']:
    df[c] = df[c].apply(fmt_qty)
for c in ['WAC','VALUE']:
    df[c] = df[c].apply(lambda x: round(float(x),2) if pd.notna(x) else x)

question = st.text_input("Ask a question about any SKU or product")
if not question:
    st.stop()

mask = df['SKU'].str.contains(question, case=False, na=False) | df['ITEM DESCRIPTION'].str.contains(question, case=False, na=False)
matches = df[mask]
if matches.empty:
    st.warning("❌ No matches found.")
    st.stop()

st.success(f"🔎 Found {len(matches)} matching result(s)")

# Display all matches in a clean styled table
styled = matches.style.format({
    "AVAILABLE QTY": "{:.0f}",
    "INTERNAL QTY": "{:.0f}",
    "WAC": "{:.2f}",
    "VALUE": "{:.2f}"
})

st.dataframe(styled, use_container_width=True)

# Generate LM Studio answers for each match
for idx, row in matches.reset_index(drop=True).iterrows():
    prompt = "Using this data:\n" + "\n".join(f"{col}: {row[col]}" for col in matches.columns) + f"\n\nQuestion: {question}\nAnswer clearly."
    answer = query_lmstudio(prompt)
    st.markdown(f"**Result {idx+1} answer:** {answer}")

"""
✅ **Run this code** to display all matched SKUs in a formatted table with clean integers and LM Studio answers below.
"""
