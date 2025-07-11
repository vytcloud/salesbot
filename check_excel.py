# ✅ check_excel.py - See what columns are actually in your Excel file

import pandas as pd

excel_file = "C:/Users/vytcl/Downloads/Chatbot/CURSTKLOT (32).xls"

print("🔍 Checking Excel file structure...")
print(f"📁 File: {excel_file}")
print("="*60)

try:
    # Read the Excel file
    df = pd.read_excel(excel_file)
    
    print("✅ Excel file loaded successfully!")
    print(f"📊 Total rows: {len(df)}")
    print(f"📊 Total columns: {len(df.columns)}")
    
    print("\n📋 COLUMN NAMES:")
    print("-" * 40)
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    
    print("\n📄 FIRST 3 ROWS OF DATA:")
    print("-" * 60)
    print(df.head(3).to_string())
    
    print("\n🔍 SAMPLE DATA FOR EACH COLUMN:")
    print("-" * 60)
    for col in df.columns:
        # Get first non-null value from this column
        sample_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else "No data"
        print(f"{col}: {sample_value}")
        
except Exception as e:
    print(f"❌ Error reading Excel file: {e}")

print("\n🎯 Now I can fix your ingest.py with the correct column names!")