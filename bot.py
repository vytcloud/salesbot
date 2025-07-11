# ✅ bot.py (final working version with .complete fix)

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import lmstudio as lms

# Initialize embeddings and ChromaDB
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"}
)
db = Chroma(
    collection_name="salesbot_db",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Load LM Studio model
llm = lms.llm("qwen2.5-72b-instruct")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text

    docs = db.similarity_search(user_query, k=3)
    context_text = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are VYT's AI Sales Assistant. Answer the user's question using only the below context data. If the answer is not found, say 'I don't have that information right now.'

Context:
{context_text}

Question:
{user_query}

Answer:
"""

    response = await llm.complete(prompt)
    await update.message.reply_text(response)

def main():
    app = ApplicationBuilder().token("8138623819:AAGY5CSCK46TyUnzHCyD4eKVmcJ6SPnkf34").build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Bot is running. Press CTRL+C to stop.")
    app.run_polling()

if __name__ == "__main__":
    main()
