# 🧠 Chatbot Project (FAISS + Gemini)

This project is a simple command-line chatbot that answers questions using an AI model (Google Gemini) and a set of documents stored in a **FAISS** vector index.  
All `.txt` files placed in the `knowledge_base/` folder are automatically processed and used as the chatbot’s knowledge base.

---

## 📌 Features

- Loads multiple `.txt` files as knowledge sources  
- Embeds all documents using **Gemini Embeddings**  
- Stores vectors in a **FAISS index**  
- Uses **Retrieval-Augmented Generation (RAG)**  
- Simple **CLI interface** for interactive chatting  
- Custom system prompt to control chatbot behavior  

---

## 📦 Requirements

- Python **3.8+**
- A Google **Gemini API Key**

Install dependencies:

```bash
pip install -r requirements.txt
