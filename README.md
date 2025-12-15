# 🧠 Chatbot Project (FAISS + Gemini)

This project is a simple command-line chatbot that answers questions using an AI model (Google Gemini) and a set of documents stored in a **FAISS** vector index.  
All `.txt` files placed in the `knowledge_base/` folder are automatically processed and used as the chatbot’s knowledge base.

---

## Project Scope

This project implements a local Retrieval-Augmented Generation (RAG) chatbot that:
✔️ Uses a FAISS vector index for retrieval
✔️ Supports plain `.txt` knowledge bases
✔️ Runs as a command-line interface

This project does _not_ include:
❌ Web UI
❌ User authentication
❌ Advanced safety filters beyond prompt constraints

---

## Guardrails & Responsible Use

To reduce risk of unsafe or hallucinated outputs:

- We use strict system prompts to limit the LLM to context from indexed documents only.
- User queries are checked for emptiness before processing.
- Files loaded into FAISS are filtered to `.txt` formats only.

⚠️ Limitations

- The system does not verify correctness of responses beyond retrieved context.
- It should not be used for critical advice (e.g., medical, legal).

---

## 📌 Features

- Loads multiple `.txt` files as knowledge sources
- Embeds all documents using **Gemini Embeddings**
- Stores vectors in a **FAISS index**
- Uses **Retrieval-Augmented Generation (RAG)**
- Simple **CLI interface** for interactive chatting
- Custom system prompt to control chatbot behavior

---

## Sample Interaction

- Ask a question about the documents: What is X?
- Answer: X is defined as…

---

## 📦 Requirements

- Python **3.8+**
- A Google **Gemini API Key**

⚙️ Setup & Run

Follow these steps to run the chatbot:

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create a .env file in the project root and add your Gemini API key

GOOGLE_API_KEY= YOUR_API_KEY

### 3. Run the chatbot

```bash
python main.py
```
