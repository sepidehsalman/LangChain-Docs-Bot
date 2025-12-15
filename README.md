# 🧠 Chatbot Project (FAISS + Gemini)

This project is a command-line chatbot that answers questions using an AI model (**Google Gemini**) and a set of documents stored in a **FAISS** vector index.  
All `.txt` files placed in the `knowledge_base/` folder are automatically processed and used as the chatbot’s knowledge base.

---

## Project Scope

This project implements a local **Retrieval-Augmented Generation (RAG) chatbot** that:

✔️ Uses a FAISS vector index for semantic retrieval  
✔️ Supports plain `.txt` knowledge bases  
✔️ Runs as a command-line interface  
✔️ Logs Top-K retrieved chunks with similarity scores for evaluation  
✔️ Uses a custom prompt template to ensure answers are based strictly on the documents

This project does _not_ include:

❌ Web UI  
❌ User authentication  
❌ Advanced safety filters beyond prompt constraints

---

## Guardrails & Responsible Use

To reduce risk of unsafe or hallucinated outputs:

- System prompts are strict and limit the LLM to context from retrieved documents only.
- User queries are validated to prevent empty inputs.
- Only `.txt` files in the knowledge base are processed.

⚠️ Limitations:

- The system does not verify correctness beyond retrieved context.
- Not suitable for critical advice (e.g., medical, legal).

---

## 📌 Features

- Retrieval-Augmented Generation (RAG) pipeline
- Configurable chunk size, overlap, and retrieval depth via `config.py`
- Google Gemini embeddings for semantic search
- Safe handling of unknown queries (no hallucinations)
- Retrieval evaluation with **Top-K chunk logging and similarity scores**
- Custom **PromptTemplate** to enforce context usage
- Modular design for maintainability

---

## RAG Workflow

1. **Document Ingestion**

   - Reads `.txt` files from `knowledge_base/` and converts them to `Document` objects.

2. **Text Chunking**

   - Splits documents into chunks (`CHUNK_SIZE`, `CHUNK_OVERLAP`) to improve retrieval quality.

3. **Vectorization**

   - Uses `GoogleGenerativeAIEmbeddings` (Gemini model) to embed chunks into vectors for FAISS.

4. **Retrieval**

   - FAISS retrieves Top-K most relevant chunks for each query.
   - Logs retrieved chunks and similarity scores for evaluation.

5. **Answer Generation**
   - `ChatGoogleGenerativeAI` generates answers based strictly on retrieved context.
   - Custom prompt template ensures safe refusals for queries outside the KB:

```text
You are a helpful assistant that answers questions strictly based on the context below.

If the answer is not contained in the context, respond with:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
```

## Sample Interaction

`You: What is LangChain?
AI: LangChain is a framework for developing applications using language models, enabling ...`

**With retrieval evaluation**:

`--- Retrieved Sources (Top-K) ---
[1] Source: doc_1 | Score: 0.8721 | Snippet: Introduction to LangChain. LangChain is a framework for developing applications ...
[2] Source: doc_2 | Score: 0.9345 | Snippet: Overview of FAISS embeddings and vector stores...`

---

## Retrieval Evaluation

- Top-K retrieval allows inspection of relevant chunks.

- Similarity scores indicate relevance (lower = more relevant).

- If no retrieved chunk meets relevance, the system **safely refuses** to answer rather than hallucinate.

---

## Project Structure

`├── main.py           # CLI chatbot
├── faiss_index.py    # Document ingestion, FAISS, RAG, PromptTemplate
├── config.py         # Configuration parameters
├── knowledge_base/   # .txt files for local KB
├── requirements.txt
├── README.md
└── LICENSE`

---

## Configuration (`config.py`)

`EMBEDDING_MODEL = "models/gemini-embedding-001"
CHUNK_SIZE = 800            # Max characters per document chunk
CHUNK_OVERLAP = 200         # Overlap between chunks
TOP_K = 5                   # Number of chunks retrieved per query`

---

## 📦 Requirements

- Python **3.8+**

- Google **Gemini API Key**

---

## ⚙️ Setup & Run

### 1\. Install Dependencies

`pip install -r requirements.txt`

### 2\. Create `.env` file in the project root

`GOOGLE_API_KEY=YOUR_API_KEY`

### 3\. Run the Chatbot

`python main.py`

> Type `exit` to quit the chat.
