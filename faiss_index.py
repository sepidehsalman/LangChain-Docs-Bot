import os
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()


def load_documents_from_directory(directory: str):
    """
    Load all .txt files from a directory into a list of strings.
    """
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs


def load_documents_to_faiss(documents: list):
    """
    Convert raw documents into FAISS index with chunking and embeddings.
    """
    document_objects = [
        Document(page_content=doc, metadata={"source": f"doc_{i}"})
        for i, doc in enumerate(documents, 1)
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    split_docs = text_splitter.split_documents(document_objects)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    faiss_index = FAISS.from_documents(split_docs, embeddings)

    return faiss_index


def ask_question(question: str, faiss_index):
    """
    Ask a question to the FAISS-based RAG system using a custom prompt template.
    Logs retrieval info and handles empty retrievals safely.
    """
    retriever = faiss_index.as_retriever(search_kwargs={"k": TOP_K})

    # Retrieve top-K chunks with scores for evaluation
    docs_with_scores = faiss_index.similarity_search_with_score(question, k=TOP_K)

    if not docs_with_scores:
        return {"result": "I could not find the answer in the provided documents."}

    # Basic retrieval evaluation: print top chunks and scores
    print("\n--- Retrieved Chunks ---")
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        print(f"[{i}] Score: {score:.4f}")
        print(doc.page_content[:200].replace("\n", " "), "...\n")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_tokens=300,
        timeout=None,
        max_retries=2,
    )

    # Define a custom PromptTemplate for RAG
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant that answers questions strictly based on the context below.

If the answer is not contained in the context, respond with:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
""",
    )

    # Build RetrievalQA chain using the prompt template
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )

    # Invoke LLM with the question
    answer = qa_chain.invoke({"query": question})
    result = answer.get("result", "").strip()

    if not result:
        return {"result": "I could not find the answer in the provided documents."}

    return {"result": result}
