import os
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

from dotenv import load_dotenv

load_dotenv()


def load_documents_from_directory(directory: str):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())  # <-- RETURN STRING
    return docs


def load_documents_to_faiss(documents: list):
    # Convert raw strings into Document objects
    document_objects = [Document(page_content=doc) for doc in documents]

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    split_docs = text_splitter.split_documents(document_objects)

    # Create embeddings and FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    faiss_index = FAISS.from_documents(split_docs, embeddings)

    return faiss_index


# Function to ask a question and get an answer
def ask_question(question: str, faiss_index):
    retriever = faiss_index.as_retriever(search_kwargs={"k": TOP_K})

    # Define your structured prompt (you can modify this to be more specific)
    prompt_structure = f"""
   You are a helpful assistant that answers questions strictly based on the retrieved documents.

   If the documents do NOT contain enough information to answer the question,
   clearly respond with:
   "I could not find the answer in the provided documents."
   Question: {question}
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_tokens=200,
        timeout=None,
        max_retries=2,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    return qa_chain.invoke(prompt_structure)
