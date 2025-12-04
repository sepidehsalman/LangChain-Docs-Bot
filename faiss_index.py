import os
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

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
    document_objects = [Document(page_content=doc) for doc in documents]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    faiss_index = FAISS.from_documents(document_objects, embeddings)
    return faiss_index


# Function to ask a question and get an answer
def ask_question(question: str, faiss_index):
    retriever = faiss_index.as_retriever()

    # Define your structured prompt (you can modify this to be more specific)
    prompt_structure = f"""
    You are a helpful assistant that answers questions based on documents.
    The documents contain a wide range of information. Use the context to provide a clear and informative answer.
    Here is the question: {question}
    Please respond in a concise and informative manner.
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
