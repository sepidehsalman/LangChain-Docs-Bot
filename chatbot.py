import os
from faiss_index import (
    load_documents_from_directory,
    ask_question,
    load_documents_to_faiss,
)
from config import TOP_K


# Function to handle the chatbot interaction
def start_chat():
    print("Welcome to the Chatbot! Type 'exit' to end the chat.\n")

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KB_DIR = os.path.join(BASE_DIR, "knowledge_base")

    # Load and index documents
    documents = load_documents_from_directory(KB_DIR)
    faiss_index = load_documents_to_faiss(documents)

    while True:
        # Get user input
        question = input("You: ").strip()

        # Exit condition
        if question.lower() == "exit":
            print("Goodbye!")
            break

        # Validate empty input
        if not question:
            print("AI: Please enter a valid question.\n")
            continue

        # Ask question
        answer = ask_question(question, faiss_index)
        result = answer.get("result", "").strip()

        # Print result
        print(f"AI: {result}\n")

        # Print top-K retrieval sources (for reviewer transparency)
        print("--- Retrieved Sources (Top-K) ---")
        docs_with_scores = faiss_index.similarity_search_with_score(question, k=TOP_K)
        if docs_with_scores:
            for i, (doc, score) in enumerate(docs_with_scores, 1):
                source = doc.metadata.get("source", "Unknown")
                snippet = doc.page_content[:150].replace("\n", " ")
                print(
                    f"[{i}] Source: {source} | Score: {score:.4f} | Snippet: {snippet}...\n"
                )
        else:
            print("No relevant documents found.\n")
