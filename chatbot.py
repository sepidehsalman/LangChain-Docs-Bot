import os
from faiss_index import (
    load_documents_from_directory,
    ask_question,
    load_documents_to_faiss,
)


# Function to handle the chatbot interaction
def start_chat():
    print("Welcome to the Chatbot! Type 'exit' to end the chat.\n")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
    # Documents
    documents = load_documents_from_directory(KB_DIR)

    # Load documents into FAISS
    faiss_index = load_documents_to_faiss(documents)

    while True:
        # Get user input (question)
        question = input("You: ").strip()

        if not question:
            print("AI: Please ask a valid question.\n")
            continue
        # Exit condition
        if question.lower() == "exit":
            print("Goodbye!")
            break

        # Get the answer from the model
        answer = ask_question(question, faiss_index)

        result = answer.get("result", "").strip()
        # If empty → tell user it was not found
        if not result:
            print("AI: It did not find anything.\n")
        else:
            print(f"AI: {result}\n")
