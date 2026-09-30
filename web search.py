import os
import numpy as np
import faiss
import fitz

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


# ==========================================
# SETUP
# ==========================================

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DOCUMENT_FOLDER = "documents"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.6-luna"


# ==========================================
# DATA MODEL
# ==========================================


class Chunk(BaseModel):
    text: str
    source: str
    page: int


chunks = []


# ==========================================
# LOAD PDF
# ==========================================


def load_documents():

    for filename in os.listdir(DOCUMENT_FOLDER):
        if not filename.endswith(".pdf"):
            continue

        path = os.path.join(DOCUMENT_FOLDER, filename)

        pdf = fitz.open(path)

        for page_number, page in enumerate(pdf):
            text = page.get_text()

            words = text.split()

            chunk_size = 300

            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[i : i + chunk_size])

                chunks.append(
                    Chunk(text=chunk_text, source=filename, page=page_number + 1)
                )

        pdf.close()

    print(f"Loaded {len(chunks)} chunks.")


# ==========================================
# CREATE EMBEDDINGS
# ==========================================


def create_embeddings():

    texts = [chunk.text for chunk in chunks]

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)

    embeddings = np.array([item.embedding for item in response.data], dtype="float32")

    faiss.normalize_L2(embeddings)

    return embeddings


# ==========================================
# VECTOR DATABASE
# ==========================================


def create_vector_database(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ==========================================
# RETRIEVAL
# ==========================================


def retrieve_documents(question, index, top_k=5):

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=question)

    query_embedding = np.array([response.data[0].embedding], dtype="float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({"score": float(score), "chunk": chunks[idx]})

    return results


# ==========================================
# RAG ANSWER
# ==========================================


class RAGAnswer(BaseModel):
    answer: str

    citations: list[int]

    confidence: float

    needs_fallback: bool


def generate_rag_answer(question, retrieved):

    context = ""

    for i, result in enumerate(retrieved):
        chunk = result["chunk"]

        context += f"""
SOURCE [{i + 1}]
File: {chunk.source}
Page: {chunk.page}

{chunk.text}

-----------------------
"""

    prompt = f"""
You are a citation-grounded RAG assistant.

Answer the question using ONLY the
provided documents.

If the documents do not contain
sufficient information, do not guess.

Question:
{question}

Documents:
{context}

Return:
- answer
- citations
- confidence from 0 to 1
- needs_fallback
"""

    response = client.responses.parse(
        model=CHAT_MODEL, input=prompt, text_format=RAGAnswer
    )

    return response.output_parsed


# ==========================================
# WEB SEARCH TOOL
# ==========================================


def web_search(question):

    print("\nRAG confidence is low.")
    print("Activating web search...\n")

    response = client.responses.create(
        model=CHAT_MODEL,
        tools=[{"type": "web_search"}],
        input=f"""
Search the web for reliable and current
information about the following question:

{question}

Use trustworthy sources.

Provide a concise answer and cite
the sources you used.
""",
    )

    return response


# ==========================================
# MAIN AGENT
# ==========================================


def ask_question(question, index):

    print("\nSearching knowledge base...")

    retrieved = retrieve_documents(question, index)

    print(f"Retrieved {len(retrieved)} chunks.")

    # -----------------------------
    # RAG
    # -----------------------------

    rag_answer = generate_rag_answer(question, retrieved)

    print(f"RAG confidence: {rag_answer.confidence:.2f}")

    # -----------------------------
    # DECISION
    # -----------------------------

    if rag_answer.confidence >= 0.80 and not rag_answer.needs_fallback:
        print("\nAnswer supported by knowledge base.")

        return rag_answer.answer

    # -----------------------------
    # FALLBACK
    # -----------------------------

    print("\nInsufficient evidence in knowledge base.")

    web_response = web_search(question)

    return web_response.output_text


# ==========================================
# APPLICATION
# ==========================================


def main():

    print("Loading documents...")

    load_documents()

    print("Creating embeddings...")

    embeddings = create_embeddings()

    print("Creating vector database...")

    index = create_vector_database(embeddings)

    print("\n================================")
    print("   CITATION GUARD RAG AGENT")
    print("================================")

    while True:
        question = input("\nAsk a question (exit to quit): ")

        if question.lower() == "exit":
            break

        answer = ask_question(question, index)

        print("\n-------------------------------")
        print("FINAL ANSWER")
        print("-------------------------------")

        print(answer)


if __name__ == "__main__":
    main()
