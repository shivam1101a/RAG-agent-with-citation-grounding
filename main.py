import os
import numpy as np
import faiss  # For building vector index and perform search
import fitz  # or pymupdf for pdf related work

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


# ---------------------------------------
# 1. SETUP
# ---------------------------------------

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DOCUMENT_FOLDER = "documents"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.6-luna"


# ---------------------------------------
# 2. DATA STRUCTURE
# ---------------------------------------


class Chunk(BaseModel):
    text: str
    source: str
    page: int


chunks = []  # List to store chunks


# ---------------------------------------
# 3. LOAD PDF DOCUMENTS
# ---------------------------------------


def load_documents():
    for filename in os.listdir(DOCUMENT_FOLDER):
        if not filename.endswith(".pdf"):
            continue

        path = os.path.join(DOCUMENT_FOLDER, filename)  # Constructs the full file path
        pdf = fitz.open(path)

        for page_number, page in enumerate(pdf):
            text = page.get_text()  # Text extraction
            if not text.strip():  # Continue for blank space
                continue

            words = text.split()  # Simple chunking using white space as a delimiter
            # Not the best case as meaning of sentence is lost
            chunk_size = 300
            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(  # Word list in size of 300 words block as a single string
                    words[i : i + chunk_size]
                )
                chunks.append(  # wraps each chunk into a chunk of 300 words
                    Chunk(text=chunk_text, source=filename, page=page_number + 1)
                )
        pdf.close()
    print(f"Loaded {len(chunks)} chunks.")


# ---------------------------------------
# 4. CREATE EMBEDDINGS
# ---------------------------------------


def create_embeddings():
    texts = [  # Creation of list of strings of chunk.text
        chunk.text for chunk in chunks
    ]
    response = client.embeddings.create(  # creation of embedding model in vector space
        model=EMBEDDING_MODEL, input=texts
    )
    embeddings = np.array([item.embedding for item in response.data], dtype="float32")
    # Normalize vectors
    faiss.normalize_L2(embeddings)
    return embeddings


# ---------------------------------------
# 5. CREATE VECTOR DATABASE
# ---------------------------------------


def create_vector_database(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


# ---------------------------------------
# 6. SEARCH DOCUMENTS
# ---------------------------------------


def retrieve_documents(question, index, top_k=5):
    response = client.embeddings.create(  # sends the ques to embedding api
        model=EMBEDDING_MODEL, input=question
    )
    query_embedding = np.array(  # Wrap the vector space into 2-D numpy array
        [response.data[0].embedding], dtype="float32"
    )
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding, top_k
    )  # search to see how this works in maths terms
    # of dot product and cosine function using faiss
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({"score": float(score), "chunk": chunks[idx]})
    return results


# ---------------------------------------
# 7. GENERATE RAG ANSWER
# ---------------------------------------


class RAGAnswer(BaseModel):
    answer: str
    citations: list[int]
    confidence: float
    needs_fallback: bool


def generate_answer(question, retrieved_documents):
    context = ""  # Empty string variable inialized
    for i, result in enumerate(retrieved_documents):
        chunk = result["chunk"]
        context += f"""
SOURCE [{i + 1}]
File: {chunk.source}
Page: {chunk.page}

{chunk.text}

--------------------
"""

    prompt = f"""
You are a citation-grounded question answering system.

Answer the user's question using ONLY the
provided sources.

If the sources don't contain enough information,
do not guess.

Every factual claim must be supported by
one or more source numbers.

Question:
{question}

Sources:
{context}

Return:

1. answer
2. citations
3. confidence between 0 and 1
4. needs_fallback

Set needs_fallback=true if the sources
are insufficient to answer the question.
"""

    response = client.responses.parse(
        model=CHAT_MODEL, input=prompt, text_format=RAGAnswer
    )

    return response.output_parsed


# ---------------------------------------
# 8. GROUNDING CHECKER
# ---------------------------------------


class GroundingResult(BaseModel):
    supported: bool
    confidence: float
    unsupported_claims: list[str]
    reason: str


def check_grounding(question, answer, retrieved_documents):
    evidence = ""
    for result in retrieved_documents:
        chunk = result["chunk"]
        evidence += f"""
{chunk.text}
"""

    prompt = f"""
You are a grounding verification system.

Determine whether the answer is fully
supported by the supplied evidence.

Question:
{question}

Answer:
{answer}

Evidence:
{evidence}

Check for:

- unsupported claims
- invented facts
- claims not present in evidence
- incorrect interpretation

Return a confidence score between
0 and 1.
"""

    response = client.responses.parse(
        model=CHAT_MODEL, input=prompt, text_format=GroundingResult
    )

    return response.output_parsed


# ---------------------------------------
# 9. MAIN RAG PIPELINE
# ---------------------------------------


def ask_question(question, index):
    print("\nSearching documents...")
    retrieved = retrieve_documents(question, index)
    print(f"Retrieved {len(retrieved)} chunks.")
    answer = generate_answer(question, retrieved)
    print(f"Initial confidence: {answer.confidence:.2f}")
    grounding = check_grounding(question, answer.answer, retrieved)
    print(f"Grounding confidence: {grounding.confidence:.2f}")

    # -----------------------------------
    # LOW CONFIDENCE
    # -----------------------------------

    if grounding.confidence < 0.80 or not grounding.supported or answer.needs_fallback:
        print("\nLow confidence.")
        print("Fallback search would be activated here.")
        return {
            "answer": answer.answer,
            "status": "LOW_CONFIDENCE",
            "confidence": grounding.confidence,
            "citations": answer.citations,
        }

    # -----------------------------------
    # HIGH CONFIDENCE
    # -----------------------------------

    return {
        "answer": answer.answer,
        "status": "GROUNDED",
        "confidence": grounding.confidence,
        "citations": answer.citations,
    }


# ---------------------------------------
# 10. APPLICATION
# ---------------------------------------


def main():

    print("Loading documents...")
    load_documents()
    print("Creating embeddings...")
    embeddings = create_embeddings()
    print("Creating vector database...")
    index = create_vector_database(embeddings)
    print("\nRAG Agent Ready!")
    while True:
        question = input("\nAsk a question (type 'exit' to quit): ")
        if question.lower() == "exit":
            break

        result = ask_question(question, index)
        print("\n==============================")
        print("ANSWER")
        print("==============================")

        print(result["answer"])

        print(f"\nConfidence: {result['confidence']:.2f}")

        print(f"Status: {result['status']}")

        print(f"Citations: {result['citations']}")


if __name__ == "__main__":
    main()
