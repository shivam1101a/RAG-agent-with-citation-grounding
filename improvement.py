from pydantic import BaseModel


class GroundingResult(BaseModel):
    confidence: float
    supported: bool
    unsupported_claims: list[str]
    missing_information: list[str]
    improved_query: str
    reason: str


# define grounding


def check_grounding(question, answer, retrieved_documents):

    evidence = ""

    for i, result in enumerate(retrieved_documents):
        chunk = result["chunk"]

        evidence += f"""
SOURCE {i + 1}

File: {chunk.source}
Page: {chunk.page}

{chunk.text}

-------------------------
"""

    prompt = f"""
You are a strict RAG grounding evaluator.

QUESTION:
{question}

GENERATED ANSWER:
{answer}

EVIDENCE:
{evidence}

Your job is to determine whether the answer
is completely supported by the evidence.

Do NOT increase the confidence merely because
the answer sounds plausible.

Check:

1. Does every important claim have evidence?
2. Is the evidence directly relevant?
3. Is the answer making assumptions?
4. Is important information missing?
5. Are there contradictory sources?

Give:

confidence:
0 = completely unsupported
1 = completely supported

If confidence is below 0.80:

- identify unsupported claims
- identify missing information
- create an improved search query

The improved query should help the retriever
find better evidence in the same knowledge base.
"""

    response = client.responses.parse(
        model=CHAT_MODEL, input=prompt, text_format=GroundingResult
    )

    return response.output_parsed


# check grounding


def iterative_rag(question, index, max_iterations=3):

    current_query = question

    best_answer = None
    best_grounding = None

    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}/{max_iterations}")

        # ----------------------------
        # RETRIEVE
        # ----------------------------

        retrieved = retrieve_documents(current_query, index, top_k=5 + iteration * 2)

        print(f"Retrieved {len(retrieved)} chunks")

        # ----------------------------
        # GENERATE
        # ----------------------------

        answer = generate_rag_answer(question, retrieved)

        print(f"Initial confidence: {answer.confidence:.2f}")

        # ----------------------------
        # GROUNDING
        # ----------------------------

        grounding = check_grounding(question, answer.answer, retrieved)

        print(f"Grounding confidence: {grounding.confidence:.2f}")

        # ----------------------------
        # SAVE BEST RESULT
        # ----------------------------

        if best_grounding is None or grounding.confidence > best_grounding.confidence:
            best_answer = answer

            best_grounding = grounding

        # ----------------------------
        # SUCCESS
        # ----------------------------

        if grounding.confidence >= 0.80 and grounding.supported:
            print("Sufficient evidence found.")

            return {
                "answer": answer.answer,
                "confidence": grounding.confidence,
                "status": "GROUNDED",
                "iteration": iteration + 1,
                "citations": answer.citations,
            }

        # ----------------------------
        # IMPROVE QUERY
        # ----------------------------

        print("Evidence insufficient.")

        print("Improved query:")

        print(grounding.improved_query)

        current_query = grounding.improved_query

    # --------------------------------
    # MAX ITERATIONS
    # --------------------------------

    print("\n Maximum RAG iterations reached.")

    return {
        "answer": best_answer.answer,
        "confidence": best_grounding.confidence,
        "status": "LOW_CONFIDENCE",
        "iteration": max_iterations,
        "citations": best_answer.citations,
        "needs_web_search": True,
    }


# improvement loop

""" best_answer
best_grounding  . so that confidence does not go backwards"""
