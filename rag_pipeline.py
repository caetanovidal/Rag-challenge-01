from rag import route_query, select_top_docs, format_context, LLM, rewrite_query, select_top_docs_bm25



# rag_pipeline.py
def answer_question(query: str, max_retries: int = 2) -> dict:
    """
    Main RAG entry point for the UI
    """

    current_query = query

    for attempt in range(max_retries + 1):
        # ----------------------------
        # ROUTING + RETRIEVAL
        # ----------------------------
        route = route_query(current_query)
        
        # Logic: Use BM25 if it's the 3rd attempt (attempt == 2)
        if attempt == 1:
            selected_docs = select_top_docs_bm25(current_query, route, k=3)
            print(selected_docs)
        else:
            selected_docs = select_top_docs(current_query, route, k=3)
            
        context = format_context(selected_docs)        

        prompt = f"""
You are answering a question using ONLY the provided document excerpts.

Rules:
- Prefer passages that explicitly define the term being asked about
- Ignore boilerplate or repeated legal language
- Cite the page number used in your answer at the end in this format:
  [Northwind_Standard_Benefits_Details.pdf#page=24]
- If the answer is not explicitly stated, say "Not found in the document"

Context:
{context}

Question:
{current_query}
"""

        llm_response = LLM.invoke(prompt)
        answer = llm_response.content.strip()

        # ----------------------------
        # SUCCESS PATH
        # ----------------------------
        if "not found in the document" not in answer.lower():
            return {
                "question": current_query,
                "answer": answer,
                "route": route,
                "attempts": attempt + 1,
                "retried": attempt > 0,
                "sources": [
                    f"{d.metadata['doc_name']}#page={d.metadata.get('page_label')}"
                    for d in selected_docs
                ],
            }

        # ----------------------------
        # RETRY PATH
        # ----------------------------
        if attempt < max_retries:
            # If semantic search failed twice, the next attempt (attempt 2) will use BM25
            pass
            current_query = rewrite_query(query)

    # ----------------------------
    # FINAL FALLBACK (still not found)
    # ----------------------------
    return {
        "question": current_query,
        "answer": answer,
        "route": route,
        "attempts": max_retries + 1,
        "retried": True,
        "sources": [
            f"{d.metadata['doc_name']}#page={d.metadata.get('page_label')}"
            for d in selected_docs
        ],
    }

