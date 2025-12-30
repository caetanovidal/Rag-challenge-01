#!/usr/bin/env python
# coding: utf-8

# In[104]:


from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from collections import Counter
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from rank_bm25 import BM25Okapi
import numpy as np
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import re



load_dotenv()


# ## Transfer notebook to python

# In[105]:


#get_ipython().system('jupyter nbconvert --to script rag.ipynb')


# ### Indexing

# In[106]:


docs = []
pdfs = {
    "standard": "northwind_docs/Northwind_Standard_Benefits_Details.pdf",
    "health_plus": "northwind_docs/Northwind_Health_Plus_Benefits_Details.pdf",
}

# Note: Ensure the 'northwind_docs' directory and files exist in your environment
for doc_name, path in pdfs.items():
    if os.path.exists(path):
        loader = PyPDFLoader(path)
        pages = loader.load()
        for page in pages:
            page.metadata["doc_name"] = doc_name
            # Ensure page_label is set for consistent retrieval
            if "page" in page.metadata and "page_label" not in page.metadata:
                page.metadata["page_label"] = str(page.metadata["page"] + 1)
        docs.extend(pages)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)


print(Counter(doc.metadata["doc_name"] for doc in chunks))


# ## Store

# In[107]:


ids = [
    f"{doc.metadata['doc_name']}_p{doc.metadata.get('page_label')}_c{i}"
    for i, doc in enumerate(chunks)
]

vectorstore = Chroma.from_documents(
    chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    ids=ids,
    persist_directory="./chroma"
)


# ### Manual test to retrieve docs

# results = vectorstore.similarity_search(
#     "what is in-network mean for Northwind Health?",
#     k=5,
#     filter={"doc_name": "standard"} # function to define filter later
# )
# 
# for r in results:
#     print(r.metadata["doc_name"], "| page:", r.metadata.get("page_label"))
# 
# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 4,
#         "fetch_k": 20,
#         "filter": {"doc_name": "standard"}
#     }
# )
# 
# docs = retriever.invoke(
#     "what is in-network mean for Northwind Health?"
# )
# 
# query = "what is in-network mean for Northwind Health?"
# docs_scores = vectorstore.similarity_search_with_score(
#     query,
#     k=10,
#     filter={"doc_name": "standard"}
# )
# 
# for d, s in docs_scores:
#     print(f"page={d.metadata.get('page_label')} score={s:.3f}")
# 
# 
# 

# ### Routing Prompt 

# In[108]:


ROUTING_PROMPT = """You are a routing agent for a benefits Q&A system.

You must choose exactly ONE document to answer the user question.

Available documents:
- Northwind_Standard_Benefits_Details.pdf
- Northwind_Health_Plus_Benefits_Details.pdf

Routing rules:
- If the question explicitly mentions “Standard”, “Standard plan”, or refers to benefits, coverage, or rules described in the Standard plan, route to:
  → Northwind_Standard_Benefits_Details.pdf

- If the question explicitly mentions “Health Plus”, “Health Plus plan”, or refers to copays, deductibles, coinsurance, balance billing protection, or Health Plus–specific features, route to:
  → Northwind_Health_Plus_Benefits_Details.pdf

- If the question mentions “Northwind Health” without specifying a plan:
  - Most likely to be the standar
  - Infer the correct plan based on the benefit details being asked.
  - Route to the single most relevant document.
  - Do NOT route to both documents.

You must output exactly one of:
- "standard"
- "health_plus"

Do not explain your reasoning.
Do not output anything else.
"""

LLM = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


def route_query(query: str) -> str:
    """
    Determines which document to search based on the user query.
    """
    messages = [
        SystemMessage(content=ROUTING_PROMPT),
        HumanMessage(content=f'User question: "{query}"')
    ]
    resp = LLM.invoke(messages)
    route = resp.content.strip().lower()

    # Validation to ensure we only return valid doc_names
    if "standard" in route:
        return "standard"
    elif "health_plus" in route:
        return "health_plus"
    else:
        # Default fallback if the LLM output is unexpected
        return "standard"


# ### Generate

# In[109]:


SYSTEM_PROMPT = """You are an expert at rewriting user questions to optimize document retrieval
from insurance and benefits policy documents.

Your task is to generate multiple search queries that express the same intent
as the user question, but using terminology and phrasing commonly found in
formal insurance policy documents.

Rules:
- Do NOT answer the question
- Do NOT introduce new topics
- Preserve the original intent
- Use policy-style language (e.g., coverage, member responsibility, exclusions)
- Generate 3 to 5 concise queries
- Each query must be standalone and suitable for semantic search
- NO need for quotes"""



# using multi-query for better results
def expand_query(query: str) -> list[str]:
    # 1. Generate expanded queries using the LLM
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f'User question:\n"{query}"\n\nGenerate retrieval-optimized search queries.')
    ]
    resp = LLM.invoke(messages)
    lines = resp.content.split("\n")
    queries = [
        line.lstrip("0123456789. ").strip()
        for line in lines
        if line.strip()
    ]

    # 2. CRITICAL: Prepend the original query to the list
    # This ensures a high-similarity match for literal phrases like titles.
    queries.insert(0, query) 

    print(queries)

    return queries


# In[110]:


def select_top_docs_bm25(query: str, doc_filter: str, k: int = 3):
    """
    Fallback retrieval using BM25 keyword matching on filtered documents
    """
    # 1. Fetch all documents for the specific doc_name using vectorstore
    # We use a large k to get all relevant chunks for the specific document
    # then we will re-rank them with BM25.
    results = vectorstore.similarity_search(
        query,
        k=100, # Fetch a large enough pool to re-rank
        filter={"doc_name": doc_filter}
    )

    if not results:
        return []

    # 2. Tokenize documents and query
    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    tokenized_corpus = [tokenize(doc.page_content) for doc in results]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = tokenize(query)

    # 3. Get top k documents from the filtered pool
    top_docs = bm25.get_top_n(tokenized_query, results, n=k)
    return top_docs


# In[111]:


def retrieve_multi_query(query, doc_filter, k_per_query=10):
    expanded = expand_query(query)

    # Use a dictionary to store the best score for each unique document
    # Key: (doc_name, page_label), Value: (document, best_score)
    best_docs = {} 

    for q in expanded:
        # Use similarity_search_with_score to get the score directly
        docs_with_scores = vectorstore.similarity_search_with_score(
            q,
            k=k_per_query,
            filter={"doc_name": doc_filter}
        )

        for d, score in docs_with_scores:
            # Use page_label for key as it's the user-facing page number
            key = (d.metadata["doc_name"], d.metadata["page_label"]) 

            # Chroma returns L2 distance (lower is better).
            # We only store the document if it's a new entry or has a better (lower) score.
            if key not in best_docs or score < best_docs[key][1]:
                best_docs[key] = (d, score)

    # Return a list of (document, score) tuples
    return list(best_docs.values())




def select_top_docs(query, doc_filter, k=3):
    docs_scores = retrieve_multi_query(query, doc_filter)

    # Sort by vector distance (lower = better)
    docs_scores.sort(key=lambda x: x[1])

    return [d for d, _ in docs_scores[:k]]


def format_context(docs):
    blocks = []
    for d in docs:
        page = d.metadata.get("page_label", d.metadata.get("page"))
        source = d.metadata.get("source", "")
        filename = source.split("/")[-1]

        blocks.append(
            f"[{filename}#page={page}]\n{d.page_content.strip()}"
        )
    return "\n\n".join(blocks)


# ### Retry if answer Not found in the document

# In[112]:


REPHRASE_PROMPT = """You are an expert at rewriting to improve RAG document retrieval.

    The previous attempt failed because the answer was not found in the document.

    Rewrite the question to:
    - use simpler vocabulary
    - Changes the phrase but keep the meaning
    - paraphrasing
    - Be more specific
    - Preserve original intent
    - Do NOT answer the question

    Return ONE rewritten question only.
    """


# In[113]:


def rewrite_query(original_query: str) -> str:
    messages = [
        SystemMessage(content=REPHRASE_PROMPT),
        HumanMessage(content=f"Original question:\n{original_query}")
    ]
    resp = LLM.invoke(messages)
    return resp.content.strip()


# In[114]:


def re_generate(query: str, max_retries: int = 2) -> dict:
    """
    Runs the RAG pipeline.
    If the answer is 'Not found in the document',
    rewrites the query and retries once.
    """

    current_query = query

    for attempt in range(max_retries + 1):
        # ----------------------------
        # ROUTE + RETRIEVE
        # ----------------------------
        route = route_query(current_query)


        if attempt > 0:
            selected_docs = select_top_docs_bm25(
                query=current_query,
                doc_filter=route,
                k=3
            )
        else:
            selected_docs = select_top_docs(
                query=current_query,
                doc_filter=route,
                k=3
            )

        context = format_context(selected_docs)

        # ----------------------------
        # GENERATE
        # ----------------------------
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
        response = LLM.invoke(prompt)
        answer = response.content.strip()

        # ----------------------------
        # SUCCESS CASE
        # ----------------------------
        if "not found in the document" not in answer.lower():
            return {
                "question": current_query,
                "answer": answer,
                "route": route,
                "retried": attempt > 0,
                "attempts": attempt + 1
            }

        # ----------------------------
        # RETRY
        # ----------------------------
        if attempt < max_retries:
            current_query = rewrite_query(query)

    # ----------------------------
    # FINAL FALLBACK
    # ----------------------------
    return {
        "question": current_query,
        "answer": answer,
        "route": route,
        "retried": True,
        "attempts": max_retries + 1
    }


# ### Evaluate implemetation based on the ground truth dataset and save to excel

# In[115]:


GROUND_TRUTH_PATH = Path("performance/ground_truth.jsonl")
OUTPUT_PATH = Path("performance/rag_evaluation.xlsx")


# In[116]:


def load_ground_truth(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


# In[ ]:


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def compute_semantic_accuracy(
    prediction: str,
    truth: str,
    threshold: float = 0.80
) -> tuple[bool, float]:
    """
    Returns:
      - is_correct (bool)
      - similarity_score (float)
    """

    vectors = embeddings.embed_documents([prediction, truth])
    pred_vec, truth_vec = vectors

    score = cosine_similarity(
        np.array(pred_vec).reshape(1, -1),
        np.array(truth_vec).reshape(1, -1)
    )[0][0]

    return score >= threshold, score


# In[ ]:


def compute_groundedness(prediction: str, truth: str) -> bool:
    """
    Simplified groundedness check.
    Returns True if both citations contain 'standard' or both contain 'health_plus'.
    """
    p = prediction.lower()
    t = truth.lower()

    # Check for 'standard' in both
    if "standard" in p and "standard" in t:
        return True

    # Check for 'health_plus' in both (handling potential typos or variations)
    if ("health" in p and "plus" in p) and ("health" in t and "plus" in t):
        return True

    # Fallback for exact matches or other documents
    # Extract the filename part to be safe
    def extract_name(text):
        match = re.search(r"\[\s*([^\]#,\s]+\.pdf)", text, re.IGNORECASE)
        return match.group(1).lower() if match else None

    p_name = extract_name(prediction)
    t_name = extract_name(truth)

    return p_name == t_name if p_name and t_name else False


# In[119]:


def run_evaluation2():
    ground_truth_data = load_ground_truth(GROUND_TRUTH_PATH)

    rows = []

    for idx, item in enumerate(ground_truth_data, start=1):
        query = item["question"]
        truth = item["truth"]

        # ----------------------------
        # RAG PIPELINE (WITH RETRY)
        # ----------------------------
        result = re_generate(query, max_retries=2)

        answer = result["answer"]
        route = result["route"]
        sources = result.get("sources", [])
        retried = result.get("retried", False)
        attempts = result.get("attempts", 1)

        # ----------------------------
        # METRICS
        # ----------------------------
        accuracy, similarity_score = compute_semantic_accuracy(answer, truth)
        groundedness = compute_groundedness(answer, truth)

        rows.append({
            "question": query,
            "route": route,
            "llm_answer": answer,
            "ground_truth": truth,
            "semantic_similarity": round(similarity_score, 3),
            "accuracy": accuracy,
            "groundedness": groundedness,
            "used_retry": retried,
            "attempts": attempts,
            "sources": "; ".join(sources)
        })

        print(f"[{idx}/{len(ground_truth_data)}] ✔ processed")

    # ----------------------------
    # SAVE RESULTS
    # ----------------------------
    df = pd.DataFrame(rows)

    summary = pd.DataFrame([{
        "total_questions": len(df),
        "accuracy_%": round(df["accuracy"].mean() * 100, 2),
        "groundedness_%": round(df["groundedness"].mean() * 100, 2),
        "retry_rate_%": round(df["used_retry"].mean() * 100, 2)
    }])

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="results", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    print(f"\n✅ Evaluation saved to: {OUTPUT_PATH}")


# In[120]:


#run_evaluation2()


# In[ ]:




