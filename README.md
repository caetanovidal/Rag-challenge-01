# Setup Instructions

1. Environment Variables

```bash
# Create a .env file in the project root and add the following:

### LangSmith (optional – only needed if you want tracing enabled)
LANGSMITH_TRACING=true  

LANGSMITH_ENDPOINT=https://api.smith.langchain.com  
LANGSMITH_API_KEY="your_langsmith_api_key"

LANGSMITH_PROJECT=rag_challenge

### OpenAI
OPENAI_API_KEY="your_openai_api_key"
```

Note: LangSmith configuration is optional. You can remove or ignore these variables if you are not using LangSmith.

2. Python Version

```bash
# Python 3.11+ is required
# Python 3.13 is recommended
```

3. Create and Activate a Virtual Environment (Recommended)

```powershell
# Windows (PowerShell)
py -3.13 -m venv .venv
Set-ExecutionPolicy Unrestricted -Scope Process
cd .venv\Scripts
.\activate
cd ..
cd ..
```

You should now see (.venv) in your terminal.

4. Install Dependencies

With the virtual environment activated, install the required packages:

pip install -r requirements.txt

5. Generate the Chroma Vector Database (Required)

Before running the Streamlit app, you must generate the ChromaDB by executing the RAG preprocessing notebook.

Open and run all cells in:

rag.ipynb


This notebook is responsible for:

Loading and processing the source documents

Chunking and embedding the data

Persisting the embeddings into a local ChromaDB instance

The Streamlit application depends on this database to function correctly.

6. Run the Streamlit App

After the ChromaDB has been generated, start the application with:

streamlit run app.py


Make sure this command is executed inside the virtual environment.

7. Access the Application

Open your browser and go to:

http://localhost:8501/




## Inside performance folder you can find a excel file comparing my model against the ground truth question by question



# My intuition / logic behind the code

Initialize the vector store using indexing, embeddings, and adding metadata with the document name to allow routing to one document or another in the future.

Routing: Given the user question, filter which document is best to retrieve chunks from: Standard or Health Plus.

Query translation: Use multi-query to expand the user query into 4–5 queries, allowing retrieval of more chunks and more context.

Retrieving: Use the multi-query to retrieve around 50 chunks, then apply filtering and thresholding to get the best 2–3 chunks.

Generate: With the best 2–3 chunks, generate the answer and return it to the user.

Fallback: In case the response is "Not found in the document", rewrite the user question and try again using a BM25 search instead of semantic search.

Resume: I learned a lot about RAG in the last week. I tried to create a simple RAG with indexing, routing, query translation, retrieval, and generation.

