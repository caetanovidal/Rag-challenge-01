Setup Instructions
1. Environment Variables

Create a .env file in the project root and add the following:

# LangSmith (optional – only needed if you want tracing enabled)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY="your_langsmith_api_key"
LANGSMITH_PROJECT=rag_challenge

# OpenAI
OPENAI_API_KEY="your_openai_api_key"


Note: LangSmith configuration is optional. You can remove or ignore these variables if you are not using LangSmith.

2. Python Version

Python 3.11+ is required

Python 3.13 is recommended

3. Create and Activate a Virtual Environment (Recommended)

Creating a virtual environment is highly recommended to avoid dependency conflicts.

Windows (PowerShell)
py -3.13 -m venv .venv
Set-ExecutionPolicy Unrestricted -Scope Process
cd .venv\Scripts
.\activate
cd ..
cd ..


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