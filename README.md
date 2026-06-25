# Production RAG Assistant

This is a Streamlit-based RAG dashboard for uploading TXT/PDF documents, indexing them with ChromaDB, and asking grounded questions using Gemini.

Live demo: https://yothtq3mtgvquhd9ppafsa.streamlit.app/

## What It Does

The app lets you upload documents, rebuild a local vector database, ask questions against the indexed content, and inspect the retrieved sources behind each answer.

It is meant as a practical RAG demo and learning project, not a long-term document storage system.

## Features

- Upload TXT and PDF files from the browser
- Sanitize uploaded filenames before saving
- Index documents into ChromaDB
- Stop oversized ingestion before Gemini embeddings are called
- Preserve the current vector database when the document safety limit is exceeded
- Show friendly quota and rate-limit errors instead of raw tracebacks
- Reset Chroma client state so deleted or replaced documents are not retrieved again
- Ask questions using a shared RAG pipeline
- Retrieve five chunks by default, with an adjustable retrieval slider
- Show retrieved sources, previews, and distance scores
- Rebuild or clear the generated vector database
- Delete uploaded documents safely
- Keep session-only query history
- Run basic local health checks without spending Gemini quota

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- Google Gemini
- pypdf
- python-dotenv

## Local Setup

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a local `.env` file from the safe template:

```text
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

Keep the real `.env` private. It is ignored by Git and should not be committed. The files `.env.example` and `.streamlit/secrets.example.toml` are safe templates only.

## Environment Variables

`GOOGLE_API_KEY` is required for Gemini embeddings and answer generation.

For local runs, put it in `.env`. For hosted deployments, set it through the hosting platform's secret manager instead of committing it to the repo.

## Run Locally

Start the Streamlit app:

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

You can also use the terminal flow:

```powershell
python ingest.py
python ask.py
```

## Useful Checks

Run these before a demo or after changing setup-related files:

```powershell
python smoke_test.py
python startup_check.py
```

`smoke_test.py` checks core project health without calling Gemini. `startup_check.py` checks whether the local environment looks ready to run.

## Deployment Note

The deployed Streamlit app is demo-oriented. Set `GOOGLE_API_KEY` as a deployment secret, and do not upload private or sensitive documents to cloud API-based demos.

For more deployment details, see `DEPLOYMENT.md`. For demo flow notes, see `DEMO_GUIDE.md`.

## Known Limitation

Streamlit Community Cloud is friendly for demos, but uploaded files and the local ChromaDB folder may not be permanent across cloud restarts. For long-running use, move uploaded files and vector storage to persistent services.

To reduce free-tier embedding failures, ingestion stops when the uploaded documents create more than 100 chunks. This is a project safety limit configured in `rag/config.py`, not a guarantee of Gemini API quota availability.

## Project Files

- `app.py` - Streamlit dashboard
- `ingest.py` - builds the ChromaDB vector database
- `ask.py` - terminal question-answering flow
- `evaluate.py` - small RAG evaluation runner
- `smoke_test.py` - no-API project health check
- `startup_check.py` - local setup readiness check
- `rag/` - shared RAG pipeline, prompts, helpers, upload safety, history, and document utilities
