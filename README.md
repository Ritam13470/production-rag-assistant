# Production RAG Assistant

A beginner-friendly Retrieval-Augmented Generation app built with Python, LangChain, ChromaDB, and Google Gemini.

The app reads TXT and PDF files from the data folder, converts them into searchable chunks, stores them in a local Chroma vector database, and answers questions using only the retrieved document context.

## Features

- Read TXT files
- Read PDF files
- Split documents into chunks
- Create embeddings using Gemini
- Store vectors locally with ChromaDB
- Ask questions from the terminal
- Show source files for retrieved context
- Refuse to answer when information is not found in the documents

## Tech Stack

- Python
- LangChain
- ChromaDB
- Google Gemini API
- pypdf
- python-dotenv

## Setup

Create and activate a virtual environment:

py -m venv .venv
.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Create a .env file:

GOOGLE_API_KEY=your_google_gemini_api_key_here

## Usage

Add TXT or PDF files inside the data folder.

Create the vector database:

python ingest.py

Ask questions:

python ask.py

Type exit to quit.

## Notes

The .env, .venv, and chroma_db folders are ignored by Git so secrets and generated files are not uploaded to GitHub.

## Web UI

This project also includes a simple Streamlit web interface.

Run the web app:

streamlit run app.py

Then open the local URL shown in the terminal, usually:

http://localhost:8501

In the browser, type a question and click Ask.

## Browser Upload Workflow

The Streamlit app supports uploading documents directly from the browser.

Steps:

1. Run the web app:

streamlit run app.py

2. Upload one or more TXT or PDF files.

3. Click Save Uploaded Files.

4. Click Rebuild Vector Database.

5. Ask questions from the uploaded documents.

Important: For learning, use sample documents only. Avoid uploading private or sensitive documents to cloud API-based apps.

## Better RAG Quality Tools

The app includes retrieval debugging tools to make answers more transparent.

Features:

- Choose how many chunks to retrieve using the sidebar slider.
- View source previews used to generate the answer.
- See Chroma distance scores for retrieved chunks.
- Open the Retrieval Debug Panel to inspect the exact context sent to Gemini.

These tools help debug common RAG problems such as weak retrieval, irrelevant chunks, missing context, and hallucinated answers.

## Project Architecture

The project now separates reusable RAG logic into a small rag package.

rag/
|-- __init__.py
|-- config.py       # shared paths, collection name, and model names
|-- embeddings.py   # safe Gemini embedding wrapper
|-- utils.py        # response parsing, source formatting, and text previews

Main app files:

ingest.py   # loads TXT/PDF files and rebuilds ChromaDB
ask.py      # terminal-based RAG assistant
app.py      # Streamlit web UI with upload, re-index, sources, and debug panel

This structure keeps the project easier to maintain and prepares it for future advanced RAG features.

## RAG Evaluation

The project includes a basic evaluation script for testing answer quality.

Evaluation files:

eval_questions.json   # test questions, expected keywords, and refusal checks
evaluate.py           # runs the test cases against the RAG pipeline

Run evaluation:

python evaluate.py

The evaluator checks whether answers contain expected keywords and whether the assistant correctly refuses to answer when the document does not contain the answer.

Example output:

Passed 3 out of 3 tests.
Evaluation status: SUCCESS

## Shared RAG Pipeline

The project uses a centralized RAG pipeline so the terminal app, Streamlit app, and evaluator all use the same retrieval and answer-generation logic.

Pipeline files:

rag/prompts.py     # stores the main RAG prompt
rag/pipeline.py    # builds the vector store, LLM, prompt, and answer flow

Main shared function:

answer_question(question, top_k=3)

This function returns a RagResult object containing:

- question
- answer
- retrieved documents
- scored retrieved documents
- final context sent to the model

This makes the project easier to maintain because future retrieval upgrades only need to be added in one place.

## Production Error Handling

The project includes centralized RAG error handling in rag/errors.py.

It converts raw technical errors into cleaner user-facing messages.

Handled cases include:

- Gemini quota or rate-limit errors
- Missing, invalid, or unauthorized API key errors
- Vector database loading errors
- Unexpected RAG pipeline failures

The Streamlit app shows a clean error message and hides technical details inside an expandable panel.

The terminal app prints a readable RAG error message instead of a long traceback.

The evaluator reports API/quota errors as blocked evaluations instead of crashing.
