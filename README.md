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
