# Production RAG Assistant Demo Guide

This guide explains how to demonstrate the Production RAG Assistant project safely.

The project supports two demo paths:

1. No-API demo path
2. Full Gemini RAG demo path

---

## 1. Project Location

Run:

cd E:\Projects\production-rag-assistant
.\.venv\Scripts\Activate.ps1

---

## 2. No-API Demo Path

Use this path when Gemini quota is exhausted or API access is unavailable.

Run:

git status
git branch -a
python smoke_test.py
python startup_check.py

Expected results:

- Git should show main branch
- Working tree should be clean
- smoke_test.py should pass
- startup_check.py should pass
- No Gemini API calls should be made

This proves the project structure, helper modules, setup files, and dependencies are healthy without spending Gemini quota.

---

## 3. Full Gemini RAG Demo Path

Use this path when Gemini API quota is available.

Run:

python startup_check.py
python smoke_test.py
python ingest.py
python ask.py

Example question:

What backend technologies does CharacterForge AI use?

Expected answer:

CharacterForge AI uses Node.js, Express, MongoDB, and Cloudinary for its backend.

Then run the Streamlit dashboard:

streamlit run app.py

Open the local URL shown in the terminal, usually:

http://localhost:8501

---

## 4. What to Show in Streamlit

Show these parts of the app:

1. Project overview sidebar
2. Document Dashboard
3. Current uploaded files
4. Upload Documents section
5. Delete Uploaded Document section
6. Manage Vector Database section
7. Ask Questions section
8. Sources and retrieval debug panel
9. Query History section

---

## 5. Safe Document Handling

The app supports:

- TXT/PDF upload
- Filename sanitization
- Duplicate filename protection
- Individual document deletion
- ChromaDB rebuild
- ChromaDB clear without deleting uploaded files

Important:

data/       = uploaded documents
chroma_db/  = generated vector database

Clearing the vector database removes only generated ChromaDB files.

---

## 6. Evaluation Demo

Run this only when Gemini API quota is available:

python evaluate.py

The evaluator uses:

eval_questions.json

It checks expected keywords and refusal behavior for missing document context.

---

## 7. Troubleshooting

### Gemini quota exhausted

Use the no-API demo path:

python smoke_test.py
python startup_check.py

### Vector database not found

Run:

python ingest.py

Or use the Streamlit button:

Rebuild Vector Database

### Missing API key

Make sure .env exists and contains:

GOOGLE_API_KEY=your_real_key_here

Do not commit the real .env file.

### Package import error

Run:

pip install -r requirements.txt

### Streamlit does not open

Run:

streamlit run app.py

Then open the Local URL shown in the terminal.

---

## 8. Recommended Demo Order

Safe no-API demo:

git status
git branch -a
python smoke_test.py
python startup_check.py
streamlit run app.py

Full Gemini demo:

python ingest.py
python ask.py
python evaluate.py
streamlit run app.py

---

## 9. Summary

This project demonstrates:

- Streamlit dashboard
- Gemini integration
- LangChain RAG pipeline
- ChromaDB vector database
- Safe document upload
- Individual document deletion
- Vector database management
- Source previews
- Retrieval debug panel
- Query history
- Evaluation script
- No-API smoke test
- Startup checklist

---

## 10. Final Demo Confidence Checklist

Before presenting, run:

git status
python smoke_test.py
python startup_check.py

If all three are clean or successful, the project is ready to demonstrate.

---

## 11. Live Demo URL

Deployed Streamlit app:

https://yothtq3mtgvquhd9ppafsa.streamlit.app/

Verified demo question:

What backend technologies does CharacterForge AI use?

Verified answer:

CharacterForge AI uses Node.js, Express, MongoDB, and Cloudinary for its backend.
