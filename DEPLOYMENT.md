# Production RAG Assistant Deployment Guide

This guide explains how to prepare and deploy the Streamlit RAG app safely.

---

## 1. Important Secret Safety Rules

Never commit real secrets to GitHub.

Private files that should stay local:

- .env
- .streamlit/secrets.toml

Safe template files that can be committed:

- .env.example
- .streamlit/secrets.example.toml

Deployment platforms should store GOOGLE_API_KEY as a secret or environment variable.

---

## 2. Local Pre-Deployment Checks

Run these commands before deploying:

git status
python smoke_test.py
python startup_check.py

Expected:

- Git working tree is clean
- Smoke test passes
- Startup checklist passes
- No Gemini API calls are made by these checks

---

## 3. Local Streamlit Test

Run:

streamlit run app.py

Open the local URL shown in the terminal.

Confirm these sections appear:

- Project Overview
- Document Dashboard
- Upload Documents
- Delete Uploaded Document
- Manage Vector Database
- Ask Questions
- Query History

---

## 4. Streamlit Community Cloud Deployment Notes

Recommended app entrypoint:

app.py

Required secret:

GOOGLE_API_KEY

Use the platform secrets manager instead of committing .env.

Important limitation:

Uploaded files and local ChromaDB may not be permanent on free/cloud hosting. For a long-term production deployment, use persistent object storage and a hosted vector database.

---

## 5. Render Deployment Notes

Possible start command:

streamlit run app.py --server.port $PORT --server.address 0.0.0.0

Required environment variable:

GOOGLE_API_KEY

Important limitation:

Free hosting disks can be temporary. For persistent document storage and vector storage, use external services.

---

## 6. Deployment v1 Scope

Deployment v1 is intended for demo access.

It is suitable for:

- Teacher review
- Project presentation
- Feature demonstration
- Browser-based demo of the Streamlit dashboard

It is not yet a full production multi-user system.

---

## 7. Future Production Deployment Improvements

Recommended future upgrades:

- Persistent document storage
- Hosted vector database such as Supabase PGVector
- Authentication
- User-level document isolation
- Rate limiting
- Logging and observability
- LangSmith tracing
- Deployment health checks
- Background indexing jobs

---

## 8. Quick Demo Deployment Checklist

Before sharing a deployed link:

1. Confirm GOOGLE_API_KEY is set in the deployment secrets
2. Run startup_check.py locally
3. Run smoke_test.py locally
4. Test streamlit run app.py locally
5. Deploy the app
6. Open the deployed URL
7. Upload or verify sample documents
8. Rebuild the vector database
9. Ask a known question
10. Confirm answer, sources, and query history work
