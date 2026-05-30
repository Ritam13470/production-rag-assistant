from rag.utils import preview_text


def initialize_query_history(session_state):
    if "query_history" not in session_state:
        session_state.query_history = []


def clear_query_history(session_state):
    session_state.query_history = []


def get_query_history_count(session_state):
    return len(session_state.query_history)


def add_query_history_entry(session_state, question, answer, scored_docs):
    sources = []

    for doc, score in scored_docs:
        sources.append(
            {
                "source": doc.metadata.get("source", "Unknown source"),
                "page": doc.metadata.get("page"),
                "type": doc.metadata.get("type"),
                "score": score,
                "preview": preview_text(doc.page_content, max_chars=500)
            }
        )

    session_state.query_history.insert(
        0,
        {
            "question": question,
            "answer": answer,
            "sources": sources
        }
    )


def get_query_history(session_state):
    return session_state.query_history
