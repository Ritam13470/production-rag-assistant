import os

from dotenv import load_dotenv


TRUTHY_VALUES = {"1", "true", "yes", "on"}
PLACEHOLDER_SECRET_VALUES = {"", "your_langsmith_api_key_here"}
DEFAULT_LANGSMITH_PROJECT = "production-rag-assistant"


def read_env_flag(name, default="false"):
    value = os.getenv(name, default)
    return value.strip().lower() in TRUTHY_VALUES


def is_real_secret(value):
    if value is None:
        return False

    return value.strip() not in PLACEHOLDER_SECRET_VALUES


def get_langsmith_status():
    load_dotenv()

    tracing_requested = read_env_flag("LANGSMITH_TRACING")
    api_key_configured = is_real_secret(
        os.getenv("LANGSMITH_API_KEY")
    )
    project_name = (
        os.getenv("LANGSMITH_PROJECT", DEFAULT_LANGSMITH_PROJECT).strip()
        or DEFAULT_LANGSMITH_PROJECT
    )

    enabled = tracing_requested and api_key_configured

    if enabled:
        reason = "LangSmith tracing is enabled."

    elif tracing_requested and not api_key_configured:
        reason = (
            "LangSmith tracing was requested, but LANGSMITH_API_KEY "
            "is not configured."
        )

    else:
        reason = "LangSmith tracing is disabled."

    return {
        "enabled": enabled,
        "tracing_requested": tracing_requested,
        "api_key_configured": api_key_configured,
        "project_name": project_name,
        "reason": reason
    }
