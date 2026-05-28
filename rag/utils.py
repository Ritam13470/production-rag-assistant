def get_response_text(response):
    content = response.content

    if isinstance(content, list):
        text = ""

        for part in content:
            if isinstance(part, dict):
                text += part.get("text", "")
            else:
                text += str(part)

        return text

    return str(content)


def preview_text(text, max_chars=900):
    clean_text = " ".join(text.split())

    if len(clean_text) <= max_chars:
        return clean_text

    return clean_text[:max_chars] + "..."


def format_source_label(doc, score=None):
    source = doc.metadata.get("source", "Unknown source")
    page = doc.metadata.get("page")
    file_type = doc.metadata.get("type")

    label_parts = [source]

    if file_type:
        label_parts.append(f"type: {file_type}")

    if page:
        label_parts.append(f"page: {page}")

    if score is not None:
        label_parts.append(f"distance: {score:.4f}")

    return " | ".join(label_parts)
