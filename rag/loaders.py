"""Read source documents into plain text, or pre-cut snippets from JSONL."""

import json
import os

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".jsonl")


def load_document(path):
    """Return the text of one PDF, .txt or .md file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    raise ValueError(f"unsupported document type: {path}")


def load_snippets(path):
    """Return the `(chunk_id, text)` pairs of a JSONL file of pre-cut snippets.

    Each line is a MedRAG snippet: `{"id", "title", "content", ...}`. The text
    is `content`, as Flower's FedRAG embeds it. `orig_id`, when present, is the
    id the snippet had before a shard file renamed it (scripts/m9), and wins.
    """
    snippets = []
    with open(path, "r", encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                snippets.append((str(row.get("orig_id") or row["id"]), row["content"]))
            except KeyError as e:
                raise ValueError(f"{path}:{number}: snippet has no {e} field") from e
    return snippets


def list_documents(path):
    """Return every supported file at `path` (a file, or a directory searched recursively)."""
    if os.path.isfile(path):
        return [path]
    found = []
    for root, _, files in os.walk(path):
        for name in files:
            if name.lower().endswith(SUPPORTED_EXTENSIONS):
                found.append(os.path.join(root, name))
    return sorted(found)
