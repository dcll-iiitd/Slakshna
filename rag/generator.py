"""Write an answer from a question and retrieved chunks."""

import httpx

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages provided. "
    "Keep the answer short and factual. If the passages do not contain the answer, "
    "say that the context does not contain it."
)


def build_prompt(query, chunks):
    passages = "\n\n".join(f"[{i}] {chunk['text']}" for i, chunk in enumerate(chunks, start=1))
    return f"Context passages:\n\n{passages}\n\nQuestion: {query}\nAnswer:"


class NoneGenerator:
    """Retrieval-only mode: returns no answer."""

    name = "none"
    model = None

    def generate(self, query, chunks):
        return ""


class OllamaGenerator:
    name = "ollama"

    def __init__(self, url, model, timeout_secs=300.0):
        self.url = url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=timeout_secs)

    def generate(self, query, chunks):
        response = self.client.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(query, chunks)},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


def make_generator(kind, url, model):
    if kind == "none":
        return NoneGenerator()
    if kind == "ollama":
        return OllamaGenerator(url, model)
    raise ValueError(f"unknown generator '{kind}' (expected 'ollama' or 'none')")
