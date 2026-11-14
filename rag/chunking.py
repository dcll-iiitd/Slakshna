"""Split document text into overlapping word windows."""

DEFAULT_CHUNK_WORDS = 200
DEFAULT_OVERLAP_WORDS = 40


def chunk_text(text, chunk_words=DEFAULT_CHUNK_WORDS, overlap_words=DEFAULT_OVERLAP_WORDS):
    """Return a list of chunks of about `chunk_words` words each.

    Consecutive chunks share `overlap_words` words, so a sentence cut by one
    chunk boundary is still whole in the neighbouring chunk.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words must be in [0, chunk_words)")

    words = text.split()
    step = chunk_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return chunks
