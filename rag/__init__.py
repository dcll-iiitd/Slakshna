"""Retrieval-augmented generation sidecar for a SLAKSHNA node.

One sidecar runs next to each node. It owns that node's document index and
answers three local requests: retrieve chunks for a query, generate an answer
from a query plus chunks, and ingest new documents. It never talks to other
nodes; all peer traffic goes through the Rust node.
"""

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
