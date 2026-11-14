"""Local HTTP API of the RAG sidecar. Bound to 127.0.0.1 by rag_engine.py.

The routes behind peer queries (`/centroids`, `/search_clusters`, and
`/retrieve` with proofs) answer for one version of the Merkle root and say
which.
"""

import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .merkle import centroid_bytes


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    with_proofs: bool = False


class GenerateRequest(BaseModel):
    query: str
    chunks: List[dict]


class Document(BaseModel):
    id: str
    text: str


class IngestRequest(BaseModel):
    path: Optional[str] = None
    documents: Optional[List[Document]] = None


class EmbedRequest(BaseModel):
    texts: List[str]


class CentroidsRequest(BaseModel):
    query: str
    n: int = Field(default=3, ge=0)


class SearchClustersRequest(BaseModel):
    query: str
    root_version: int
    clusters: List[int]
    per_cluster: int = Field(default=3, ge=0)


def create_app(store, generator):
    app = FastAPI(title="SLAKSHNA RAG sidecar")

    def embed_query(query):
        return store.embedder.embed([query])[0]

    @app.get("/health")
    def health():
        summary = store.commitment_summary()
        return {
            "ready": True,
            "embedding_model": store.embedder.model_name,
            "dim": store.embedder.dim,
            "num_chunks": store.num_chunks,
            "num_documents": len(store.documents),
            "generator": generator.name,
            "generator_model": generator.model,
            "commitment": {"root": summary["root"], "version": summary["version"]},
        }

    @app.get("/commitment")
    def commitment():
        # Unsigned: the Rust node signs it with its key.
        return store.commitment_summary()

    @app.post("/embed")
    def embed(request: EmbedRequest):
        vectors = store.embedder.embed(request.texts) if request.texts else []
        return {"dim": store.embedder.dim, "vectors": [centroid_bytes(v).hex() for v in vectors]}

    @app.post("/retrieve")
    def retrieve(request: RetrieveRequest):
        started = time.perf_counter()
        response = {"embedding_model": store.embedder.model_name}
        if request.with_proofs:
            with store.lock:
                response["root_version"] = store.version
                response.update(store.search(request.query, request.top_k, with_proofs=True))
        else:
            response["chunks"] = store.search(request.query, request.top_k)
        response["retrieve_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return response

    @app.get("/clusters")
    def clusters():
        # The whole centroid directory, for peers' directory mode.
        with store.lock:
            return {"root_version": store.version, "clusters": store.all_clusters()}

    @app.post("/centroids")
    def centroids(request: CentroidsRequest):
        q = embed_query(request.query)
        with store.lock:
            return {"root_version": store.version, "centroids": store.nearest_centroids(q, request.n)}

    @app.post("/search_clusters")
    def search_clusters(request: SearchClustersRequest):
        q = embed_query(request.query)
        with store.lock:
            if request.root_version != store.version:
                # The asker chose clusters from an older (or unknown) root;
                # cluster ids may mean something else now.
                raise HTTPException(status_code=409, detail={
                    "error": "stale_root",
                    "root": store.commitment_summary(),
                })
            try:
                result = store.search_clusters(q, request.clusters, request.per_cluster)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            return {"root_version": store.version, **result}

    @app.post("/generate")
    def generate(request: GenerateRequest):
        started = time.perf_counter()
        try:
            answer = generator.generate(request.query, request.chunks)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"generator failed: {e}")
        return {
            "answer": answer,
            "model": generator.model,
            "generate_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    @app.post("/ingest")
    def ingest(request: IngestRequest):
        if (request.path is None) == (request.documents is None):
            raise HTTPException(status_code=400, detail="give exactly one of 'path' or 'documents'")
        try:
            if request.path is not None:
                added = store.add_path(request.path)
            else:
                added = store.add_documents((doc.id, doc.text) for doc in request.documents)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        # A new root_version tells the node to sign its root again.
        with store.lock:
            return {"added_chunks": added, "num_chunks": store.num_chunks, "root_version": store.version}

    return app
