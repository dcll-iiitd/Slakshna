"""RAG sidecar for a SLAKSHNA node.

    python rag_engine.py ingest --data-dir ./data-rag-node1 --path corpus/node1
    python rag_engine.py ingest --data-dir ./node --path shard/ --vectors cache/   # pre-embedded snippets
    python rag_engine.py serve  --data-dir ./data-rag-node1 --port 8600

The index lives in <data-dir>/rag/. `serve` listens on 127.0.0.1 only.
"""

import argparse
import os
import sys
import threading
import time

from rag import DEFAULT_EMBEDDING_MODEL


def open_store(args):
    from rag.embedder import Embedder
    from rag.store import VectorStore
    from rag.vector_cache import VectorCache

    print(f"Loading embedding model {args.embedding_model}...", file=sys.stderr)
    embedder = Embedder(args.embedding_model)
    cache = VectorCache(args.vectors) if getattr(args, "vectors", None) else None
    return VectorStore(os.path.join(args.data_dir, "rag"), embedder, n_clusters=args.n_clusters,
                       vector_cache=cache)


def describe_commitment(store):
    summary = store.commitment_summary()
    return (f"{summary['num_clusters']} clusters, Merkle root {summary['root'][:16]}… "
            f"(version {summary['version']})")


def cmd_ingest(args):
    store = open_store(args)
    added = store.add_path(args.path)
    print(f"Added {added} chunks; index now holds {store.num_chunks} chunks "
          f"from {len(store.documents)} documents in {describe_commitment(store)}.", file=sys.stderr)
    if store.vector_cache is not None:
        print(f"Vectors from cache {args.vectors}; {store.vector_cache.checked} spot-checked, "
              f"max difference {store.vector_cache.max_difference:.1e}.", file=sys.stderr)


def exit_when_parent_dies():
    """Stop this process once the process that launched it is gone.

    A node that crashes or is killed cannot clean up its sidecar; without this
    the sidecar would linger and keep the port.
    """
    parent = os.getppid()

    def watch():
        while os.getppid() == parent:
            time.sleep(2)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def cmd_serve(args):
    import uvicorn

    from rag.generator import make_generator
    from rag.server import create_app

    if args.exit_with_parent:
        exit_when_parent_dies()
    store = open_store(args)
    generator = make_generator(args.generator, args.generator_url, args.generator_model)
    print(f"Index holds {store.num_chunks} chunks in {describe_commitment(store)}; "
          f"generator: {generator.name} "
          f"({generator.model}).", file=sys.stderr)
    uvicorn.run(create_app(store, generator), host="127.0.0.1", port=args.port, log_level="warning")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", required=True, help="the node's data_dir; the index is stored in <data-dir>/rag/")
    common.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    common.add_argument("--n-clusters", type=int, default=0,
                        help="k-means clusters to group chunks into; 0 = round(sqrt(number of chunks))")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", parents=[common], help="add documents to the index")
    ingest.add_argument("--path", required=True,
                        help="a PDF/.txt/.md file, a .jsonl file of pre-cut snippets, or a directory of them")
    ingest.add_argument("--vectors", help="directory of pre-computed snippet vectors (scripts/m9/embed_cache.py)")
    ingest.set_defaults(func=cmd_ingest)

    serve = sub.add_parser("serve", parents=[common], help="run the local HTTP service")
    serve.add_argument("--port", type=int, default=8600)
    serve.add_argument("--generator", choices=["ollama", "none"], default="ollama")
    serve.add_argument("--generator-model", default="qwen2.5:1.5b")
    serve.add_argument("--generator-url", default="http://127.0.0.1:11434")
    serve.add_argument("--exit-with-parent", action="store_true",
                       help="stop when the launching process exits (used by the node)")
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
