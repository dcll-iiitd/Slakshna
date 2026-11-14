"""Apply the M9 changes to a copy of Flower's FedRAG example (decision L6).

    python3 scripts/m9/flower/patch_fedrag.py --app flower-fedrag-m9 --variant m9 --nodes 5 \
        --model BAAI/bge-base-en-v1.5 --dim 768 --mirage datasets/textbooks/raw/benchmark.json

Every change is an exact replacement that must match once, so a different
Flower version fails here instead of being patched wrongly. None of them
touches how clients search or how the server merges:

- fedrag/retriever.py: the corpus folder comes from FEDRAG_CORPUS_DIR (the
  Deployment Runtime runs an installed copy of the app, away from the example's
  data/ folder); FEDRAG_NPROBE optionally widens the IVF search (run E2b).
- fedrag/retriever.yaml: the embedding model shared with Slakshna.
- pyproject.toml: the run config (corpora, k values, M9 keys); for the m9
  variant the ServerApp is fedrag.m9_server_app, which saves contexts.
- data/mirage.json: MIRAGE questions, which the stock ServerApp reads.

Variant `stock` keeps Flower's own ServerApp (with its LLM), for check E0(a).
Standard library only.
"""

import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))


def replace_once(path, old, new):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.count(old) != 1:
        raise SystemExit(f"{path}: expected exactly one match of:\n{old}\nFlower's code changed; update the patch.")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new))


def run_config(args):
    corpora = "|".join(f"textbooks_s{i}" for i in range(args.nodes))
    return f'''[tool.flwr.app.config]
# Written by scripts/m9/flower/patch_fedrag.py (Slakshna M9). Flower's own keys:
server-qa-datasets = "medqa|mmlu"
server-qa-num = 0  # 0 = all questions
clients-corpus-names = "{corpora}"
k-rrf = 60
k-nn = 8
server-llm-hfpath = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
server-llm-use-gpu = "false"
# M9 keys (read by fedrag/m9_server_app.py); paths must be absolute.
m9-questions = ""
m9-out = ""
m9-qids = ""
m9-system = "flower"
m9-scheme = ""
m9-nodes = {args.nodes}
m9-node-timeout = 300
m9-repeat = 1
m9-warmup = 0
m9-llm = "false"
'''


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--app", required=True, help="a fresh copy of examples/fedrag")
    p.add_argument("--variant", choices=["m9", "stock"], required=True)
    p.add_argument("--nodes", type=int, default=5)
    p.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--mirage", help="MIRAGE benchmark.json, copied to data/mirage.json (stock variant)")
    args = p.parse_args()

    retriever = os.path.join(args.app, "fedrag", "retriever.py")
    replace_once(retriever,
                 'CORPUS_DIR = os.path.join(DIR_PATH, "../data/corpus")\n',
                 '# M9: the corpus folder can be set from outside (see scripts/m9/flower/patch_fedrag.py).\n'
                 'CORPUS_DIR = os.environ.get("FEDRAG_CORPUS_DIR", os.path.join(DIR_PATH, "../data/corpus"))\n')
    replace_once(retriever,
                 '        index = faiss.read_index(index_path)\n',
                 '        index = faiss.read_index(index_path)\n'
                 '        # M9 (run E2b only): search more IVF buckets; "all" = exact search.\n'
                 '        if os.environ.get("FEDRAG_NPROBE"):\n'
                 '            nprobe = os.environ["FEDRAG_NPROBE"]\n'
                 '            index.nprobe = index.nlist if nprobe == "all" else int(nprobe)\n')

    with open(os.path.join(args.app, "fedrag", "retriever.yaml"), "w", encoding="utf-8") as f:
        f.write(f'embedding_model: "{args.model}"\nembedding_dimension: {args.dim}\n')

    pyproject = os.path.join(args.app, "pyproject.toml")
    with open(pyproject, encoding="utf-8") as f:
        text = f.read()
    start = text.index("[tool.flwr.app.config]")
    if "\n[" in text[start + 1:]:
        raise SystemExit(f"{pyproject}: [tool.flwr.app.config] is no longer the last section; update the patch")
    with open(pyproject, "w", encoding="utf-8") as f:
        f.write(text[:start] + run_config(args))
    if args.variant == "m9":
        replace_once(pyproject, 'serverapp = "fedrag.server_app:app"', 'serverapp = "fedrag.m9_server_app:app"')
        shutil.copy(os.path.join(HERE, "m9_server_app.py"), os.path.join(args.app, "fedrag", "m9_server_app.py"))
    if args.mirage:
        shutil.copy(args.mirage, os.path.join(args.app, "data", "mirage.json"))
    print(f"patched {args.app} ({args.variant}, {args.nodes} corpora, {args.model})")


if __name__ == "__main__":
    main()
