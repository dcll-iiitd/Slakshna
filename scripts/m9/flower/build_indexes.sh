#!/usr/bin/env bash
# Build Flower's FAISS index for every shard of one scheme, with Flower's own code.
#
#   bash scripts/m9/flower/build_indexes.sh --data datasets/textbooks --scheme book [--nodes 5] [--jobs 5 --threads 24]
#
# Calls fedrag.retriever.Retriever().build_faiss_index for corpus textbooks_s<i>
# (an IndexIVFFlat with nlist = sqrt(snippets), L2 distance, all snippets),
# using the embedding model in flower-fedrag-m9/fedrag/retriever.yaml. The index
# lands in <data>/flower/<scheme>/textbooks_s<i>/. --jobs shards are built at
# once, each with --threads torch threads. Logs: <data>/flower/<scheme>/build_s<i>.log
set -euo pipefail
cd "$(dirname "$0")/../../.."

DATA=""; SCHEME=""; NODES=5; JOBS=1; THREADS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA="$2"; shift 2 ;;
    --scheme) SCHEME="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$DATA" ] && [ -n "$SCHEME" ] || { echo "--data and --scheme are required" >&2; exit 2; }
export FEDRAG_CORPUS_DIR="$(realpath "$DATA/flower/$SCHEME")"
[ "$THREADS" -gt 0 ] && export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"

build() {
  local i=$1
  (cd flower-fedrag-m9 && ../.venv-flower/bin/python -c "
import sys, time, torch
if $THREADS > 0:
    torch.set_num_threads($THREADS)
from fedrag.retriever import Retriever
started = time.time()
Retriever().build_faiss_index('textbooks_s$i', batch_size=32)
print(f'textbooks_s$i built in {time.time() - started:.0f} s', file=sys.stderr)
") > "$FEDRAG_CORPUS_DIR/build_s$i.log" 2>&1 && tail -1 "$FEDRAG_CORPUS_DIR/build_s$i.log" \
    || { echo "shard $i failed; see $FEDRAG_CORPUS_DIR/build_s$i.log" >&2; return 1; }
}

failed=0
for i in $(seq 0 $((NODES - 1))); do
  build "$i" &
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || failed=1; done
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || failed=1; done
exit $failed
