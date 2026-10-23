#!/usr/bin/env bash
# Make every node of a run "cold": forget all peers' adapters it has downloaded.
# Deletes each node's iroh blob store (a node re-adds its own adapter blob when it starts) and the adapter
# folders it fetched from peers (fetched copies have no adapter_meta.json; the node's own exports do).
# Run only while the run's nodes are stopped.   usage: experiments/reset_caches.sh <tag>
TAG="$1"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for d in "$REPO"/experiments/runs/"$TAG"/data-node*; do
  rm -rf "$d/blobs"
  for a in "$d"/adapters/*/; do [ -f "$a/adapter_meta.json" ] || rm -rf "$a"; done
  rm -rf "$d"/adapters/.fetch_*
done
echo "caches reset for $TAG"
