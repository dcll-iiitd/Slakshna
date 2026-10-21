#!/usr/bin/env bash
# Prepare the per-node /app directories used by compose.workaround-rootfull.yaml.
# Copies the image's /app (ml_engine.py, federated_communication/, ...) to
# $WORKAROUND_APP_ROOT/node{1,2,3}. Safe to re-run: it refreshes the code and keeps
# runtime files already written there.
set -euo pipefail

: "${WORKAROUND_APP_ROOT:?set WORKAROUND_APP_ROOT to a directory on a disk with free space}"
IMAGE="${SLAKSHNA_IMAGE:-slakshna:dev}"
NODES="${WORKAROUND_NODES:-1 2 3}"

for n in $NODES; do
    dir="$WORKAROUND_APP_ROOT/node$n"
    mkdir -p "$dir"
    # runs as the image's user (uid = the UID the image was built with), so the files
    # are owned by the user that will write to them later
    docker run --rm --entrypoint sh -v "$dir:/out" "$IMAGE" -c 'cp -a /app/. /out/'
    echo "prepared $dir ($(ls "$dir" | wc -l) entries)"
done
