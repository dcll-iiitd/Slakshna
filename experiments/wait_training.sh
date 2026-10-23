#!/usr/bin/env bash
# Block until every node has published the final round, or any ML engine fails. usage: wait_training.sh <tag> <rounds> <nodes>
TAG="$1"; R="$2"; N="$3"; L=experiments/runs/$TAG/logs
strip(){ sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$@"; }
while true; do
  done_n=0; for k in $(seq 1 $N); do strip $L/node${k}_train.log | grep -q "Publishing adapter round $R " && done_n=$((done_n+1)); done
  [ "$done_n" = "$N" ] && { echo "ALL_PUBLISHED round $R $(date -u +%T)"; break; }
  strip $L/node*_train.log | grep -q "ML Engine failed" && { echo "FAILED $(date -u +%T)"; strip $L/node*_train.log | grep -A8 "ML Engine failed" | head -30 | cut -c1-250; break; }
  sleep 20
done
for k in $(seq 1 $N); do strip $L/node${k}_train.log | grep -E "Publishing adapter|Local Training Complete" | cut -c1-160; done
