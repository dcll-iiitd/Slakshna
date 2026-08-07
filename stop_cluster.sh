#!/usr/bin/env bash

if [ -f "cluster.pids" ]; then
    echo "Stopping cluster nodes..."
    while read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "Killed process $pid"
        fi
    done < cluster.pids
    rm cluster.pids
    echo "✅ Cluster stopped."
else
    echo "No cluster.pids file found. Searching for running iiitd processes..."
    pkill iiitd || echo "No iiitd processes found."
    echo "✅ Cluster stopped."
fi
