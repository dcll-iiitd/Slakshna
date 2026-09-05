import os
import pandas as pd
import matplotlib.pyplot as plt

# Load trust scores CSV from dcll/Slakshna/logs/trust_scores_new.csv
csv_path = "/mnt/disk1/slakshna/dcll/Slakshna/logs/trust_scores_new.csv"
df = pd.read_csv(csv_path)
df = df[df["timestamp"] != "timestamp"]
df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
df = df.dropna(subset=["weight"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Identify unique nodes and map synchronization sync events
distinct_timestamps = sorted(df["timestamp"].unique())
time_to_event = {ts: idx + 1 for idx, ts in enumerate(distinct_timestamps)}
df["sync_event"] = df["timestamp"].map(time_to_event)

observer = df["observer_node"].iloc[0]
unique_peers = list(df["peer_node"].unique())

# Setup plot style to match the reference figure
plt.figure(figsize=(7.5, 5.2), dpi=300)

# Colors matching the reference image
color_self = "#1f2d48"    # Deep Navy Blue
color_peer = "#cf6d53"    # Terracotta / Coral Orange

for node in unique_peers:
    node_df = df[df["peer_node"] == node].sort_values("sync_event")
    is_self = (node == observer)
    
    if is_self:
        label = f"Self ({node[:13]}...)"
        color = color_self
    else:
        label = f"Peer ({node[:13]}...)"
        color = color_peer
        
    plt.plot(
        node_df["sync_event"],
        node_df["weight"],
        marker="o",
        markersize=6.5,
        linewidth=2.2,
        color=color,
        label=label
    )

# Title & Labels matching reference
plt.title("C. Dynamic Peer Trust Scores & Model Aggregation Weights", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Synchronization Sync Event", fontsize=11, fontweight="bold")
plt.ylabel(r"Aggregation Weight ($w_i$)", fontsize=11, fontweight="bold")

# Axis Limits and Ticks
plt.xlim(0.6, len(distinct_timestamps) + 0.4)
plt.ylim(-0.05, 1.05)
plt.xticks(range(1, len(distinct_timestamps) + 1), [str(i) for i in range(1, len(distinct_timestamps) + 1)], fontsize=10)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)

# Dotted grid matching reference
plt.grid(True, linestyle=":", color="#d3d3d3", alpha=0.9)

# Legend on the center right
plt.legend(
    loc="center right",
    frameon=True,
    facecolor="white",
    edgecolor="#e0e0e0",
    framealpha=1.0,
    fontsize=9.5
)

plt.tight_layout()

# Save plot to paths
output_paths = [
    "/mnt/disk1/slakshna/dcll/Slakshna/logs/trust_scores_plot.png",
    "/mnt/disk1/slakshna/dcll/Slakshna/results/trust_scores_plot.png",
    "/mnt/disk1/slakshna/dcll/Slakshna/trust_scores_plot.png"
]

for p in output_paths:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    plt.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {p}")

plt.close()
