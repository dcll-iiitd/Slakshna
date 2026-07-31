import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

# Read the CSV file
csv_path = "logs/epoch_loss_tracking.csv"
df = pd.read_csv(csv_path)

# Filter for the first step of every epoch
# The step column might be string or int, so let's safely convert and filter
df['step'] = df['step'].astype(int)
df_step1 = df[df['step'] == 1].copy()

# Sort by epoch just in case
df_step1['epoch'] = df_step1['epoch'].astype(int)
df_step1 = df_step1.sort_values('epoch')

# Plotting
plt.figure(figsize=(10, 6))

unique_nodes = df_step1['node_id'].unique()
for node in unique_nodes:
    node_data = df_step1[df_step1['node_id'] == node]
    # Keep node name short for legend
    label = node[:8] + "..." if len(node) > 8 else node
    plt.plot(node_data['epoch'], node_data['loss'], marker='o', linestyle='-', linewidth=2, markersize=8, label=label)

plt.title('Global Model Loss at Step 1 of Each Epoch', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss (Step 1)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(sorted(df_step1['epoch'].unique()))
plt.legend()

# Save to the current directory
local_plot_path = "epoch_loss_step1.png"
plt.savefig(local_plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {os.path.abspath(local_plot_path)}")

# Save a copy to the artifact directory so it can be displayed in the UI
artifact_dir = "/home/gauranshi/.gemini/antigravity-ide/brain/5a10c60a-063c-4cde-b887-1a71550cca63"
if os.path.exists(artifact_dir):
    artifact_plot_path = os.path.join(artifact_dir, "epoch_loss_step1.png")
    shutil.copy(local_plot_path, artifact_plot_path)
    print(f"Plot copied to artifact directory: {artifact_plot_path}")
