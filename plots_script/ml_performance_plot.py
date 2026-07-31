import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure clean, high-resolution styles for research paper formats
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif'
})

def generate_paper_plots(csv_path, output_image_path="ml_convergence.png"):
    if not os.path.exists(csv_path):
        print(f"Error: Target performance log file '{csv_path}' not found.")
        return

    # Load and clean up tracking parameters
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Clear accidental whitespace formatting
    
    # CRITICAL: Sort sequentially to fix the asynchronous thread logging artifacts
    df = df.sort_values(by=["node_id", "round"])

    # Instantiate the figure canvas layout (Side-by-Side subplots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Establish consistent color spaces for individual clients
    unique_nodes = sorted(df['node_id'].unique())
    # Academic color scheme (Using distinct bold tones)
    colors = sns.color_palette("Set1", n_colors=len(unique_nodes))
    node_colors = dict(zip(unique_nodes, colors))

    print("📈 Extracting metrics for plot generation...")

    # Iterate over nodes and plot their individual pathways
    for node in unique_nodes:
        node_df = df[df['node_id'] == node]
        
        # Determine visual weights based on configuration profile
        # highlight node-2 as malicious to show the performance gap explicitly
        if node == "node-2":
            line_style = '--'
            marker_style = 'x'
            label_text = f"{node} (Malicious Poisoner)"
        else:
            line_style = '-'
            marker_style = 'o'
            label_text = f"{node} (Honest)"

        # SUBPLOT 1: Model Loss Degradation Profiles (Loss After Step Adjustment)
        ax1.plot(
            node_df['round'], 
            node_df['loss_after'], 
            label=label_text,
            color=node_colors[node],
            linestyle=line_style,
            marker=marker_style,
            markersize=4,
            alpha=0.85,
            linewidth=1.8
        )
        
        # SUBPLOT 2: Convergence Model Accuracy Velocities
        ax2.plot(
            node_df['round'], 
            node_df['accuracy'], 
            label=label_text,
            color=node_colors[node],
            linestyle=line_style,
            marker=marker_style,
            markersize=4,
            alpha=0.85,
            linewidth=1.8
        )

    # ---------------------------------------------------------
    # Format Subplot 1: Cross-Entropy Loss Curves
    # ---------------------------------------------------------
    ax1.set_title("Model Convergence Profiles (Loss After Step Adjustment)")
    ax1.set_xlabel("Learning Rounds Sequence")
    ax1.set_ylabel("Cross-Entropy Loss Magnitude")
    ax1.set_xlim(left=0)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # ---------------------------------------------------------
    # Format Subplot 2: Accuracy Trends
    # ---------------------------------------------------------
    ax2.set_title("Convergence Velocities: Model Classification Accuracy")
    ax2.set_xlabel("Learning Rounds Sequence")
    ax2.set_ylabel("Validation Evaluation Accuracy (%)")
    ax2.set_ylim(-5, 105)
    ax2.set_xlim(left=0)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Consolidate unified layout legends
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()
    
    # Save output artifacts
    dir_name = os.path.dirname(output_image_path)
    if dir_name:  # Only attempt to create directories if a directory path exists
        os.makedirs(dir_name, exist_ok=True)
        
    plt.savefig(output_image_path, dpi=300)
    print(f"📊 Figures successfully compiled and saved to: {output_image_path}")
    plt.show()

if __name__ == "__main__":
    TARGET_LOG = "logs/ml_performance_new.csv"
    generate_paper_plots(TARGET_LOG)