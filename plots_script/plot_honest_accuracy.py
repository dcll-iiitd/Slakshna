import pandas as pd
import matplotlib.pyplot as plt
import os

# Configure matplotlib for a clean look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.titlesize': 14,
    'font.family': 'sans-serif'
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")

def get_malicious_nodes(num_malicious):
    """Returns the list of malicious nodes for a given experiment (matches run_ml_sweep.py)"""
    return [f"node-{i}" for i in range(num_malicious)]

def plot_honest_accuracies(output_file="honest_accuracy_vs_malicious.png"):
    plt.figure(figsize=(10, 6))
    
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    markers = ['o', 's', '^', 'D']
    
    data_found = False

    for m in range(1, 5):
        csv_path = os.path.join(LOG_DIR, f"ml_performance_{m}_malicious.csv")
        
        if not os.path.exists(csv_path):
            print(f"⚠️ Warning: {csv_path} not found. Skipping {m} malicious nodes curve.")
            continue
            
        data_found = True
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        # Identify malicious nodes for this specific run
        malicious_nodes = get_malicious_nodes(m)
        
        # Filter strictly for HONEST nodes
        honest_df = df[~df['node_id'].isin(malicious_nodes)]
        
        # Calculate the average accuracy of honest nodes per round
        avg_acc = honest_df.groupby('round')['accuracy'].mean().reset_index()
        
        # Plot the curve
        label_text = f"{m}0% Malicious ({m}/10 nodes)"
        plt.plot(
            avg_acc['round'], 
            avg_acc['accuracy'], 
            label=label_text,
            color=colors[m-1],
            marker=markers[m-1],
            markersize=5,
            linewidth=2,
            alpha=0.85
        )

    if not data_found:
        print("❌ Error: No performance CSVs found. Have you run `python run_ml_sweep.py` yet?")
        return

    plt.title("Honest Client Accuracy Convergence Under Poisoning Attacks")
    plt.xlabel("Federated Learning Round")
    plt.ylabel("Average Honest Node Accuracy (%)")
    plt.ylim(-5, 105)
    plt.xlim(left=0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", frameon=True, facecolor="white")
    
    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, output_file)
    plt.savefig(output_path, dpi=300)
    print(f"📊 Plot successfully generated and saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_honest_accuracies()
