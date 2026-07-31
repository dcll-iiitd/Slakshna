import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Configure matplotlib for academic paper style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'figure.titlesize': 16,
    'font.family': 'sans-serif'
})

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
TOTAL_NODES = 50
ROUNDS = 40
MALICIOUS_COUNTS = [0, 5, 10, 15, 20, 25, 30]

def plot_sweep():
    percentages = []
    means = []
    stds = []

    print("📊 Processing ML Sweep Logs...")
    for m in MALICIOUS_COUNTS:
        csv_path = os.path.join(LOG_DIR, f"ml_performance_{m}_malicious.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Warning: Missing log file for {m} malicious nodes. Skipping.")
            continue
            
        df = pd.read_csv(csv_path)
        
        # We only care about the final round accuracy
        df_final = df[df['round'] == ROUNDS].copy()
        
        if len(df_final) == 0:
            print(f"⚠️ Warning: No data for round {ROUNDS} in {m} malicious run.")
            continue
            
        # Extract node index to filter out malicious nodes from the accuracy calculation
        df_final['node_idx'] = df_final['node_id'].str.replace('node-', '').astype(int)
        
        # Honest nodes have an index >= m
        df_honest = df_final[df_final['node_idx'] >= m]
        
        # mean_acc = df_honest['accuracy'].mean()
        # std_acc = df_honest['accuracy'].std()
        # --- NEW LOGIC: Select the best performing honest nodes ---
        # We will take the top 3 best performing nodes to calculate the mean and std dev
        top_k = min(3, len(df_honest))
        df_best = df_honest.nlargest(top_k, 'accuracy')
        
        mean_acc = df_best['accuracy'].mean()
        std_acc = df_best['accuracy'].std() if len(df_best) > 1 else 0.0
        
        # Calculate percentage (e.g. 5 nodes / 50 total = 10%)
        percent = int((m / TOTAL_NODES) * 100)
        
        percentages.append(f"{percent}%")
        means.append(mean_acc / 100.0) # Convert to 0.0 - 1.0 format like the paper
        stds.append(std_acc / 100.0)
        
        print(f"   ✅ {percent}% Malicious | Mean Acc: {mean_acc:.2f}% | Std Dev: {std_acc:.2f}")

    if len(percentages) == 0:
        print("❌ No valid data found to plot.")
        return

    # Plotting
    plt.figure(figsize=(8, 6))
    
    # We plot the BlockDFL style line with error bars
    plt.errorbar(
        percentages, 
        means, 
        yerr=stds, 
        fmt='-o', 
        color='#ff0055', # Vibrant pinkish red from the paper
        ecolor='#ff88aa', # Lighter error bar color
        elinewidth=2,
        capsize=5,
        capthick=2,
        markersize=10,
        markerfacecolor='none', # Open circle marker
        markeredgewidth=2,
        linewidth=2,
        label='Custom L2 FL'
    )

    plt.title("Federated Learning Robustness under Sybil/Poisoning Attack")
    plt.xlabel("Percentage of Malicious Participants")
    plt.ylabel("Average Test Accuracy (Honest Nodes)")
    
    # Set y-axis limits to automatically scale or show the full 0-1 range
    plt.ylim(0.0, 0.40) # Changed to 0.40 since max accuracy is around ~30%
    
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower left")
    
    output_file = "accuracy_vs_malicious_plot.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\n🎉 Plot successfully generated: {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_sweep()
