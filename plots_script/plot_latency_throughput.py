import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Configure matplotlib for a clean look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.titlesize': 14,
    'font.family': 'sans-serif'
})

def plot_latency_vs_throughput(csv_file="logs/benchmark_results_multi.csv", output_file="latency_vs_throughput.png"):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    # Load data
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    
    # We only care about rows where Confirmation_Time is available
    if "Confirmation_Time" not in df.columns:
        print("Error: Confirmation_Time column missing. Make sure you run the updated benchmark scripts.")
        return
    
    df = df.dropna(subset=["Confirmation_Time"])
    
    # Calculate Throughput (tx/s)
    # Throughput = Network_Total_TX / Confirmation_Time
    df['Throughput'] = df['Network_Total_TX'] / df['Confirmation_Time']
    df['Latency'] = df['Confirmation_Time']
    
    # Group by transaction type and target load to get averages in case of multiple runs
    grouped = df.groupby(['Type', 'Target_TX']).agg({
        'Throughput': 'mean',
        'Latency': 'mean'
    }).reset_index()

    # Create a figure with two side-by-side subplots for professional academic styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    types = grouped['Type'].unique()
    colors = ['#e74c3c', '#2980b9', '#27ae60', '#f39c12']
    
    for i, tx_type in enumerate(types):
        # Sort by Target_TX so the line goes smoothly from left to right
        type_data = grouped[grouped['Type'] == tx_type].sort_values(by='Target_TX')
        
        # Subplot 1: Latency vs Load
        ax1.plot(
            type_data['Target_TX'], 
            type_data['Latency'], 
            marker='o', 
            markersize=8,
            linewidth=2,
            label=f"{tx_type.capitalize()} Transactions",
            color=colors[i % len(colors)]
        )
        
        # Subplot 2: Throughput vs Load
        ax2.plot(
            type_data['Target_TX'], 
            type_data['Throughput'], 
            marker='s', 
            markersize=8,
            linewidth=2,
            label=f"{tx_type.capitalize()} Transactions",
            color=colors[i % len(colors)]
        )

    # Style Subplot 1 (Latency)
    ax1.set_title("Confirmation Latency vs Transaction Load")
    ax1.set_xlabel("Transaction Load (Total TXs Sent)")
    ax1.set_ylabel("Latency (Seconds to Confirm)")
    # Set y-axis to start at 0 so the ~1-3s variations look appropriately flat
    ax1.set_ylim(0, max(grouped['Latency']) + 2) 
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(loc="upper left")
    
    # Style Subplot 2 (Throughput)
    ax2.set_title("Network Throughput vs Transaction Load")
    ax2.set_xlabel("Transaction Load (Total TXs Sent)")
    ax2.set_ylabel("Throughput (Transactions / Second)")
    ax2.set_ylim(0, max(grouped['Throughput']) * 1.2)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"📊 Professional plot successfully generated and saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_latency_vs_throughput()

