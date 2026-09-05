#!/usr/bin/env python3
"""
plot_loss.py

Generates detailed, publication-quality visualizations of Slakshna federated learning:
1. Continuous training loss trajectory with explicit federated merge/sync boundaries.
2. Pre-merge (local model) vs Post-merge (globally aggregated model) loss & perplexity comparison.
3. Learning Rate (LR) schedule & step evolution across federated rounds (with merge resets).
4. Model update delta L2 norm convergence and token/sample throughput.
5. Dual-axis Loss vs. Learning Rate progression.

Outputs saved to `results/` and copied to the active IDE artifact directory.
"""

from __future__ import annotations

import os
import shutil
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# Set high-quality styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 1.0


def compute_lr_schedule(step: int, max_steps: int = 8, base_lr: float = 3.0e-4, min_lr: float = 3.0e-5, warmup_steps: int = 0) -> float:
    """Computes learning rate for a given step within an epoch based on CosineAnnealing schedule."""
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * (1e-3 + (1.0 - 1e-3) * (step / warmup_steps))
    
    # Cosine decay
    effective_max = max(max_steps, step)
    if warmup_steps > 0:
        progress = (step - warmup_steps) / max(1, effective_max - warmup_steps)
    else:
        progress = step / effective_max
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress))


def load_data(logs_dir: str):
    """Loads loss tracking and communication event logs."""
    loss_path = os.path.join(logs_dir, "epoch_loss_tracking_muon.csv")
    comm_path = os.path.join(logs_dir, "runtime_comm.log")

    df_loss = pd.DataFrame()
    if os.path.exists(loss_path):
        df_loss = pd.read_csv(loss_path)
        df_loss['step'] = pd.to_numeric(df_loss['step'], errors='coerce')
        df_loss['epoch'] = pd.to_numeric(df_loss['epoch'], errors='coerce')
        df_loss['loss'] = pd.to_numeric(df_loss['loss'], errors='coerce')
        df_loss['perplexity'] = pd.to_numeric(df_loss['perplexity'], errors='coerce')
        df_loss = df_loss.dropna(subset=['epoch', 'step', 'loss'])
        df_loss['epoch'] = df_loss['epoch'].astype(int)
        df_loss['step'] = df_loss['step'].astype(int)

    delta_norms = []
    if os.path.exists(comm_path):
        try:
            with open(comm_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 6 and parts[3] == "delta_encoded":
                        ts = parts[0]
                        try:
                            norm_val = float(parts[5])
                            delta_norms.append({"timestamp": ts, "delta_l2_norm": norm_val})
                        except ValueError:
                            pass
        except Exception as e:
            print(f"[!] Warning reading comm log: {e}")
    df_delta = pd.DataFrame(delta_norms)

    return df_loss, df_delta


def build_continuous_dataframe(df_loss: pd.DataFrame) -> pd.DataFrame:
    """Constructs a continuous global step timeline across epochs with computed LR."""
    df_sorted = df_loss.sort_values(by=['epoch', 'timestamp', 'step']).copy().reset_index(drop=True)
    
    # Filter out redundant intermediate zero-sample entries if any
    df_sorted = df_sorted.drop_duplicates(subset=['epoch', 'step'], keep='first').reset_index(drop=True)
    df_sorted['global_step'] = np.arange(1, len(df_sorted) + 1)
    
    # Compute learning rate for each step
    lrs = []
    for _, row in df_sorted.iterrows():
        ep = int(row['epoch'])
        st = int(row['step'])
        ep_max = df_sorted[df_sorted['epoch'] == ep]['step'].max()
        lrs.append(compute_lr_schedule(step=st, max_steps=max(ep_max, 8), base_lr=3.0e-4, min_lr=3.0e-5))
    df_sorted['lr'] = lrs
    
    return df_sorted


def plot_detailed_dashboard(df_loss: pd.DataFrame, df_delta: pd.DataFrame, output_path: str):
    """Creates a comprehensive 4-panel federated learning dashboard with LR in Panel C."""
    df_cont = build_continuous_dataframe(df_loss)
    if df_cont.empty:
        print("[!] No loss data available to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw={'hspace': 0.35, 'wspace': 0.25})
    epochs = sorted(df_cont['epoch'].unique())

    # Identify epoch boundaries (where merging / aggregation happens)
    merge_indices = []
    epoch_ranges = {}
    
    for ep in epochs:
        ep_data = df_cont[df_cont['epoch'] == ep]
        start_idx = ep_data['global_step'].iloc[0]
        end_idx = ep_data['global_step'].iloc[-1]
        epoch_ranges[ep] = (start_idx, end_idx)
        if ep > epochs[0]:
            merge_indices.append(start_idx - 0.5)

    # -------------------------------------------------------------
    # Panel 1: Full Continuous Loss Trajectory with FL Sync Boundaries
    # -------------------------------------------------------------
    ax1 = axes[0, 0]

    # Shaded alternating epoch bands
    for i, ep in enumerate(epochs):
        start_x, end_x = epoch_ranges[ep]
        bg_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        ax1.axvspan(start_x - 0.4, end_x + 0.4, color=bg_color, zorder=0)
        mid_x = (start_x + end_x) / 2
        ax1.text(mid_x, df_cont['loss'].max() * 0.98, f"R{ep}", 
                 ha='center', va='top', fontsize=8.5, fontweight='bold', color='#495057',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#ced4da', alpha=0.9))

    # Plot continuous loss line & markers
    ax1.plot(df_cont['global_step'], df_cont['loss'], color='#2b5c8f', linewidth=2.2, zorder=3, label='Training Loss')
    ax1.scatter(df_cont['global_step'], df_cont['loss'], color='#1d3557', s=30, zorder=4)

    # Highlight step 1 (Post-Merge initial loss) with distinctive red diamonds
    step1_mask = df_cont['step'] == 1
    ax1.scatter(df_cont.loc[step1_mask, 'global_step'], df_cont.loc[step1_mask, 'loss'], 
                color='#e63946', s=65, marker='D', zorder=5, label='Post-Merge Step 1 (Global State)')

    # Draw vertical merge event lines
    for merge_x in merge_indices:
        ax1.axvline(x=merge_x, color='#e63946', linestyle='--', linewidth=1.4, alpha=0.75, zorder=2)

    ax1.set_title('A. Continuous Training Loss & Federated Merge Boundaries (FL Sync)', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Continuous Optimization Step', fontsize=11, fontweight='semibold')
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=11, fontweight='semibold')
    ax1.grid(True, linestyle=':', alpha=0.6, zorder=1)
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: Pre-Merge (Local) vs Post-Merge (Global) Loss Comparison
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    
    post_merge_loss = []
    pre_merge_loss = []
    round_nums = []

    for ep in epochs:
        ep_data = df_cont[df_cont['epoch'] == ep]
        if len(ep_data) > 0:
            round_nums.append(ep)
            post_merge_loss.append(ep_data['loss'].iloc[0]) # step 1
            pre_merge_loss.append(ep_data['loss'].iloc[-1]) # final step

    round_nums = np.array(round_nums)
    width = 0.35

    ax2.bar(round_nums - width/2, post_merge_loss, width=width, color='#457b9d', label='Post-Merge Initial Loss (Step 1)', alpha=0.9, zorder=3)
    ax2.bar(round_nums + width/2, pre_merge_loss, width=width, color='#2a9d8f', label='Pre-Merge Final Loss (Local)', alpha=0.9, zorder=3)

    # Secondary axis for Perplexity
    ax2_ppl = ax2.twinx()
    ppl_step1 = [np.exp(l) if l < 15 else np.nan for l in post_merge_loss]
    ax2_ppl.plot(round_nums, ppl_step1, color='#e76f51', marker='s', linewidth=2.0, linestyle='-', label='Post-Merge Perplexity', zorder=4)
    ax2_ppl.set_ylabel('Perplexity (PPL)', color='#e76f51', fontsize=11, fontweight='semibold')
    ax2_ppl.tick_params(axis='y', labelcolor='#e76f51')

    ax2.set_title('B. Post-Merge (Global Model) vs Pre-Merge (Local Model) Loss', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Federated Round (Epoch)', fontsize=11, fontweight='semibold')
    ax2.set_ylabel('Loss', fontsize=11, fontweight='semibold')
    ax2.set_xticks(round_nums)
    ax2.grid(True, linestyle=':', alpha=0.6, zorder=1)
    
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_ppl.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='upper right', framealpha=0.95, fontsize=8.5)

    # -------------------------------------------------------------
    # Panel 3: Learning Rate (LR) Schedule & Step Evolution (Replaces Trust Score)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]

    # Shaded alternating epoch bands
    for i, ep in enumerate(epochs):
        start_x, end_x = epoch_ranges[ep]
        bg_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        ax3.axvspan(start_x - 0.4, end_x + 0.4, color=bg_color, zorder=0)

    # Plot LR curve
    ax3.plot(df_cont['global_step'], df_cont['lr'], color='#d90429', linewidth=2.2, zorder=3, label='Learning Rate (Cosine Decay)')
    ax3.scatter(df_cont['global_step'], df_cont['lr'], color='#7209b7', s=28, zorder=4)

    # Highlight merge boundaries & LR resets
    for merge_x in merge_indices:
        ax3.axvline(x=merge_x, color='#e63946', linestyle='--', linewidth=1.4, alpha=0.75, zorder=2)

    ax3.set_title('C. Learning Rate (LR) Schedule & Periodic Reset at Merge Boundaries', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('Continuous Optimization Step', fontsize=11, fontweight='semibold')
    ax3.set_ylabel('Learning Rate (LR)', fontsize=11, fontweight='semibold')
    ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax3.grid(True, linestyle=':', alpha=0.6, zorder=1)
    ax3.legend(loc='upper right', framealpha=0.95, fontsize=9)

    # -------------------------------------------------------------
    # Panel 4: Update Delta L2 Norm & Token Throughput Progress
    # -------------------------------------------------------------
    ax4 = axes[1, 1]

    if not df_delta.empty and 'delta_l2_norm' in df_delta.columns:
        rounds_delta = np.arange(1, len(df_delta) + 1)
        color_delta = '#9b5de5'
        ax4.plot(rounds_delta, df_delta['delta_l2_norm'], marker='^', color=color_delta, linewidth=2.2, markersize=8, label=r'Model Delta L2 Norm ($||\Delta||_2$)')
        ax4.set_xlabel('Federated Round', fontsize=11, fontweight='semibold')
        ax4.set_ylabel(r'Delta L2 Norm ($||\Delta||_2$)', color=color_delta, fontsize=11, fontweight='semibold')
        ax4.tick_params(axis='y', labelcolor=color_delta)
        ax4.set_xticks(rounds_delta)
        ax4.grid(True, linestyle=':', alpha=0.6)

        # Plot cumulative tokens on twin axis
        if 'tokens' in df_cont.columns:
            ax4_tok = ax4.twinx()
            max_tok_per_round = [df_cont[df_cont['epoch'] == ep]['tokens'].max() for ep in epochs if ep <= len(rounds_delta)]
            color_tok = '#00bbf9'
            ax4_tok.plot(rounds_delta[:len(max_tok_per_round)], np.array(max_tok_per_round) / 1000, 
                         color=color_tok, marker='o', linestyle='--', linewidth=1.8, label='Tokens Processed (kTok)')
            ax4_tok.set_ylabel('Tokens Processed (kilo-tokens)', color=color_tok, fontsize=11, fontweight='semibold')
            ax4_tok.tick_params(axis='y', labelcolor=color_tok)

        ax4.set_title('D. Gradient Delta L2 Norm Convergence & Token Scale', fontsize=13, fontweight='bold', pad=10)
    else:
        # Fallback to tokens vs loss
        ax4.plot(df_cont['global_step'], df_cont['loss'], color='#2b5c8f', label='Loss')
        ax4.set_title('D. Training Progression', fontsize=13, fontweight='bold')

    plt.suptitle('Slakshna Federated Learning — Comprehensive Training, LR & Merging Analysis', fontsize=16, fontweight='bold', y=0.99)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved Detailed Dashboard: {output_path}")


def plot_loss_and_lr_dual_axis(df_loss: pd.DataFrame, output_path: str):
    """Generates a dedicated dual-axis visualization of Loss and Learning Rate across steps."""
    df_cont = build_continuous_dataframe(df_loss)
    if df_cont.empty:
        return

    epochs = sorted(df_cont['epoch'].unique())
    merge_indices = []
    epoch_ranges = {}
    
    for ep in epochs:
        ep_data = df_cont[df_cont['epoch'] == ep]
        start_idx = ep_data['global_step'].iloc[0]
        end_idx = ep_data['global_step'].iloc[-1]
        epoch_ranges[ep] = (start_idx, end_idx)
        if ep > epochs[0]:
            merge_indices.append(start_idx - 0.5)

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Shaded alternating epoch bands
    for i, ep in enumerate(epochs):
        start_x, end_x = epoch_ranges[ep]
        bg_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        ax1.axvspan(start_x - 0.4, end_x + 0.4, color=bg_color, zorder=0)
        mid_x = (start_x + end_x) / 2
        ax1.text(mid_x, df_cont['loss'].max() * 0.98, f"Round {ep}", 
                 ha='center', va='top', fontsize=9, fontweight='bold', color='#495057',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ced4da', alpha=0.9))

    # Primary axis: Loss
    line_loss = ax1.plot(df_cont['global_step'], df_cont['loss'], color='#1d3557', linewidth=2.4, zorder=3, label='Training Loss')
    ax1.scatter(df_cont['global_step'], df_cont['loss'], color='#1d3557', s=35, zorder=4)

    # Highlight step 1 (Post-Merge initial loss)
    step1_mask = df_cont['step'] == 1
    ax1.scatter(df_cont.loc[step1_mask, 'global_step'], df_cont.loc[step1_mask, 'loss'], 
                color='#e63946', s=70, marker='D', zorder=5, label='Post-Merge Step 1 (Global State)')

    ax1.set_xlabel('Continuous Optimization Step', fontsize=12, fontweight='semibold')
    ax1.set_ylabel('Cross-Entropy Loss', color='#1d3557', fontsize=12, fontweight='semibold')
    ax1.tick_params(axis='y', labelcolor='#1d3557')
    ax1.grid(True, linestyle=':', alpha=0.6, zorder=1)

    # Secondary axis: Learning Rate (LR)
    ax2 = ax1.twinx()
    line_lr = ax2.plot(df_cont['global_step'], df_cont['lr'], color='#d90429', linestyle='--', linewidth=2.0, zorder=3, label='Learning Rate (LR)')
    ax2.set_ylabel('Learning Rate', color='#d90429', fontsize=12, fontweight='semibold')
    ax2.tick_params(axis='y', labelcolor='#d90429')
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Draw vertical merge lines
    for merge_x in merge_indices:
        ax1.axvline(x=merge_x, color='#e63946', linestyle=':', linewidth=1.5, alpha=0.7, zorder=2)

    ax1.set_title('Slakshna Federated Learning — Training Loss & Learning Rate (LR) Schedule\n(with Federated Merge Boundaries)', fontsize=15, fontweight='bold', pad=14)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', framealpha=0.95, fontsize=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved Loss & LR Dual-Axis Plot: {output_path}")


def plot_focused_step1_loss(df_loss: pd.DataFrame, output_path: str):
    """Generates the enhanced step 1 epoch loss plot (compatible with original output)."""
    df_step1 = df_loss[df_loss['step'] == 1].copy().sort_values('epoch')
    df_final = df_loss.groupby('epoch').last().reset_index()

    fig, ax = plt.subplots(figsize=(11, 6.5))

    unique_nodes = df_step1['node_id'].unique()
    for node in unique_nodes:
        node_step1 = df_step1[df_step1['node_id'] == node]
        node_final = df_final[df_final['node_id'] == node]
        
        # Step 1 (Post-Merge global model)
        ax.plot(node_step1['epoch'], node_step1['loss'], marker='o', color='#1d3557', linestyle='-', 
                linewidth=2.5, markersize=9, label=f'Global Model Loss (Post-Merge Step 1)', zorder=4)
        
        # Final step (Pre-Merge local model)
        ax.plot(node_final['epoch'], node_final['loss'], marker='s', color='#2a9d8f', linestyle='--', 
                linewidth=2.0, markersize=7, label=f'Local Model Loss (Pre-Merge Final Step)', zorder=3)
        
        # Connect each round's drop with subtle vertical markers
        for _, row in node_step1.iterrows():
            ep = row['epoch']
            l_start = row['loss']
            l_end = node_final.loc[node_final['epoch'] == ep, 'loss'].values
            if len(l_end) > 0:
                ax.vlines(x=ep, ymin=l_end[0], ymax=l_start, color='#e63946', linestyle=':', alpha=0.6, linewidth=1.5)

    ax.set_title('Slakshna Federated Model Loss Across Epochs\n(Post-Merge Step 1 vs Pre-Merge Final Step)', fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('Federated Epoch (Round)', fontsize=13, fontweight='semibold')
    ax.set_ylabel('Cross-Entropy Loss', fontsize=13, fontweight='semibold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(sorted(df_step1['epoch'].unique()))
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved Step 1 Loss Plot: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "..", "logs")
    results_dir = os.path.join(script_dir, "..", "results")

    print("[*] Loading training logs from:", os.path.abspath(logs_dir))
    df_loss, df_delta = load_data(logs_dir)

    if df_loss.empty:
        print("[!] Error: No data found in epoch_loss_tracking_muon.csv", file=sys.stderr)
        return 1

    # 1. Main detailed dashboard (with LR in Panel C)
    dashboard_path = os.path.join(results_dir, "detailed_federated_training_dashboard.png")
    plot_detailed_dashboard(df_loss, df_delta, dashboard_path)

    # 2. Dual-axis Loss vs. LR trajectory
    dual_axis_path = os.path.join(results_dir, "loss_vs_learning_rate.png")
    plot_loss_and_lr_dual_axis(df_loss, dual_axis_path)

    # 3. Focused Step 1 & Pre/Post merge plot (backwards-compatible filename)
    step1_plot_path = os.path.join(results_dir, "epoch_loss_step1.png")
    plot_focused_step1_loss(df_loss, step1_plot_path)

    # 4. Copy plots to active IDE artifact directory for display
    artifact_dirs = [
        "/home/gauranshi/.gemini/antigravity-ide/brain/82668ce9-bc6b-4626-8b29-7b1587a8203e",
        "/home/gauranshi/.gemini/antigravity-ide/brain/5a10c60a-063c-4cde-b887-1a71550cca63",
    ]

    for ad in artifact_dirs:
        if os.path.exists(ad):
            shutil.copy(dashboard_path, os.path.join(ad, "detailed_federated_training_dashboard.png"))
            shutil.copy(dual_axis_path, os.path.join(ad, "loss_vs_learning_rate.png"))
            shutil.copy(step1_plot_path, os.path.join(ad, "epoch_loss_step1.png"))
            print(f"[+] Copied plots to artifact directory: {ad}")

    print("\n[✓] All plots successfully generated and saved!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())