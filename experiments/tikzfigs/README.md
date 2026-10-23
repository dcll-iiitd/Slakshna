# TikZ & PGFPlots Figures for LaTeX / Overleaf

This directory contains standalone, highly customizable **PGFPlots / TikZ** vector figures generated directly from the experimental evaluation data. They are formatted specifically for top-tier computer science conferences (NeurIPS, ICLR, ICML, SIGCOMM, MobiCom).

## Included Figure Files

| File | Description | Recommended Environment |
| :--- | :--- | :--- |
| `fig_accuracy_by_selection_method.tex` | Selection strategy comparison on Global vs Local test sets | `\begin{figure}[t]` (1-col) |
| `fig_accuracy_per_topic.tex` | Heatmap matrix showing per-topic accuracy and blind spots | `\begin{figure}[t]` (1-col) |
| `fig_latency_cold_vs_warm.tex` | Stacked horizontal bar chart of TTFT breakdown | `\begin{figure}[t]` (1-col) |
| `fig_latency_vs_prompt_length.tex` | Dual-panel plot: TTFT & GPU MFU vs. prompt tokens | `\begin{figure*}[t]` (2-col wide) |
| `fig_latency_vs_output_length.tex` | Dual-panel plot: TPOT and E2E latency vs. output tokens | `\begin{figure*}[t]` (2-col wide) |
| `fig_throughput_vs_concurrency.tex` | Dual-panel plot: Throughput and Goodput vs. concurrency | `\begin{figure*}[t]` (2-col wide) |
| `fig_fault_tolerance.tex` | Dual-panel plot: Uptime % and failover latency under peer crashes | `\begin{figure*}[t]` (2-col wide) |
| `fig_fetch_vs_adapter_size.tex` | P2P transfer and GPU attach time vs LoRA rank | `\begin{figure}[t]` (1-col) |
| `fig_communication_bytes.tex` | Log-scale comparison of bytes sent and prompt leakage | `\begin{figure}[t]` (1-col) |
| `fig_dolly_generation_quality.tex` | BERTScore-F1 on open-ended generation (Dolly) | `\begin{figure}[t]` (1-col) |
| `main_test.tex` | Standalone preview document containing all figures | Compile directly in Overleaf |

---

## How to Import in Your Overleaf Paper

### Step 1: Add required packages in your paper's preamble
```latex
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{calc,positioning}

% Define the color palette
\definecolor{oursblue}{RGB}{46,111,183}
\definecolor{vllmred}{RGB}{209,73,91}
\definecolor{basegray}{RGB}{154,164,174}
\definecolor{boundorange}{RGB}{224,164,58}
\definecolor{traingreen}{RGB}{89,161,79}
```

### Step 2: Input the figure wherever desired
```latex
\begin{figure}[t]
\centering
\input{tikzfigs/fig_accuracy_by_selection_method.tex}
\caption{Comparison of adapter selection methods on AG News.}
\label{fig:selection_method}
\end{figure}
```
