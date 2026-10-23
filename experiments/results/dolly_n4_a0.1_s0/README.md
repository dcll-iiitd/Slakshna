# Results: Dolly — Open-Ended Generation Benchmarks (4 Nodes, Task Skew α=0.1, Seed 0)

This document details the **inference-time evaluation** of our decentralized federated serving framework on the open-ended instruction-following dataset **Databricks Dolly-15k** across 4 heterogeneous edge nodes.

- **Figures**: `figures/` (PNG and PDF formats)
- **Numerical Tables**: `tables/` (CSV source data behind every plot)
- **Raw Query Logs**: `raw/predictions.csv` (Detailed per-question model outputs, reference texts, and similarity scores)

---

## 1. Open-Ended Answers: Quality of Each Way of Answering
![generation_quality_by_method](figures/generation_quality_by_method.png)

### How to Read the Plot
- **X-Axis / Groups**: Evaluated under two test environments:
  - **Top (All Task Types)**: Balanced global test set of 200 questions across diverse open-ended tasks (Open QA, Closed QA, Summarization, Classification, Brainstorming, Information Extraction).
  - **Bottom (Node's Own Task Mix)**: Local test set of 100 questions following the client's skewed training distribution.
- **Metric**: **BERTScore-F1** (using `roberta-large`, rescaled with baseline so 0 = unrelated text; higher is better). Compares the semantic meaning of the model's generated answer against human reference completions.
- **Bars / Colors**:
  - **Blue (Federated Inference Options)**: Dynamic query-aware adapter routing, merged adapters (uniform average), random peer adapter, and local-only adapter.
  - **Grey**: Base foundation model without adapters (Qwen2.5-1.5B-Instruct).

### Key Findings
- **Base Foundation Model (0.204)** struggles with specific instruction formatting and task style, scoring lowest across all categories.
- Fine-tuned client adapters improve quality, but using only a client's **own adapter** yields **0.294** on all task types due to task specialization skew.
- **Query-aware routing (0.316)** and **Weight Averaging (0.316)** achieve the highest quality across arbitrary open-ended tasks, effectively leveraging multi-client capabilities.

### 🏆 Winner & Method Ranking (Global Balanced Test Set)
1. **🥇 1st Place (Tie): Query-Aware Routing (`query_aware`) — 0.316 BERTScore-F1**  
   *Why it wins:* Semantically routes open-ended prompts to the peer adapter best suited for that specific instruction style.
2. **🥇 1st Place (Tie): Weight Averaging (`merged_avg`) — 0.316 BERTScore-F1**  
   *Why:* Merging all client LoRAs creates a versatile multi-task instruction-following adapter without per-query routing overhead.
3. **🥈 2nd Place: Random Peer Adapter (`random_adapter`) — 0.294 BERTScore-F1**  
   *Performance:* Matches a local single-adapter baseline on average across heterogeneous tasks.
4. **🥉 3rd Place: Own Adapter Only (`own_local`) — 0.294 BERTScore-F1**  
   *Performance:* Strong on familiar instruction categories (e.g., 0.442 on Node 4's summarization/closed QA), but less effective when given novel prompt formats.
5. **4th Place: Base Model (`base`) — 0.204 BERTScore-F1**  
   *Why:* Lacks task-specific supervised fine-tuning.

**Table Source:** `tables/generation_quality_by_method.csv`

---

## 2. Agreement Across Output Evaluation Metrics
![generation_quality_metrics](figures/generation_quality_metrics.png)

### How to Read the Plot
- **Y-Axis (Rows)**: Evaluated adapters (Base model, Node 1–4 local adapters, and Merged average adapter).
- **Bars / Colors**: Three distinct complementary NLP evaluation metrics comparing generated text to human reference answers:
  - **Blue**: **BERTScore-F1** (token-level semantic similarity via RoBERTa-large).
  - **Orange**: **Embedding Cosine Similarity** (whole-answer topic/intent alignment via `all-MiniLM-L6-v2`).
  - **Green**: **ROUGE-L** (longest common subsequence / word sequence overlap).

### Key Findings
- **High Rank Agreement:** ROUGE-L correlates strongly with BERTScore-F1 across adapters (Spearman rank correlation **0.95**).
- **Embedding Cosine Behavior:** Embedding cosine rates the base model relatively high (**0.71**), close to fine-tuned adapters. This is because cosine similarity measures general topic relevance (whether the answer is in the right domain), whereas BERTScore and ROUGE-L measure actual instruction-following precision and answer correctness.
- **All Adapters Merged (`merged_avg`)** demonstrates consistently high scores across all three metrics simultaneously (BERTScore-F1: 0.316, Embedding Cosine: 0.700, ROUGE-L: 0.315).

### 🏆 Winner & Summary
- **🥇 Best Multi-Task Generation Quality:** **All Adapters Merged (`merged_avg`)** and **Node 4 Specialist Adapter** deliver the strongest instruction-following fidelity.
- **Metric Recommendation:** Use **BERTScore-F1** and **ROUGE-L** for quality claims on text generation, as whole-sentence embedding cosine tends to overestimate base-model performance.

**Table Source:** `tables/generation_quality_adapter_vs_test_set.csv`
