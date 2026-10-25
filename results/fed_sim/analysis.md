# Federated RL experiments (Qwen2.5-0.5B-Instruct, GSM8K test, 1,319 problems)

Untrained model: 0.3487


## E1: methods, 4 nodes, IID data

| Method | Accuracy (mean ± std over seeds) | Seeds | Consensus gap | MB sent / run |
|---|---|---|---|---|
| Isolated RL | 40.81 ± 0.55 | 5 | 1.899 | 0 |
| Federated RL | 41.58 ± 0.68 | 5 | 0.622 | 147 |
| Centralized RL | 42.64 ± 1.60 | 5 | - | 0 |
| Federated SFT | 31.57 ± 1.15 | 5 | 1.230 | 147 |

Paired comparisons (positive = first method better):

- Federated RL vs Isolated RL: +0.77 pts (95% CI +0.09 to +1.42), McNemar p=0.836, seed t-test p=0.135, 5 seeds
- Federated RL vs Centralized RL: -1.06 pts (95% CI -2.02 to -0.18), McNemar p=0.00317, seed t-test p=0.184, 5 seeds
- Federated RL vs Federated SFT: +10.01 pts (95% CI +8.23 to +11.86), McNemar p=1.57e-13, seed t-test p=8.74e-06, 5 seeds
- Centralized RL vs Isolated RL: +1.83 pts (95% CI +0.93 to +2.70), McNemar p=0.00605, seed t-test p=0.0428, 5 seeds

## E2: 4 nodes, non-IID data

| Method | Accuracy (mean ± std over seeds) | Seeds | Consensus gap | MB sent / run |
|---|---|---|---|---|
| Isolated RL | 42.38 ± 0.81 | 3 | 1.906 | 0 |
| Federated RL | 41.64 ± 1.96 | 3 | 0.701 | 147 |
| Federated SFT | 30.14 ± 2.12 | 3 | 1.522 | 147 |

Paired comparisons (positive = first method better):

- Federated RL vs Isolated RL: -0.74 pts (95% CI -1.52 to +0.12), McNemar p=0.403, seed t-test p=0.466, 3 seeds
- Federated RL vs Federated SFT: +11.50 pts (95% CI +9.86 to +13.28), McNemar p=1.55e-22, seed t-test p=0.000118, 3 seeds
