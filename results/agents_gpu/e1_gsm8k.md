| adversary | k of 9 | mode | accuracy [95% CI] | tasks | tokens/task | agent-seconds/task | msgs/task | latency ms |
|---|---|---|---|---|---|---|---|---|
| none | 0 | single (largest model) | 79.3% [75.0, 83.7] | 300 | 203 | 0.37 | 2.0 | 371 |
| none | 0 | centralized orchestrator | 55.7% [50.0, 61.0] | 300 | 253 | 0.40 | 2.0 | 410 |
| none | 0 | decentralized, votes only | 61.3% [55.7, 67.0] | 300 | 800 | 1.48 | 24.0 | 698 |
| none | 0 | decentralized + probes | 76.3% [71.7, 81.3] | 300 | 950 | 1.70 | 30.1 | 627 |
| Colluding | 2 | single (largest model) | 78.3% [73.7, 83.0] | 300 | 207 | 0.37 | 2.0 | 377 |
| Colluding | 2 | centralized orchestrator | 47.3% [41.7, 53.0] | 300 | 200 | 0.31 | 2.0 | 317 |
| Colluding | 2 | decentralized, votes only | 0.0% [0.0, 0.0] | 300 | 261 | 0.45 | 24.0 | 468 |
| Colluding | 2 | decentralized + probes | 75.3% [70.7, 80.0] | 300 | 914 | 1.64 | 30.1 | 607 |
| Colluding | 3 | single (largest model) | 77.7% [73.0, 82.3] | 300 | 204 | 0.35 | 2.0 | 354 |
| Colluding | 3 | centralized orchestrator | 35.0% [29.7, 40.7] | 300 | 167 | 0.23 | 2.0 | 233 |
| Colluding | 3 | decentralized, votes only | 0.0% [0.0, 0.0] | 300 | 0 | 0.00 | 24.0 | 12 |
| Colluding | 3 | decentralized + probes | 80.7% [76.3, 85.0] | 300 | 862 | 1.31 | 30.1 | 453 |
| Lying | 2 | single (largest model) | 77.7% [73.0, 82.3] | 300 | 204 | 0.35 | 2.0 | 358 |
| Lying | 2 | centralized orchestrator | 45.3% [39.7, 51.0] | 300 | 198 | 0.31 | 2.0 | 312 |
| Lying | 2 | decentralized, votes only | 74.7% [69.3, 79.7] | 300 | 699 | 1.17 | 24.0 | 547 |
| Lying | 2 | decentralized + probes | 74.7% [69.7, 79.7] | 300 | 921 | 1.60 | 30.1 | 592 |
| Lying | 3 | single (largest model) | 79.0% [74.3, 83.7] | 300 | 204 | 0.36 | 2.0 | 362 |
| Lying | 3 | centralized orchestrator | 35.0% [29.7, 40.7] | 300 | 166 | 0.24 | 2.0 | 245 |
| Lying | 3 | decentralized, votes only | 58.7% [53.0, 64.3] | 300 | 567 | 0.94 | 24.0 | 427 |
| Lying | 3 | decentralized + probes | 81.0% [76.3, 85.3] | 300 | 790 | 1.30 | 30.1 | 422 |

