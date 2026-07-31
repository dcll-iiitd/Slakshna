import pandas as pd

import os

# Use robust paths regardless of where the script is run from
base_dir = os.path.dirname(os.path.abspath(__file__))
trust_csv = os.path.join(base_dir, 'trust_scores_new.csv')
malicious_txt = os.path.join(base_dir, 'malicious_nodes.txt')

df = pd.read_csv(trust_csv)
# Columns: 'timestamp', 'observer_node', 'peer_node', 'weight'

# HARDCODED MALICIOUS SELECTION
malicious_nodes = ["node-2"]

# Clean up duplicate headers caused by concurrent node startups
df = df[df['timestamp'] != 'timestamp']

df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
df = df.dropna(subset=['weight'])
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create a mapping for long node IDs to shorter names
unique_nodes = set(df['observer_node'].unique()).union(set(df['peer_node'].unique()))
id_mapping = {}
counter = 1
for node in unique_nodes:
    if node.startswith('node-'): # Keep 'node-X' names as is
        id_mapping[node] = node
    else:
        id_mapping[node] = f"peer-{counter}"
        counter += 1

# Apply mapping
df['observer_node'] = df['observer_node'].map(id_mapping)
df['peer_node'] = df['peer_node'].map(id_mapping)

malicious_nodes = [id_mapping.get(n, n) for n in malicious_nodes]

# Group by the pair: observer -> peer
grouped = df.groupby(['observer_node', 'peer_node'])

results = []
for (observer, peer), group in grouped:
    group = group.sort_values('timestamp')
    if len(group) > 1: # Need at least 2 points to see a change
        initial = group.iloc[0]['weight']
        final = group.iloc[-1]['weight']
        diff = final - initial
        is_malicious = peer in malicious_nodes
        results.append({
            'observer': observer, 
            'peer': peer, 
            'initial': initial, 
            'final': final, 
            'diff': diff, 
            'malicious_peer': is_malicious
        })

res_df = pd.DataFrame(results)

print("=== PAIRWISE TRUST SCORE ANALYSIS ===")
print(f"Total connections (pairs) analyzed: {len(res_df)}")

# 1. Observers trusting MALICIOUS nodes
mal_pairs = res_df[res_df['malicious_peer'] == True]
print(f"\n--- Observers rating MALICIOUS peers (Total Pairs: {len(mal_pairs)}) ---")
decreased_mal = len(mal_pairs[mal_pairs['diff'] < -0.01])
print(f"Observers who decreased trust (< -0.01): {decreased_mal}/{len(mal_pairs)}")
print(f"Average Initial Trust: {mal_pairs['initial'].mean():.4f}")
print(f"Average Final Trust:   {mal_pairs['final'].mean():.4f}")
print(f"Average Change:        {mal_pairs['diff'].mean():+.4f}")

print("\nDetails of trust drops on malicious nodes:")
for _, row in mal_pairs.iterrows():
    print(f"  {row['observer']} -> {row['peer']} | Initial: {row['initial']:.4f} | Final: {row['final']:.4f} | Change: {row['diff']:+.4f}")

# 2. Observers trusting HONEST nodes
honest_pairs = res_df[res_df['malicious_peer'] == False]
print(f"\n--- Observers rating HONEST peers (Total Pairs: {len(honest_pairs)}) ---")
increased_or_same = len(honest_pairs[honest_pairs['diff'] >= -0.01])
decreased_honest = len(honest_pairs[honest_pairs['diff'] < -0.01])

print(f"Observers who maintained or increased trust (>= -0.01): {increased_or_same}/{len(honest_pairs)}")
print(f"Observers who decreased trust (< -0.01): {decreased_honest}/{len(honest_pairs)}")
print(f"Average Initial Trust: {honest_pairs['initial'].mean():.4f}")
print(f"Average Final Trust:   {honest_pairs['final'].mean():.4f}")
print(f"Average Change:        {honest_pairs['diff'].mean():+.4f}")

