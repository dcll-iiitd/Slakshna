//! Simulated misbehaving peers, for experiments only.
//!
//! With `[rag] adversary` set, this node answers peers' requests honestly
//! through its sidecar and then corrupts the answer in one specific way just
//! before sending it. The asking side is unchanged, so this tests whether its
//! checks catch each attack. Never set it outside an experiment.

use super::merkle::{decode_f32le, ClusterEntry, Commitment, SignedRoot};
use super::protocol::{ProvenChunk, RequestBody, ResponseBody};
use super::RagService;
use crate::config::Adversary;
use crate::history::BoxError;
use iroh::SecretKey;
use std::sync::atomic::Ordering;
use tracing::warn;

/// The word `tamper_text` writes into every chunk, so tampered text is easy
/// to look for in contexts and answers.
pub const TAMPER_MARK: &str = "FORGED";
/// What `inflate_score` adds to every score it sends.
pub const INFLATION: f64 = 0.3;

/// Replace the middle word of every chunk with `TAMPER_MARK`.
pub fn tamper_text(chunks: &mut [ProvenChunk]) {
    for chunk in chunks {
        let mut words: Vec<&str> = chunk.text.split(' ').collect();
        let middle = words.len() / 2;
        words[middle] = TAMPER_MARK;
        chunk.text = words.join(" ");
    }
}

/// Add `INFLATION` to every chunk score and every claimed best chunk score.
pub fn inflate_scores(body: &mut ResponseBody) {
    match body {
        ResponseBody::Centroids { centroids, .. } => {
            for entry in centroids {
                if let Some(score) = entry.best_chunk_score.as_mut() {
                    *score += INFLATION;
                }
            }
        }
        ResponseBody::Chunks { chunks, .. } | ResponseBody::Search { chunks, .. } => {
            for chunk in chunks {
                chunk.score += INFLATION;
            }
        }
        _ => {}
    }
}

/// A vector as little-endian float32 hex, the format centroids travel in.
pub fn encode_f32le(vector: &[f32]) -> String {
    hex::encode(
        vector
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
}

/// Replace every centroid with `query_vec`. A centroid equal to the query has
/// similarity 1, the highest possible, so it would always win round 1.
pub fn fake_centroids(entries: &mut [ClusterEntry], query_vec: &[f32]) {
    let fake = encode_f32le(query_vec);
    for entry in entries {
        entry.centroid = fake.clone();
    }
}

/// The normalised sum of all `entries`' centroids: a vector close to every
/// topic the node holds. `fake_centroid` sends it in `directory` answers,
/// which carry no query to copy. `None` if there are no usable centroids.
pub fn mean_direction(entries: &[ClusterEntry]) -> Option<Vec<f32>> {
    let mut sum: Vec<f32> = Vec::new();
    for entry in entries {
        let dim = (entry.centroid.len() / 8) as u32;
        let Ok(centroid) = decode_f32le(&entry.centroid, dim) else {
            continue;
        };
        if sum.is_empty() {
            sum = vec![0.0; centroid.len()];
        }
        if centroid.len() == sum.len() {
            sum.iter_mut().zip(&centroid).for_each(|(s, c)| *s += c);
        }
    }
    let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
    (norm > 0.0).then(|| sum.iter().map(|x| x / norm).collect())
}

/// Remove the entry with the highest cluster id (`hide_cluster`).
pub fn hide_last_cluster(entries: &mut Vec<ClusterEntry>) {
    if let Some(last) = entries.iter().map(|e| e.cluster_id).max() {
        entries.retain(|e| e.cluster_id != last);
    }
}

/// A second, different root with the same version as `real`, properly signed
/// with this node's key: the real root with its first byte flipped.
pub fn equivocating_root(secret: &SecretKey, real: &SignedRoot) -> Result<SignedRoot, BoxError> {
    let mut root = real.root_bytes()?;
    root[0] ^= 0xff;
    let commitment = Commitment {
        root: hex::encode(root),
        version: real.version,
        num_chunks: real.num_chunks,
        num_clusters: real.num_clusters,
        dim: real.dim,
        embedding_model: real.embedding_model.clone(),
    };
    SignedRoot::sign(secret, &commitment)
}

impl RagService {
    /// Corrupt an honest answer to a peer as `[rag] adversary` says.
    pub(super) async fn misbehave(&self, request: &RequestBody, body: &mut ResponseBody) {
        match self.config.adversary {
            Adversary::None => {}
            Adversary::TamperText => {
                if let ResponseBody::Chunks { chunks, .. } | ResponseBody::Search { chunks, .. } =
                    body
                {
                    tamper_text(chunks);
                }
            }
            Adversary::InflateScore => inflate_scores(body),
            Adversary::FakeCentroid => {
                if let ResponseBody::Directory { clusters, .. } = body {
                    if let Some(fake) = mean_direction(clusters) {
                        fake_centroids(clusters, &fake);
                    }
                    return;
                }
                let query = match request {
                    RequestBody::Centroids { query, .. }
                    | RequestBody::Chunks { query, .. }
                    | RequestBody::Search { query, .. } => query,
                    RequestBody::Hello | RequestBody::Directory => return,
                };
                let query_vec = match self.sidecar.embed(&[query.as_str()]).await {
                    Ok(mut vectors) if !vectors.is_empty() => vectors.remove(0),
                    _ => {
                        warn!("😈 fake_centroid: could not embed the query; answering honestly");
                        return;
                    }
                };
                match body {
                    ResponseBody::Centroids { centroids, .. } => {
                        fake_centroids(centroids, &query_vec)
                    }
                    ResponseBody::Chunks { clusters, .. }
                    | ResponseBody::Search { clusters, .. } => fake_centroids(clusters, &query_vec),
                    _ => {}
                }
            }
            Adversary::HideCluster => {
                if let ResponseBody::Directory { clusters, .. } = body {
                    hide_last_cluster(clusters);
                }
            }
            Adversary::Equivocate => {
                if let ResponseBody::Hello { root } = body {
                    if self.hellos_served.fetch_add(1, Ordering::Relaxed) % 2 == 1 {
                        match equivocating_root(&self.secret_key, root) {
                            Ok(fake) => *root = fake,
                            Err(e) => warn!("😈 equivocate: could not sign a second root: {}", e),
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::merkle::{Hash, ProofStep};
    use crate::rag::registry::{RootOutcome, RootRegistry};
    use crate::rag::retrieval::{apply_rescore, check_chunks, overclaimed};
    use crate::rag::Chunk;
    use serde_json::Value;
    use std::collections::HashMap;
    use std::time::Instant;

    const FIXTURE: &str = include_str!("../../tests/fixtures/merkle_vectors.json");

    /// The fixture's three-cluster corpus: chunks with proofs, cluster
    /// entries, subtree roots and the root.
    fn fixture() -> (
        Vec<ProvenChunk>,
        Vec<ClusterEntry>,
        HashMap<u32, Hash>,
        Hash,
    ) {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let case = &fixture["commitments"][0];
        let chunks = case["chunks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| ProvenChunk {
                chunk_id: c["chunk_id"].as_str().unwrap().to_string(),
                source: "fixture".to_string(),
                text: c["text"].as_str().unwrap().to_string(),
                score: 0.5,
                cluster_id: c["cluster"].as_u64().unwrap() as u32,
                chunk_proof: serde_json::from_value::<Vec<ProofStep>>(c["chunk_proof"].clone())
                    .unwrap(),
            })
            .collect();
        let entries: Vec<ClusterEntry> = serde_json::from_value(case["clusters"].clone()).unwrap();
        let roots = entries
            .iter()
            .map(|e| (e.cluster_id, e.subtree_root().unwrap()))
            .collect();
        let mut root = [0u8; 32];
        hex::decode_to_slice(case["root"].as_str().unwrap(), &mut root).unwrap();
        (chunks, entries, roots, root)
    }

    #[test]
    fn tampered_chunks_fail_their_proofs() {
        let (mut chunks, _, roots, _) = fixture();
        tamper_text(&mut chunks);
        assert!(chunks.iter().all(|c| c.text.contains(TAMPER_MARK)));
        let checked = check_chunks(chunks, &roots, 10, false, true, "adversary");
        assert!(checked.kept.is_empty());
        assert_eq!(checked.rejected, 6);
        assert!(checked.bad_proof);
    }

    #[test]
    fn inflated_scores_are_caught_by_rescoring() {
        let (chunks, _, _, _) = fixture();
        let mut body = ResponseBody::Chunks {
            root_version: 1,
            clusters: Vec::new(),
            chunks: chunks[..1].to_vec(),
            retrieve_ms: 0,
            root: None,
        };
        inflate_scores(&mut body);
        let ResponseBody::Chunks { chunks, .. } = body else {
            unreachable!()
        };
        assert!((chunks[0].score - 0.8).abs() < 1e-9);

        // The asker recomputes 0.5 (q · v), flags the difference, and the
        // inflated round-1 claim is more than any delivered chunk reaches.
        let mut received: Vec<Chunk> = vec![Chunk {
            score: chunks[0].score,
            cluster_id: Some(0),
            verification: Some("verified".to_string()),
            ..Default::default()
        }];
        assert!(apply_rescore(
            &mut received,
            &[vec![0.5, 0.0]],
            &[1.0, 0.0],
            0.01
        ));
        assert_eq!(
            overclaimed(&[(0, 0.5 + INFLATION)], &received, 0.01),
            vec![0]
        );
    }

    #[test]
    fn fake_centroids_fail_their_proofs() {
        let (_, mut entries, _, root) = fixture();
        assert!(entries.iter().all(|e| e.verify(&root, 4).is_ok()));
        fake_centroids(&mut entries, &[1.0, 0.0, 0.0, 0.0]);
        assert!(entries.iter().all(|e| e.verify(&root, 4).is_err()));
    }

    #[test]
    fn directory_attacks_change_what_is_sent() {
        let (_, mut entries, _, root) = fixture();
        // hide_cluster: the highest id is gone, the rest is untouched.
        let mut hidden = entries.clone();
        hide_last_cluster(&mut hidden);
        assert_eq!(
            hidden.iter().map(|e| e.cluster_id).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(hidden.iter().all(|e| e.verify_at(&root, 4, 3).is_ok()));

        // fake_centroid on a directory: a unit vector replaces every centroid,
        // and every entry then fails its proof.
        let fake = mean_direction(&entries).unwrap();
        let norm: f32 = fake.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        fake_centroids(&mut entries, &fake);
        assert!(entries.iter().all(|e| e.verify_at(&root, 4, 3).is_err()));
    }

    #[test]
    fn a_second_root_under_one_version_is_caught_as_equivocation() {
        let secret = SecretKey::from_bytes(&[9; 32]);
        let real = SignedRoot::sign(
            &secret,
            &Commitment {
                root: "ab".repeat(32),
                version: 7,
                num_chunks: 10,
                num_clusters: 3,
                dim: 384,
                embedding_model: "m".to_string(),
            },
        )
        .unwrap();
        let fake = equivocating_root(&secret, &real).unwrap();
        assert_eq!(fake.version, real.version);
        assert_ne!(fake.root, real.root);
        assert!(fake.verify(&secret.public()));

        let mut registry = RootRegistry::default();
        let now = Instant::now();
        registry.record_root(secret.public(), real.clone(), now);
        assert_eq!(
            registry.record_root(secret.public(), fake.clone(), now),
            RootOutcome::Equivocation
        );
        assert_eq!(
            registry.get(&secret.public()).unwrap().evidence,
            vec![real, fake]
        );
    }
}
