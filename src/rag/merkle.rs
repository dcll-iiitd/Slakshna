//! Checking what peers send against the Merkle root they committed to.
//!
//! Each node's sidecar fingerprints its whole index (chunks and cluster
//! centroids) as one 32-byte root (`rag/merkle.py`). This module is the Rust
//! side: it signs this node's root, checks peers' signed roots, and checks the
//! proofs that come with every centroid and chunk a peer sends.
//!
//! The exact bytes hashed and signed are defined here and in `rag/merkle.py`.
//! `tests/fixtures/merkle_vectors.json` is checked by both the
//! Python and the Rust tests, so the two languages cannot drift apart.

use crate::history::BoxError;
use iroh::{EndpointId, SecretKey, Signature};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub type Hash = [u8; 32];

const LEAF: u8 = 0x00;
const CLUSTER: u8 = 0x01;
const INNER: u8 = 0x02;
const ROOT_MESSAGE_TAG: &[u8] = b"slakshna-rag-root-v1";

/// One step of a proof: the sibling hash, and whether it sits on the left.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofStep {
    pub h: String,
    pub left: bool,
}

fn parse_hash(hex_str: &str) -> Result<Hash, BoxError> {
    let mut hash = [0u8; 32];
    hex::decode_to_slice(hex_str, &mut hash)
        .map_err(|e| format!("not a 32-byte hex hash ({}): {:.16}", e, hex_str))?;
    Ok(hash)
}

fn u32_len(len: usize) -> [u8; 4] {
    // Chunk texts and ids are far below 4 GiB; the sidecar enforces the same.
    (len as u32).to_be_bytes()
}

/// `H(0x00 ‖ u32be(len(id)) ‖ id ‖ u32be(len(text)) ‖ text)` over UTF-8 bytes.
pub fn leaf_hash(chunk_id: &str, text: &str) -> Hash {
    let mut h = Sha256::new();
    h.update([LEAF]);
    h.update(u32_len(chunk_id.len()));
    h.update(chunk_id.as_bytes());
    h.update(u32_len(text.len()));
    h.update(text.as_bytes());
    h.finalize().into()
}

/// `H(0x02 ‖ left ‖ right)`.
pub fn inner_hash(left: &Hash, right: &Hash) -> Hash {
    let mut h = Sha256::new();
    h.update([INNER]);
    h.update(left);
    h.update(right);
    h.finalize().into()
}

/// `H(0x01 ‖ u32be(cluster_id) ‖ u32be(dim) ‖ centroid_bytes ‖ subtree_root)`,
/// where `centroid_bytes` is `dim` little-endian float32 values.
pub fn cluster_hash(cluster_id: u32, dim: u32, centroid_bytes: &[u8], subtree_root: &Hash) -> Hash {
    let mut h = Sha256::new();
    h.update([CLUSTER]);
    h.update(cluster_id.to_be_bytes());
    h.update(dim.to_be_bytes());
    h.update(centroid_bytes);
    h.update(subtree_root);
    h.finalize().into()
}

/// Recompute a root from a starting hash and the sibling hashes above it.
pub fn fold_proof(start: Hash, proof: &[ProofStep]) -> Result<Hash, BoxError> {
    let mut current = start;
    for step in proof {
        let sibling = parse_hash(&step.h)?;
        current = if step.left {
            inner_hash(&sibling, &current)
        } else {
            inner_hash(&current, &sibling)
        };
    }
    Ok(current)
}

/// Decode `dim` little-endian float32 values from hex (e.g. embeddings from
/// the sidecar's `/embed`).
pub fn decode_f32le(hex_str: &str, dim: u32) -> Result<Vec<f32>, BoxError> {
    let bytes = hex::decode(hex_str).map_err(|e| format!("vector is not hex: {}", e))?;
    if bytes.len() != dim as usize * 4 {
        return Err(format!(
            "vector has {} bytes, expected {}",
            bytes.len(),
            dim as usize * 4
        )
        .into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// What a peer sends to prove one of its clusters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterEntry {
    pub cluster_id: u32,
    pub size: u64,
    /// `dim` little-endian float32 values, as hex.
    pub centroid: String,
    pub subtree_root: String,
    pub cluster_proof: Vec<ProofStep>,
    /// The peer's claim about its best chunk in this cluster (round 1 only).
    /// It is not covered by the proof; round 2 checks it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_chunk_score: Option<f64>,
}

impl ClusterEntry {
    pub fn subtree_root(&self) -> Result<Hash, BoxError> {
        parse_hash(&self.subtree_root)
    }

    /// Check this entry against a peer's `root` and return its centroid.
    ///
    /// Fails if the centroid is not `dim` float32 values or if its proof does
    /// not lead to `root`.
    pub fn verify(&self, root: &Hash, dim: u32) -> Result<Vec<f32>, BoxError> {
        let bytes =
            hex::decode(&self.centroid).map_err(|e| format!("centroid is not hex: {}", e))?;
        if bytes.len() != dim as usize * 4 {
            return Err(format!(
                "centroid has {} bytes, expected {} for dimension {}",
                bytes.len(),
                dim as usize * 4,
                dim
            )
            .into());
        }
        let start = cluster_hash(self.cluster_id, dim, &bytes, &self.subtree_root()?);
        if fold_proof(start, &self.cluster_proof)? != *root {
            return Err(
                format!("cluster {} does not match the peer's root", self.cluster_id).into(),
            );
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }
}

/// The left/right pattern of an honest proof for leaf `index` of an `n`-leaf
/// tree (section 2.2): one entry per step, `true` where the sibling is on the
/// left. The last node of an odd-sized level moves up with no step.
pub fn expected_sides(index: u32, n: u32) -> Vec<bool> {
    let (mut i, mut n) = (index as u64, n as u64);
    let mut sides = Vec::new();
    while n > 1 {
        if i % 2 == 1 {
            sides.push(true);
        } else if i + 1 < n {
            sides.push(false);
        }
        i /= 2;
        n = (n + 1) / 2;
    }
    sides
}

impl ClusterEntry {
    /// `verify`, plus the position check: the
    /// cluster id must be one of the signed `num_clusters`, and the proof must
    /// have exactly the shape an honest proof for that position has.
    ///
    /// Together with "a directory holds ids 0..num_clusters-1", this pins the
    /// tree to exactly `num_clusters` leaves, so a peer cannot leave a cluster
    /// out while its proofs still fold to the root.
    pub fn verify_at(
        &self,
        root: &Hash,
        dim: u32,
        num_clusters: u32,
    ) -> Result<Vec<f32>, BoxError> {
        if self.cluster_id >= num_clusters {
            return Err(format!(
                "cluster {} is outside the {} signed clusters",
                self.cluster_id, num_clusters
            )
            .into());
        }
        let sides: Vec<bool> = self.cluster_proof.iter().map(|step| step.left).collect();
        if sides != expected_sides(self.cluster_id, num_clusters) {
            return Err(format!(
                "cluster {}'s proof does not have the shape of position {} of {}",
                self.cluster_id, self.cluster_id, num_clusters
            )
            .into());
        }
        self.verify(root, dim)
    }
}

/// Tree building and proofs in Rust, for tests only (the sidecar builds the
/// real trees in Python). Used by this module's and `directory.rs`'s tests.
#[cfg(test)]
pub(crate) mod testing {
    use super::{inner_hash, Hash, ProofStep};

    /// Every level of the tree over `nodes`, bottom first (section 2.2).
    pub fn build_levels(nodes: &[Hash]) -> Vec<Vec<Hash>> {
        let mut levels = vec![nodes.to_vec()];
        while levels.last().unwrap().len() > 1 {
            let level = levels.last().unwrap();
            let mut upper: Vec<Hash> = level
                .chunks_exact(2)
                .map(|pair| inner_hash(&pair[0], &pair[1]))
                .collect();
            if level.len() % 2 == 1 {
                upper.push(*level.last().unwrap());
            }
            levels.push(upper);
        }
        levels
    }

    pub fn root(levels: &[Vec<Hash>]) -> Hash {
        levels.last().unwrap().first().copied().unwrap_or([0u8; 32])
    }

    /// The proof for node `index` of the bottom level (section 2.3).
    pub fn make_proof(levels: &[Vec<Hash>], mut index: usize) -> Vec<ProofStep> {
        let mut proof = Vec::new();
        for level in &levels[..levels.len() - 1] {
            if index % 2 == 1 {
                proof.push(ProofStep {
                    h: hex::encode(level[index - 1]),
                    left: true,
                });
            } else if index + 1 < level.len() {
                proof.push(ProofStep {
                    h: hex::encode(level[index + 1]),
                    left: false,
                });
            }
            index /= 2;
        }
        proof
    }
}

/// True if this chunk's proof leads from its id and text to `expected_subtree_root`.
pub fn verify_chunk(
    chunk_id: &str,
    text: &str,
    chunk_proof: &[ProofStep],
    expected_subtree_root: &Hash,
) -> bool {
    matches!(
        fold_proof(leaf_hash(chunk_id, text), chunk_proof),
        Ok(root) if root == *expected_subtree_root
    )
}

/// A node's unsigned commitment, as the sidecar's `GET /commitment` returns it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Commitment {
    pub root: String,
    pub version: u64,
    pub num_chunks: u64,
    pub num_clusters: u32,
    pub dim: u32,
    pub embedding_model: String,
}

/// A commitment signed with the node's Iroh key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedRoot {
    pub root: String,
    pub version: u64,
    pub num_chunks: u64,
    pub num_clusters: u32,
    pub dim: u32,
    pub embedding_model: String,
    /// 64-byte Ed25519 signature, as hex.
    pub signature: String,
}

/// The bytes that are signed:
/// `"slakshna-rag-root-v1" ‖ 0x00 ‖ root ‖ u64be(version) ‖ u64be(num_chunks)
///  ‖ u32be(num_clusters) ‖ u32be(dim) ‖ u32be(len(model)) ‖ model`.
pub fn root_message(
    root: &Hash,
    version: u64,
    num_chunks: u64,
    num_clusters: u32,
    dim: u32,
    embedding_model: &str,
) -> Vec<u8> {
    let mut msg =
        Vec::with_capacity(ROOT_MESSAGE_TAG.len() + 1 + 32 + 8 + 8 + 12 + embedding_model.len());
    msg.extend_from_slice(ROOT_MESSAGE_TAG);
    msg.push(0x00);
    msg.extend_from_slice(root);
    msg.extend_from_slice(&version.to_be_bytes());
    msg.extend_from_slice(&num_chunks.to_be_bytes());
    msg.extend_from_slice(&num_clusters.to_be_bytes());
    msg.extend_from_slice(&dim.to_be_bytes());
    msg.extend_from_slice(&u32_len(embedding_model.len()));
    msg.extend_from_slice(embedding_model.as_bytes());
    msg
}

impl SignedRoot {
    pub fn root_bytes(&self) -> Result<Hash, BoxError> {
        parse_hash(&self.root)
    }

    /// The bytes covered by the signature. Fails only if `root` is not a hash.
    pub fn message_bytes(&self) -> Result<Vec<u8>, BoxError> {
        Ok(root_message(
            &self.root_bytes()?,
            self.version,
            self.num_chunks,
            self.num_clusters,
            self.dim,
            &self.embedding_model,
        ))
    }

    /// Sign this node's commitment with its Iroh transport key.
    pub fn sign(secret: &SecretKey, commitment: &Commitment) -> Result<SignedRoot, BoxError> {
        let mut signed = SignedRoot {
            root: commitment.root.clone(),
            version: commitment.version,
            num_chunks: commitment.num_chunks,
            num_clusters: commitment.num_clusters,
            dim: commitment.dim,
            embedding_model: commitment.embedding_model.clone(),
            signature: String::new(),
        };
        signed.signature = hex::encode(secret.sign(&signed.message_bytes()?).to_bytes());
        Ok(signed)
    }

    /// True if `signer` (a peer's EndpointId, which is its public key) signed
    /// exactly these fields.
    pub fn verify(&self, signer: &EndpointId) -> bool {
        let mut sig = [0u8; Signature::LENGTH];
        if hex::decode_to_slice(&self.signature, &mut sig).is_err() {
            return false;
        }
        match self.message_bytes() {
            Ok(msg) => signer.verify(&msg, &Signature::from_bytes(&sig)).is_ok(),
            Err(_) => false,
        }
    }
}

/// Dot product of two vectors of the same length, summed in f64. With unit
/// vectors this is their cosine similarity.
pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &str = include_str!("../../tests/fixtures/merkle_vectors.json");

    fn fixture() -> Value {
        serde_json::from_str(FIXTURE).unwrap()
    }

    fn hash(v: &Value) -> Hash {
        parse_hash(v.as_str().unwrap()).unwrap()
    }

    fn proof(v: &Value) -> Vec<ProofStep> {
        serde_json::from_value(v.clone()).unwrap()
    }

    /// The tree-building rule of `rag/merkle.py`. Only Python
    /// builds trees for real; this checks Rust reads the rule the same way.
    fn tree_root(nodes: &[Hash]) -> Hash {
        if nodes.is_empty() {
            return [0u8; 32];
        }
        let mut level = nodes.to_vec();
        while level.len() > 1 {
            let mut upper: Vec<Hash> = level
                .chunks_exact(2)
                .map(|p| inner_hash(&p[0], &p[1]))
                .collect();
            if level.len() % 2 == 1 {
                upper.push(*level.last().unwrap());
            }
            level = upper;
        }
        level[0]
    }

    fn commitment(name: &str) -> Value {
        fixture()["commitments"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .unwrap()
            .clone()
    }

    fn test_key(seed: u8) -> SecretKey {
        SecretKey::from_bytes(&[seed; 32])
    }

    fn sample_commitment() -> Commitment {
        let signed = &fixture()["signed_root"];
        serde_json::from_value(signed.clone()).unwrap()
    }

    #[test]
    fn fixture_leaf_and_inner_hashes_match() {
        let f = fixture();
        for leaf in f["leaves"].as_array().unwrap() {
            let computed = leaf_hash(
                leaf["chunk_id"].as_str().unwrap(),
                leaf["text"].as_str().unwrap(),
            );
            assert_eq!(computed, hash(&leaf["hash"]), "{}", leaf["chunk_id"]);
        }
        let inner = &f["inner"];
        assert_eq!(
            inner_hash(&hash(&inner["left"]), &hash(&inner["right"])),
            hash(&inner["hash"])
        );
    }

    #[test]
    fn fixture_trees_build_and_fold_to_the_same_roots() {
        for tree in fixture()["trees"].as_array().unwrap() {
            let leaves: Vec<Hash> = tree["leaves"]
                .as_array()
                .unwrap()
                .iter()
                .map(hash)
                .collect();
            let root = hash(&tree["root"]);
            assert_eq!(tree_root(&leaves), root, "{}", tree["name"]);
            for (leaf, steps) in leaves.iter().zip(tree["proofs"].as_array().unwrap()) {
                assert_eq!(
                    fold_proof(*leaf, &proof(steps)).unwrap(),
                    root,
                    "{}",
                    tree["name"]
                );
            }
        }
    }

    #[test]
    fn fixture_commitments_verify() {
        for case in fixture()["commitments"].as_array().unwrap() {
            let root = hash(&case["root"]);
            let dim = case["dim"].as_u64().unwrap() as u32;
            let entries: Vec<ClusterEntry> =
                serde_json::from_value(case["clusters"].clone()).unwrap();
            let centroids = case["centroids"].as_array().unwrap();
            let cluster_hashes: Vec<Hash> = entries.iter().map(|e| entry_hash(e, dim)).collect();
            assert_eq!(tree_root(&cluster_hashes), root, "{}", case["name"]);

            for (entry, raw) in entries
                .iter()
                .zip(&case["clusters"].as_array().unwrap()[..])
            {
                assert_eq!(entry_hash(entry, dim), hash(&raw["hash"]));
                let values = entry.verify(&root, dim).unwrap();
                let expected: Vec<f32> = centroids[entry.cluster_id as usize]["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect();
                assert_eq!(values, expected);
                assert_eq!(
                    entry.centroid,
                    centroids[entry.cluster_id as usize]["hex"]
                        .as_str()
                        .unwrap()
                );
            }

            for chunk in case["chunks"].as_array().unwrap() {
                let (id, text) = (
                    chunk["chunk_id"].as_str().unwrap(),
                    chunk["text"].as_str().unwrap(),
                );
                assert_eq!(leaf_hash(id, text), hash(&chunk["leaf"]));
                let entry = &entries[chunk["cluster"].as_u64().unwrap() as usize];
                let subtree = entry.subtree_root().unwrap();
                assert!(
                    verify_chunk(id, text, &proof(&chunk["chunk_proof"]), &subtree),
                    "{}",
                    id
                );
            }
        }
    }

    fn entry_hash(entry: &ClusterEntry, dim: u32) -> Hash {
        let bytes = hex::decode(&entry.centroid).unwrap();
        cluster_hash(
            entry.cluster_id,
            dim,
            &bytes,
            &entry.subtree_root().unwrap(),
        )
    }

    #[test]
    fn fixture_edge_cases() {
        // One cluster: the root is that cluster's hash, and its proof is empty.
        let one = commitment("one_cluster");
        let entry: ClusterEntry = serde_json::from_value(one["clusters"][0].clone()).unwrap();
        assert!(entry.cluster_proof.is_empty());
        assert_eq!(entry_hash(&entry, 4), hash(&one["root"]));

        // Empty corpus: no clusters and the all-zero root.
        let empty = commitment("empty");
        assert!(empty["clusters"].as_array().unwrap().is_empty());
        assert_eq!(hash(&empty["root"]), [0u8; 32]);

        // A one-chunk cluster: the subtree root is the chunk's leaf and its proof is empty.
        let three = commitment("three_clusters");
        assert_eq!(
            three["clusters"][1]["subtree_root"],
            three["chunks"][1]["leaf"]
        );
        assert!(three["chunks"][1]["chunk_proof"]
            .as_array()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn fixture_signed_root_message_matches() {
        let signed = &fixture()["signed_root"];
        let msg = root_message(
            &hash(&signed["root"]),
            signed["version"].as_u64().unwrap(),
            signed["num_chunks"].as_u64().unwrap(),
            signed["num_clusters"].as_u64().unwrap() as u32,
            signed["dim"].as_u64().unwrap() as u32,
            signed["embedding_model"].as_str().unwrap(),
        );
        assert_eq!(hex::encode(&msg), signed["message"].as_str().unwrap());
    }

    #[test]
    fn tampered_chunk_or_centroid_fails() {
        let case = commitment("three_clusters");
        let root = hash(&case["root"]);
        let entries: Vec<ClusterEntry> = serde_json::from_value(case["clusters"].clone()).unwrap();
        let chunk = &case["chunks"][0];
        let subtree = entries[0].subtree_root().unwrap();
        let steps = proof(&chunk["chunk_proof"]);
        let text = chunk["text"].as_str().unwrap();
        assert!(verify_chunk("chandrayaan_3.pdf#0", text, &steps, &subtree));
        assert!(!verify_chunk(
            "chandrayaan_3.pdf#0",
            &text.replace("23", "24"),
            &steps,
            &subtree
        ));
        assert!(!verify_chunk("chandrayaan_3.pdf#1", text, &steps, &subtree));

        // One flipped bit in a centroid.
        let mut bytes = hex::decode(&entries[1].centroid).unwrap();
        bytes[5] ^= 0x01;
        let flipped = ClusterEntry {
            centroid: hex::encode(bytes),
            ..entries[1].clone()
        };
        assert!(flipped.verify(&root, 4).is_err());

        // Wrong dimension, wrong root, a different cluster id, and a broken proof step.
        assert!(entries[1].verify(&root, 3).is_err());
        assert!(entries[1].verify(&[7u8; 32], 4).is_err());
        assert!(ClusterEntry {
            cluster_id: 0,
            ..entries[1].clone()
        }
        .verify(&root, 4)
        .is_err());
        let mut broken = entries[0].clone();
        broken.cluster_proof[0].h.truncate(10);
        assert!(broken.verify(&root, 4).is_err());
    }

    #[test]
    fn sign_and_verify_round_trip() {
        let secret = test_key(1);
        let signed = SignedRoot::sign(&secret, &sample_commitment()).unwrap();
        assert_eq!(signed.signature.len(), 128);
        assert!(signed.verify(&secret.public()));

        // The JSON form round-trips and still verifies.
        let json = serde_json::to_string(&signed).unwrap();
        let parsed: SignedRoot = serde_json::from_str(&json).unwrap();
        assert!(parsed.verify(&secret.public()));
    }

    #[test]
    fn signature_from_another_key_or_changed_field_fails() {
        let signed = SignedRoot::sign(&test_key(1), &sample_commitment()).unwrap();
        assert!(!signed.verify(&test_key(2).public()));

        let newer = SignedRoot {
            version: signed.version + 1,
            ..signed.clone()
        };
        assert!(!newer.verify(&test_key(1).public()));
        let other_model = SignedRoot {
            embedding_model: "other".into(),
            ..signed.clone()
        };
        assert!(!other_model.verify(&test_key(1).public()));
        let garbage = SignedRoot {
            signature: "zz".into(),
            ..signed.clone()
        };
        assert!(!garbage.verify(&test_key(1).public()));
        let bad_root = SignedRoot {
            root: "00".into(),
            ..signed
        };
        assert!(!bad_root.verify(&test_key(1).public()));
    }

    fn sides(proof: &[ProofStep]) -> Vec<bool> {
        proof.iter().map(|step| step.left).collect()
    }

    #[test]
    fn every_fixture_proof_has_the_expected_shape() {
        let f = fixture();
        for tree in f["trees"].as_array().unwrap() {
            let proofs = tree["proofs"].as_array().unwrap();
            for (i, steps) in proofs.iter().enumerate() {
                assert_eq!(
                    sides(&proof(steps)),
                    expected_sides(i as u32, proofs.len() as u32)
                );
            }
        }
        for case in f["commitments"].as_array().unwrap() {
            let clusters = case["clusters"].as_array().unwrap();
            for entry in clusters {
                let id = entry["cluster_id"].as_u64().unwrap() as u32;
                let expected = expected_sides(id, clusters.len() as u32);
                assert_eq!(sides(&proof(&entry["cluster_proof"])), expected);
            }
            // A chunk's position is its place among its cluster's chunks.
            let mut seen = std::collections::HashMap::new();
            for chunk in case["chunks"].as_array().unwrap() {
                let cluster = chunk["cluster"].as_u64().unwrap() as usize;
                let position = seen.entry(cluster).or_insert(0u32);
                let size = clusters[cluster]["size"].as_u64().unwrap() as u32;
                assert_eq!(
                    sides(&proof(&chunk["chunk_proof"])),
                    expected_sides(*position, size)
                );
                *position += 1;
            }
        }
    }

    #[test]
    fn rust_proof_builder_matches_the_fixture() {
        for tree in fixture()["trees"].as_array().unwrap() {
            let leaves: Vec<Hash> = tree["leaves"]
                .as_array()
                .unwrap()
                .iter()
                .map(hash)
                .collect();
            let levels = testing::build_levels(&leaves);
            assert_eq!(testing::root(&levels), hash(&tree["root"]));
            for (i, steps) in tree["proofs"].as_array().unwrap().iter().enumerate() {
                assert_eq!(testing::make_proof(&levels, i), proof(steps));
            }
        }
    }

    /// `n` honest cluster entries (4-dim centroids) and their root.
    fn honest_entries(n: u32) -> (Vec<ClusterEntry>, Hash) {
        let centroids: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                [i as f32, 1.0, -0.5, 0.25]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect()
            })
            .collect();
        let subtrees: Vec<Hash> = (0..n)
            .map(|i| Sha256::digest(i.to_be_bytes()).into())
            .collect();
        let leaves: Vec<Hash> = (0..n)
            .map(|i| cluster_hash(i, 4, &centroids[i as usize], &subtrees[i as usize]))
            .collect();
        let levels = testing::build_levels(&leaves);
        let entries = (0..n)
            .map(|i| ClusterEntry {
                cluster_id: i,
                size: 1,
                centroid: hex::encode(&centroids[i as usize]),
                subtree_root: hex::encode(subtrees[i as usize]),
                cluster_proof: testing::make_proof(&levels, i as usize),
                best_chunk_score: None,
            })
            .collect();
        (entries, testing::root(&levels))
    }

    #[test]
    fn honest_proofs_pass_the_position_check() {
        for n in 1..=40 {
            let (entries, root) = honest_entries(n);
            for entry in &entries {
                assert!(
                    entry.verify_at(&root, 4, n).is_ok(),
                    "n={} id={}",
                    n,
                    entry.cluster_id
                );
            }
        }
    }

    #[test]
    fn a_tree_with_one_more_cluster_cannot_pass_as_n() {
        // The plan's test (15.2): an honest tree with n + 1 leaves; its real
        // proofs for ids 0..n-1, checked against a claimed n, must not all pass.
        for n in 1..=40 {
            let (entries, root) = honest_entries(n + 1);
            let failures = entries[..n as usize]
                .iter()
                .filter(|entry| entry.verify_at(&root, 4, n).is_err())
                .count();
            assert!(
                failures >= 1,
                "n={}: every proof passed as {} clusters",
                n,
                n
            );
            // ...and the hidden cluster itself is outside the claimed range.
            assert!(entries[n as usize].verify_at(&root, 4, n).is_err());
        }
    }

    #[test]
    fn position_check_rejects_wrong_ids_and_shapes() {
        let (entries, root) = honest_entries(5);
        assert!(entries[4].verify_at(&root, 4, 5).is_ok());
        // A real proof presented under another id fails (the id is in the hash
        // and the shape differs).
        let moved = ClusterEntry {
            cluster_id: 3,
            ..entries[4].clone()
        };
        assert!(moved.verify_at(&root, 4, 5).is_err());
        // Out of range, and no clusters at all.
        assert!(entries[0].verify_at(&root, 4, 0).is_err());
        let mut flipped = entries[0].clone();
        flipped.cluster_proof[0].left = !flipped.cluster_proof[0].left;
        assert!(flipped.verify_at(&root, 4, 5).is_err());
    }

    #[test]
    fn dot_accumulates_in_f64() {
        assert_eq!(dot(&[0.5, 0.5, 0.5, 0.5], &[0.5, 0.5, 0.5, 0.5]), 1.0);
        assert_eq!(dot(&[1.0, -2.0], &[3.0, 4.0]), -5.0);
    }
}
