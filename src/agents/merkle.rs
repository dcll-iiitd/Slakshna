//! Binary Merkle tree with domain-separated leaves and nodes (A4 commitments).
//!
//! leaf = SHA256(0x00 || data), node = SHA256(0x01 || left || right). An odd
//! node at any level is promoted unchanged (not duplicated), which keeps
//! distinct leaf sets from sharing a root. The leaf count is bound into the
//! committed root so a proof cannot be replayed against a tree of another size.

use sha2::{Digest, Sha256};

pub type Hash = [u8; 32];

fn leaf(data: &[u8]) -> Hash {
    let mut h = Sha256::new();
    h.update([0u8]);
    h.update(data);
    h.finalize().into()
}

fn node(l: &Hash, r: &Hash) -> Hash {
    let mut h = Sha256::new();
    h.update([1u8]);
    h.update(l);
    h.update(r);
    h.finalize().into()
}

fn bind_size(root: &Hash, n: u64) -> Hash {
    let mut h = Sha256::new();
    h.update([2u8]);
    h.update(n.to_le_bytes());
    h.update(root);
    h.finalize().into()
}

#[derive(Debug, Clone)]
pub struct MerkleTree {
    levels: Vec<Vec<Hash>>,
    n: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Proof {
    pub index: u64,
    pub leaf_count: u64,
    /// Sibling hashes bottom-up (hex); levels where the node was promoted are skipped.
    pub siblings: Vec<String>,
}

impl MerkleTree {
    pub fn new<T: AsRef<[u8]>>(items: &[T]) -> Option<Self> {
        if items.is_empty() {
            return None;
        }
        let mut level: Vec<Hash> = items.iter().map(|d| leaf(d.as_ref())).collect();
        let mut levels = vec![level.clone()];
        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|c| if c.len() == 2 { node(&c[0], &c[1]) } else { c[0] })
                .collect();
            levels.push(level.clone());
        }
        Some(MerkleTree { levels, n: items.len() as u64 })
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn len(&self) -> u64 {
        self.n
    }

    /// Hex commitment (tree root bound to the leaf count).
    pub fn root(&self) -> String {
        hex::encode(bind_size(&self.levels.last().unwrap()[0], self.n))
    }

    pub fn prove(&self, index: u64) -> Option<Proof> {
        if index >= self.n {
            return None;
        }
        let mut idx = index as usize;
        let mut siblings = vec![];
        for level in &self.levels[..self.levels.len() - 1] {
            let sib = idx ^ 1;
            if sib < level.len() {
                siblings.push(hex::encode(level[sib]));
            }
            idx /= 2;
        }
        Some(Proof { index, leaf_count: self.n, siblings })
    }
}

/// Check that `data` is leaf `proof.index` of the tree committed to by `root`.
pub fn verify(root_hex: &str, data: &[u8], proof: &Proof) -> bool {
    if proof.index >= proof.leaf_count {
        return false;
    }
    let mut cur = leaf(data);
    let mut idx = proof.index;
    let mut width = proof.leaf_count;
    let mut sibs = proof.siblings.iter();
    while width > 1 {
        let has_sibling = (idx ^ 1) < width;
        if has_sibling {
            let Some(Ok(s)) = sibs.next().map(hex::decode) else { return false };
            let Ok(s): Result<Hash, _> = s.as_slice().try_into() else { return false };
            cur = if idx % 2 == 0 { node(&cur, &s) } else { node(&s, &cur) };
        }
        idx /= 2;
        width = width.div_ceil(2);
    }
    sibs.next().is_none() && hex::encode(bind_size(&cur, proof.leaf_count)) == root_hex
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items(n: usize) -> Vec<Vec<u8>> {
        (0..n).map(|i| format!("row-{i}").into_bytes()).collect()
    }

    #[test]
    fn every_leaf_of_every_size_verifies() {
        for n in 1..=33 {
            let it = items(n);
            let t = MerkleTree::new(&it).unwrap();
            let root = t.root();
            for (i, d) in it.iter().enumerate() {
                let p = t.prove(i as u64).unwrap();
                assert!(verify(&root, d, &p), "n={n} i={i}");
            }
            assert!(t.prove(n as u64).is_none());
        }
    }

    #[test]
    fn wrong_data_index_root_or_size_fails() {
        let it = items(10);
        let t = MerkleTree::new(&it).unwrap();
        let root = t.root();
        let p = t.prove(3).unwrap();
        assert!(!verify(&root, b"row-4", &p));
        let mut moved = p.clone();
        moved.index = 4;
        assert!(!verify(&root, &it[3], &moved));
        let mut resized = p.clone();
        resized.leaf_count = 11;
        assert!(!verify(&root, &it[3], &resized));
        let other = MerkleTree::new(&items(11)).unwrap().root();
        assert!(!verify(&other, &it[3], &p));
        let mut extra = p.clone();
        extra.siblings.push("00".repeat(32));
        assert!(!verify(&root, &it[3], &extra));
    }

    #[test]
    fn leaf_and_node_domains_are_separated() {
        // A two-leaf tree's inner node must not be acceptable as a single leaf.
        let t2 = MerkleTree::new(&[b"a".to_vec(), b"b".to_vec()]).unwrap();
        let inner = [leaf(b"a"), leaf(b"b")].concat();
        let t1 = MerkleTree::new(&[inner]).unwrap();
        assert_ne!(t1.root(), t2.root());
    }

    #[test]
    fn empty_set_has_no_tree() {
        assert!(MerkleTree::new::<Vec<u8>>(&[]).is_none());
    }
}
