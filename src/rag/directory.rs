//! Directory mode: a local, verified copy of every peer's centroids.
//!
//! A peer's centroids only change when it ingests documents, so instead of
//! asking every peer for its nearest centroids on every query (round 1 of
//! `two_round`), each node fetches each peer's whole set of cluster entries
//! once per root version and keeps it here. A query then scores the question
//! against all stored centroids locally and only asks the owners of the best
//! clusters for chunks (round 2, unchanged).
//!
//! Every stored entry is checked against the peer's accepted signed root,
//! including its position in the tree (`ClusterEntry::verify_at`), and a
//! directory must list exactly the signed number of clusters, so a peer can't
//! quietly leave a cluster out.

use super::merkle::{dot, ClusterEntry, Hash, SignedRoot};
use super::protocol::{self, ByteCount, RagRequest, RequestBody, ResponseBody};
use super::retrieval::{Candidate, FLAG_BAD_CENTROID_PROOF};
use super::{PeerLink, RagService};
use iroh::EndpointId;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// A directory did not list exactly the clusters `0..num_clusters-1`, once each.
pub const FLAG_INCOMPLETE_DIRECTORY: &str = "incomplete_directory";

/// One verified cluster of a peer.
#[derive(Debug, Clone)]
pub struct DirectoryEntry {
    pub cluster_id: u32,
    pub subtree_root: Hash,
    pub centroid: Vec<f32>,
}

/// What this node holds about one peer's clusters.
#[derive(Debug, Clone)]
pub struct PeerDirectory {
    /// The peer's root version these entries were checked against.
    pub root_version: u64,
    /// The cluster count in that signed root.
    pub num_clusters: u32,
    /// Entries that passed their proofs, by cluster id.
    pub entries: Vec<DirectoryEntry>,
    /// Every signed cluster is present and verified.
    pub complete: bool,
    pub fetched_at: Instant,
    pub flags: Vec<String>,
}

/// The result of checking one fetched directory.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectoryUpdate {
    pub kept: usize,
    pub flags: Vec<String>,
}

/// One peer's line in `/rag/status`.
#[derive(Debug, Clone, Serialize)]
pub struct PeerDirectoryStatus {
    pub endpoint_id: String,
    pub root_version: u64,
    /// Verified clusters held, out of `signed_clusters`.
    pub clusters: usize,
    pub signed_clusters: u32,
    pub complete: bool,
    pub fetched_secs_ago: u64,
    pub flags: Vec<String>,
}

/// The `directory` section of `/rag/status`.
#[derive(Debug, Clone, Serialize)]
pub struct DirectoryStatus {
    pub enabled: bool,
    pub total_centroids: usize,
    /// Memory held by the stored centroids and subtree roots.
    pub approx_bytes: usize,
    pub peers: Vec<PeerDirectoryStatus>,
}

#[derive(Debug, Default)]
pub struct CentroidDirectory {
    peers: HashMap<EndpointId, PeerDirectory>,
}

impl CentroidDirectory {
    pub fn get(&self, peer: &EndpointId) -> Option<&PeerDirectory> {
        self.peers.get(peer)
    }

    /// Check a directory fetched from `peer` against its accepted signed
    /// `root` and store what verifies. The caller
    /// has already made sure the directory was sent for `root.version`.
    pub fn update(
        &mut self,
        peer: EndpointId,
        root: &SignedRoot,
        clusters: Vec<ClusterEntry>,
        now: Instant,
    ) -> DirectoryUpdate {
        let mut flags = Vec::new();
        let n = root.num_clusters;

        // Completeness: exactly the ids 0..n-1, each once.
        let mut seen = HashSet::new();
        let mut incomplete = clusters.len() != n as usize;
        for entry in &clusters {
            if entry.cluster_id >= n || !seen.insert(entry.cluster_id) {
                incomplete = true;
            }
        }
        if incomplete {
            flags.push(FLAG_INCOMPLETE_DIRECTORY.to_string());
        }

        // Each entry: proof and position against the signed root.
        let mut entries = Vec::new();
        let mut bad_proof = false;
        let mut kept_ids = HashSet::new();
        match root.root_bytes() {
            Ok(root_hash) => {
                for entry in &clusters {
                    if kept_ids.contains(&entry.cluster_id) {
                        continue; // a repeated id counts once
                    }
                    match (
                        entry.verify_at(&root_hash, root.dim, n),
                        entry.subtree_root(),
                    ) {
                        (Ok(centroid), Ok(subtree_root)) => {
                            kept_ids.insert(entry.cluster_id);
                            entries.push(DirectoryEntry {
                                cluster_id: entry.cluster_id,
                                subtree_root,
                                centroid,
                            });
                        }
                        _ => bad_proof = true,
                    }
                }
            }
            Err(_) => bad_proof = true,
        }
        if bad_proof {
            flags.push(FLAG_BAD_CENTROID_PROOF.to_string());
        }
        entries.sort_by_key(|e| e.cluster_id);

        let kept = entries.len();
        self.peers.insert(
            peer,
            PeerDirectory {
                root_version: root.version,
                num_clusters: n,
                complete: kept == n as usize && !incomplete,
                entries,
                fetched_at: now,
                flags: flags.clone(),
            },
        );
        DirectoryUpdate { kept, flags }
    }

    /// One round-1-style candidate per stored cluster of `peers`, scored
    /// against the query vector `q`. There is no claimed best chunk score.
    pub fn candidates(&self, q: &[f32], peers: &[EndpointId]) -> Vec<Candidate> {
        let mut out = Vec::new();
        for peer in peers {
            let Some(directory) = self.peers.get(peer) else {
                continue;
            };
            for entry in &directory.entries {
                if entry.centroid.len() != q.len() {
                    continue; // another embedding size: not comparable
                }
                out.push(Candidate {
                    peer: *peer,
                    cluster_id: entry.cluster_id,
                    sim: dot(q, &entry.centroid),
                    claimed_best: None,
                });
            }
        }
        out
    }

    pub fn total_centroids(&self) -> usize {
        self.peers.values().map(|d| d.entries.len()).sum()
    }

    /// Bytes held by stored centroids (4 per value) and subtree roots (32).
    pub fn approx_bytes(&self) -> usize {
        self.peers
            .values()
            .flat_map(|d| &d.entries)
            .map(|e| e.centroid.len() * 4 + 32)
            .sum()
    }

    pub fn status(&self, enabled: bool, now: Instant) -> DirectoryStatus {
        let mut peers: Vec<PeerDirectoryStatus> = self
            .peers
            .iter()
            .map(|(peer, d)| PeerDirectoryStatus {
                endpoint_id: peer.to_string(),
                root_version: d.root_version,
                clusters: d.entries.len(),
                signed_clusters: d.num_clusters,
                complete: d.complete,
                fetched_secs_ago: now.saturating_duration_since(d.fetched_at).as_secs(),
                flags: d.flags.clone(),
            })
            .collect();
        peers.sort_by(|a, b| a.endpoint_id.cmp(&b.endpoint_id));
        DirectoryStatus {
            enabled,
            total_centroids: self.total_centroids(),
            approx_bytes: self.approx_bytes(),
            peers,
        }
    }
}

/// How fetching one peer's directory went.
#[derive(Debug, Clone, Serialize)]
pub struct DirectoryFetch {
    pub endpoint_id: String,
    /// "ok", "timeout", "error" or "no_root".
    pub status: String,
    pub kept: usize,
    pub flags: Vec<String>,
    pub bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl RagService {
    /// The registry's accepted root for `peer`, if any.
    fn accepted_root(&self, peer: &EndpointId) -> Option<SignedRoot> {
        self.registry()
            .get(peer)
            .and_then(|entry| entry.root.clone())
    }

    /// Fetch `peer`'s directory, check it against its accepted signed root
    /// and store what verifies (15.2). If the peer answers under a newer root
    /// version than the registry holds, `hello` is sent once first.
    ///
    /// A failed fetch does not mark the peer as failing in the registry: a
    /// peer with `directory = false` or an older build answers with an error
    /// but is up, and is still asked through round 1.
    pub async fn fetch_directory(
        &self,
        endpoint: &iroh::Endpoint,
        peer: EndpointId,
    ) -> DirectoryFetch {
        let mut result = DirectoryFetch {
            endpoint_id: peer.to_string(),
            status: "ok".to_string(),
            kept: 0,
            flags: Vec::new(),
            bytes: 0,
            error: None,
        };
        let mut bytes = ByteCount::default();
        let request = RagRequest::new(RequestBody::Directory);
        let outcome = tokio::time::timeout(
            Duration::from_millis(self.config.peer_timeout_ms),
            protocol::request_peer(
                endpoint,
                peer,
                &request,
                self.config.max_message_bytes,
                &mut bytes,
            ),
        )
        .await;
        result.bytes = bytes.sent + bytes.received;

        let (root_version, clusters) = match outcome {
            Ok(Ok(response)) => match response.body {
                ResponseBody::Directory {
                    root_version,
                    clusters,
                } => (root_version, clusters),
                _ => {
                    result.status = "error".to_string();
                    result.error = Some("peer answered with another kind".to_string());
                    return result;
                }
            },
            Ok(Err(e)) => {
                result.status = "error".to_string();
                result.error = Some(e.to_string());
                return result;
            }
            Err(_) => {
                result.status = "timeout".to_string();
                return result;
            }
        };

        let mut root = self.accepted_root(&peer);
        if root.as_ref().map(|r| r.version) != Some(root_version) {
            let hello = self.hello_peer(endpoint, peer).await;
            result.bytes += hello.bytes_sent + hello.bytes_received;
            root = self.accepted_root(&peer);
        }
        let Some(root) = root.filter(|r| r.version == root_version) else {
            result.status = "no_root".to_string();
            result.error = Some(format!(
                "no accepted signed root at version {}",
                root_version
            ));
            return result;
        };

        let update = self
            .directory()
            .update(peer, &root, clusters, Instant::now());
        for flag in &update.flags {
            self.registry().add_flag(peer, flag);
        }
        if update.flags.is_empty() {
            info!(
                "🗂️ Stored directory of peer {}: {} clusters at root version {}",
                peer.fmt_short(),
                update.kept,
                root.version
            );
        } else {
            warn!(
                "🚩 Directory of peer {} flagged {:?}; kept {} of {} clusters",
                peer.fmt_short(),
                update.flags,
                update.kept,
                root.num_clusters
            );
        }
        result.kept = update.kept;
        result.flags = update.flags;
        result
    }

    /// Fetch the directory of every target whose accepted root version is not
    /// the version of its stored directory. Run after each background
    /// `hello` round when `[rag] directory = true`.
    pub async fn refresh_directories(&self, link: &PeerLink) -> Vec<DirectoryFetch> {
        let stale: Vec<EndpointId> = {
            let registry = self.registry();
            let directory = self.directory();
            let now = Instant::now();
            let wait = Duration::from_secs(self.config.peer_stale_secs);
            link.targets
                .iter()
                .filter(|peer| registry.should_ask(peer, now, wait))
                .filter(|peer| {
                    let accepted = registry
                        .get(peer)
                        .and_then(|e| e.root.as_ref())
                        .map(|r| r.version);
                    accepted.is_some() && accepted != directory.get(peer).map(|d| d.root_version)
                })
                .copied()
                .collect()
        };
        futures::future::join_all(
            stale
                .iter()
                .map(|peer| self.fetch_directory(&link.endpoint, *peer)),
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SelectBy;
    use crate::rag::merkle::{cluster_hash, testing};
    use crate::rag::retrieval::select_clusters;
    use rand::{Rng, SeedableRng};
    use sha2::{Digest, Sha256};

    /// `n` honest cluster entries with random unit centroids of size `dim`,
    /// and a signed root over them (signature not needed by `update`).
    fn peer_directory(
        rng: &mut impl Rng,
        seed: u64,
        n: u32,
        dim: usize,
    ) -> (Vec<ClusterEntry>, SignedRoot) {
        let centroids: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / norm).collect()
            })
            .collect();
        let bytes: Vec<Vec<u8>> = centroids
            .iter()
            .map(|c| c.iter().flat_map(|x| x.to_le_bytes()).collect())
            .collect();
        let subtrees: Vec<Hash> = (0..n)
            .map(|i| Sha256::digest([seed.to_be_bytes(), (i as u64).to_be_bytes()].concat()).into())
            .collect();
        let leaves: Vec<Hash> = (0..n)
            .map(|i| cluster_hash(i, dim as u32, &bytes[i as usize], &subtrees[i as usize]))
            .collect();
        let levels = testing::build_levels(&leaves);
        let entries = (0..n)
            .map(|i| ClusterEntry {
                cluster_id: i,
                size: 10,
                centroid: hex::encode(&bytes[i as usize]),
                subtree_root: hex::encode(subtrees[i as usize]),
                cluster_proof: testing::make_proof(&levels, i as usize),
                best_chunk_score: None,
            })
            .collect();
        let root = SignedRoot {
            root: hex::encode(testing::root(&levels)),
            version: 1000 + seed,
            num_chunks: 10 * n as u64,
            num_clusters: n,
            dim: dim as u32,
            embedding_model: "m".to_string(),
            signature: String::new(),
        };
        (entries, root)
    }

    fn peer(seed: u8) -> EndpointId {
        iroh::SecretKey::from_bytes(&[seed; 32]).public()
    }

    fn rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(7)
    }

    #[test]
    fn honest_directory_is_stored_complete() {
        let (entries, root) = peer_directory(&mut rng(), 1, 13, 8);
        let mut directory = CentroidDirectory::default();
        let update = directory.update(peer(1), &root, entries, Instant::now());
        assert_eq!(
            update,
            DirectoryUpdate {
                kept: 13,
                flags: vec![]
            }
        );
        let stored = directory.get(&peer(1)).unwrap();
        assert!(stored.complete);
        assert_eq!(stored.root_version, root.version);
        assert_eq!(
            stored
                .entries
                .iter()
                .map(|e| e.cluster_id)
                .collect::<Vec<_>>(),
            (0..13).collect::<Vec<_>>()
        );
        assert_eq!(directory.total_centroids(), 13);
        assert_eq!(directory.approx_bytes(), 13 * (8 * 4 + 32));
    }

    #[test]
    fn a_hidden_cluster_is_flagged_and_the_rest_kept() {
        let (mut entries, root) = peer_directory(&mut rng(), 1, 13, 8);
        entries.pop(); // what `hide_cluster` does
        let mut directory = CentroidDirectory::default();
        let update = directory.update(peer(1), &root, entries, Instant::now());
        assert_eq!(update.kept, 12);
        assert_eq!(update.flags, vec![FLAG_INCOMPLETE_DIRECTORY.to_string()]);
        assert!(!directory.get(&peer(1)).unwrap().complete);
    }

    #[test]
    fn repeated_extra_and_forged_entries_are_caught() {
        let (entries, root) = peer_directory(&mut rng(), 1, 5, 8);
        let mut directory = CentroidDirectory::default();

        // A repeated id: incomplete, and counted once.
        let mut repeated = entries.clone();
        repeated[4] = entries[3].clone();
        let update = directory.update(peer(1), &root, repeated, Instant::now());
        assert_eq!(update.kept, 4);
        assert_eq!(update.flags, vec![FLAG_INCOMPLETE_DIRECTORY.to_string()]);

        // An extra id beyond the signed count.
        let mut extra = entries.clone();
        extra.push(ClusterEntry {
            cluster_id: 5,
            ..entries[0].clone()
        });
        let update = directory.update(peer(1), &root, extra, Instant::now());
        assert_eq!(update.kept, 5);
        assert!(update
            .flags
            .contains(&FLAG_INCOMPLETE_DIRECTORY.to_string()));
        assert!(update.flags.contains(&FLAG_BAD_CENTROID_PROOF.to_string()));

        // A forged centroid: its proof fails, the others are kept.
        let mut forged = entries.clone();
        forged[2].centroid = forged[1].centroid.clone();
        let update = directory.update(peer(1), &root, forged, Instant::now());
        assert_eq!(update.kept, 4);
        assert_eq!(update.flags, vec![FLAG_BAD_CENTROID_PROOF.to_string()]);
    }

    #[test]
    fn a_directory_signed_for_fewer_clusters_than_built_fails() {
        // The peer built 6 clusters but signs 5: listing ids 0..4 can't pass.
        let (entries, mut root) = peer_directory(&mut rng(), 1, 6, 8);
        root.num_clusters = 5;
        let mut directory = CentroidDirectory::default();
        let update = directory.update(peer(1), &root, entries[..5].to_vec(), Instant::now());
        assert!(update.kept < 5);
        assert!(update.flags.contains(&FLAG_BAD_CENTROID_PROOF.to_string()));
    }

    #[test]
    fn candidates_score_every_stored_cluster_of_the_given_peers() {
        let mut r = rng();
        let (a, root_a) = peer_directory(&mut r, 1, 3, 4);
        let (b, root_b) = peer_directory(&mut r, 2, 2, 4);
        let mut directory = CentroidDirectory::default();
        directory.update(peer(1), &root_a, a, Instant::now());
        directory.update(peer(2), &root_b, b, Instant::now());

        let q = directory.get(&peer(2)).unwrap().entries[1].centroid.clone();
        let both = directory.candidates(&q, &[peer(1), peer(2)]);
        assert_eq!(both.len(), 5);
        assert!(both.iter().all(|c| c.claimed_best.is_none()));
        let best = select_clusters(&both, SelectBy::Centroid, 1, 0.0);
        assert_eq!((both[best[0]].peer, both[best[0]].cluster_id), (peer(2), 1));
        assert!((both[best[0]].sim - 1.0).abs() < 1e-5);
        assert_eq!(directory.candidates(&q, &[peer(1)]).len(), 3);
    }

    #[test]
    fn status_reports_every_peer() {
        let (entries, root) = peer_directory(&mut rng(), 1, 4, 8);
        let mut directory = CentroidDirectory::default();
        let fetched = Instant::now();
        directory.update(peer(1), &root, entries, fetched);
        let status = directory.status(true, fetched + Duration::from_secs(12));
        assert!(status.enabled);
        assert_eq!(status.total_centroids, 4);
        assert_eq!(status.peers.len(), 1);
        let line = &status.peers[0];
        assert_eq!(
            (line.clusters, line.signed_clusters, line.complete),
            (4, 4, true)
        );
        assert_eq!(line.fetched_secs_ago, 12);
    }

    /// Plan 15.6, check 2. Run with:
    /// `cargo test --release directory_scale -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn directory_scale() {
        const PEERS: usize = 1000;
        const CLUSTERS: u32 = 13;
        const DIM: usize = 384;
        const QUERIES: usize = 100;
        let mut r = rand::rngs::StdRng::seed_from_u64(42);

        let peers: Vec<EndpointId> = (0..PEERS)
            .map(|i| {
                let mut seed = [0u8; 32];
                seed[..8].copy_from_slice(&(i as u64 + 1).to_be_bytes());
                iroh::SecretKey::from_bytes(&seed).public()
            })
            .collect();
        let fetched: Vec<(Vec<ClusterEntry>, SignedRoot)> = (0..PEERS)
            .map(|i| peer_directory(&mut r, i as u64, CLUSTERS, DIM))
            .collect();

        let mut directory = CentroidDirectory::default();
        let started = Instant::now();
        for (peer, (entries, root)) in peers.iter().zip(fetched) {
            let update = directory.update(*peer, &root, entries, Instant::now());
            assert_eq!(update.kept, CLUSTERS as usize);
            assert!(update.flags.is_empty());
        }
        let check_ms = started.elapsed().as_secs_f64() * 1000.0;

        let mut times = Vec::with_capacity(QUERIES);
        for _ in 0..QUERIES {
            let q: Vec<f32> = (0..DIM).map(|_| r.gen_range(-1.0f32..1.0)).collect();
            let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            let q: Vec<f32> = q.iter().map(|x| x / norm).collect();
            let started = Instant::now();
            let candidates = directory.candidates(&q, &peers);
            let winners = select_clusters(&candidates, SelectBy::Centroid, 5, 0.2);
            times.push(started.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(candidates.len(), PEERS * CLUSTERS as usize);
            assert!(!winners.is_empty());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[QUERIES / 2];
        let p95 = times[QUERIES * 95 / 100];

        // Wire size of one peer's directory, for comparison with max_message_bytes.
        let (one, _) = peer_directory(&mut r, 0, CLUSTERS, DIM);
        let wire = serde_json::to_vec(&one).unwrap().len();
        println!(
            "directory_scale: {} peers x {} clusters = {} centroids (dim {}); \
             checking every entry took {:.0} ms ({:.1} us per entry); \
             scoring + selection per query: median {:.2} ms, p95 {:.2} ms over {} queries; \
             memory ~{:.1} MB of centroids and subtree roots; \
             one directory answer ~{:.0} KB of JSON",
            PEERS,
            CLUSTERS,
            directory.total_centroids(),
            DIM,
            check_ms,
            check_ms * 1000.0 / directory.total_centroids() as f64,
            median,
            p95,
            QUERIES,
            directory.approx_bytes() as f64 / 1e6,
            wire as f64 / 1e3
        );
        assert!(p95 < 10.0, "scoring took {:.2} ms at p95", p95);
    }
}
