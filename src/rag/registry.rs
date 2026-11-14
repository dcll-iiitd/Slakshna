//! What this node knows about each peer's signed Merkle root, and which peers
//! are worth asking.
//!
//! A background task fetches every peer's root (`hello`) once a minute. The
//! registry keeps the latest accepted root per peer, flags peers whose roots
//! break the rules, and remembers when each peer last answered, so queries can
//! skip peers that are down instead of waiting out their timeout.
//!
//! Everything lives in memory: after a restart every peer is trusted again on
//! first contact (section 11).

use super::merkle::SignedRoot;
use iroh::EndpointId;
use serde::Serialize;
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub const FLAG_BAD_SIGNATURE: &str = "bad_signature";
pub const FLAG_ROLLBACK: &str = "rollback";
pub const FLAG_EQUIVOCATION: &str = "equivocation";

/// What this node knows about one peer.
#[derive(Debug, Clone, Default)]
pub struct PeerRoot {
    /// Latest accepted signed root.
    pub root: Option<SignedRoot>,
    /// When this exact root was first accepted.
    pub first_seen: Option<Instant>,
    /// Last time the peer answered a `hello` or a query.
    pub last_ok: Option<Instant>,
    /// Last time it did not (timeout, refused connection, error reply).
    pub last_fail: Option<Instant>,
    pub last_error: Option<String>,
    /// Rule breaks seen from this peer, each listed once.
    pub flags: Vec<String>,
    /// Conflicting signed roots: two of them with one version prove the peer
    /// signed two different corpora under the same version.
    pub evidence: Vec<SignedRoot>,
}

impl PeerRoot {
    fn flag(&mut self, flag: &str) {
        if !self.flags.iter().any(|f| f == flag) {
            self.flags.push(flag.to_string());
        }
    }

    fn keep_evidence(&mut self, root: &SignedRoot) {
        if !self.evidence.contains(root) {
            self.evidence.push(root.clone());
        }
    }

    /// True if the most recent contact with this peer failed.
    fn last_contact_failed(&self) -> bool {
        match (self.last_ok, self.last_fail) {
            (Some(ok), Some(fail)) => fail > ok,
            (None, Some(_)) => true,
            _ => false,
        }
    }

    pub fn has_flag(&self, flag: &str) -> bool {
        self.flags.iter().any(|f| f == flag)
    }
}

/// What happened to a root fetched from a peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RootOutcome {
    /// First root from this peer, or a higher version.
    Accepted,
    /// The same root as before.
    Unchanged,
    /// Not signed by the peer's key; the old root is kept.
    BadSignature,
    /// A lower version than one already seen; the old root is kept.
    Rollback,
    /// Same version, different content; both are kept as evidence.
    Equivocation,
}

/// One peer's line in `GET /rag/status`.
#[derive(Debug, Clone, Serialize)]
pub struct PeerRootStatus {
    pub endpoint_id: String,
    /// "untried", "ok" (its latest contact succeeded) or "failing".
    pub status: &'static str,
    /// Whether a query sent now would include this peer.
    pub asked_in_queries: bool,
    pub version: Option<u64>,
    /// First 16 hex characters of the accepted root.
    pub root: Option<String>,
    pub first_seen_secs_ago: Option<u64>,
    pub last_ok_secs_ago: Option<u64>,
    pub last_fail_secs_ago: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    pub flags: Vec<String>,
    pub evidence: usize,
}

#[derive(Debug, Default)]
pub struct RootRegistry {
    peers: HashMap<EndpointId, PeerRoot>,
}

impl RootRegistry {
    pub fn get(&self, peer: &EndpointId) -> Option<&PeerRoot> {
        self.peers.get(peer)
    }

    /// Record a root fetched from `peer` and decide whether to accept it.
    ///
    /// Any answer, even one with a bad root, shows the peer is reachable, so
    /// it counts as contact for skipping stale peers. Trust is handled by the
    /// flags, not by skipping.
    pub fn record_root(&mut self, peer: EndpointId, root: SignedRoot, now: Instant) -> RootOutcome {
        let entry = self.peers.entry(peer).or_default();
        entry.last_ok = Some(now);

        if !root.verify(&peer) {
            entry.flag(FLAG_BAD_SIGNATURE);
            return RootOutcome::BadSignature;
        }
        let outcome = match &entry.root {
            None => RootOutcome::Accepted,
            Some(current) if root.version > current.version => RootOutcome::Accepted,
            Some(current) if root.version < current.version => RootOutcome::Rollback,
            // Ed25519 signatures are deterministic, so equal fields mean an
            // equal root; compare the fields, not the signature.
            Some(current) if current.message_bytes().ok() == root.message_bytes().ok() => {
                RootOutcome::Unchanged
            }
            Some(_) => RootOutcome::Equivocation,
        };
        match outcome {
            RootOutcome::Accepted => {
                entry.root = Some(root);
                entry.first_seen = Some(now);
            }
            RootOutcome::Rollback => entry.flag(FLAG_ROLLBACK),
            RootOutcome::Equivocation => {
                let current = entry
                    .root
                    .clone()
                    .expect("equivocation needs a current root");
                entry.keep_evidence(&current);
                entry.keep_evidence(&root);
                entry.flag(FLAG_EQUIVOCATION);
            }
            RootOutcome::Unchanged | RootOutcome::BadSignature => {}
        }
        outcome
    }

    /// Note a rule break seen while querying `peer` (e.g. "score_mismatch").
    pub fn add_flag(&mut self, peer: EndpointId, flag: &str) {
        self.peers.entry(peer).or_default().flag(flag);
    }

    /// The peer answered a query.
    pub fn record_ok(&mut self, peer: EndpointId, now: Instant) {
        self.peers.entry(peer).or_default().last_ok = Some(now);
    }

    /// The peer did not answer, or answered with an error.
    pub fn record_failure(&mut self, peer: EndpointId, error: String, now: Instant) {
        let entry = self.peers.entry(peer).or_default();
        entry.last_fail = Some(now);
        entry.last_error = Some(error);
    }

    /// Whether a query should go to `peer`: it has never been tried, or it
    /// answered within `stale` and its latest contact did not fail.
    ///
    /// A peer is skipped from its first failure on, so a peer that has just
    /// died costs at most one timeout. The background refresh keeps asking
    /// every peer, and its next answer brings the peer back.
    pub fn should_ask(&self, peer: &EndpointId, now: Instant, stale: Duration) -> bool {
        match self.peers.get(peer) {
            None => true,
            Some(entry) => match entry.last_ok {
                Some(last_ok) => {
                    now.saturating_duration_since(last_ok) <= stale && !entry.last_contact_failed()
                }
                None => entry.last_fail.is_none(),
            },
        }
    }

    /// One line per peer in `targets` or in the registry, sorted by id.
    pub fn status(
        &self,
        targets: &[EndpointId],
        now: Instant,
        stale: Duration,
    ) -> Vec<PeerRootStatus> {
        let mut ids: Vec<EndpointId> = targets.to_vec();
        for id in self.peers.keys() {
            if !ids.contains(id) {
                ids.push(*id);
            }
        }
        ids.sort_by_key(|id| id.to_string());

        let ago = |t: Option<Instant>| t.map(|t| now.saturating_duration_since(t).as_secs());
        ids.iter()
            .map(|id| {
                let entry = self.peers.get(id).cloned().unwrap_or_default();
                let status = match (entry.last_ok, entry.last_fail) {
                    (None, None) => "untried",
                    _ if entry.last_contact_failed() => "failing",
                    _ => "ok",
                };
                PeerRootStatus {
                    endpoint_id: id.to_string(),
                    status,
                    // Queries also refuse equivocating peers (retrieval.rs).
                    asked_in_queries: self.should_ask(id, now, stale)
                        && !entry.has_flag(FLAG_EQUIVOCATION),
                    version: entry.root.as_ref().map(|r| r.version),
                    root: entry
                        .root
                        .as_ref()
                        .map(|r| r.root.chars().take(16).collect()),
                    first_seen_secs_ago: ago(entry.first_seen),
                    last_ok_secs_ago: ago(entry.last_ok),
                    last_fail_secs_ago: ago(entry.last_fail),
                    last_error: entry.last_error.clone(),
                    flags: entry.flags.clone(),
                    evidence: entry.evidence.len(),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::merkle::Commitment;
    use iroh::SecretKey;

    fn key(seed: u8) -> SecretKey {
        SecretKey::from_bytes(&[seed; 32])
    }

    fn signed(secret: &SecretKey, root_byte: u8, version: u64) -> SignedRoot {
        let commitment = Commitment {
            root: format!("{:02x}", root_byte).repeat(32),
            version,
            num_chunks: 10,
            num_clusters: 3,
            dim: 384,
            embedding_model: "m".to_string(),
        };
        SignedRoot::sign(secret, &commitment).unwrap()
    }

    #[test]
    fn first_root_and_higher_versions_are_accepted() {
        let peer = key(1);
        let id = peer.public();
        let mut registry = RootRegistry::default();
        let t0 = Instant::now();

        assert_eq!(
            registry.record_root(id, signed(&peer, 1, 10), t0),
            RootOutcome::Accepted
        );
        assert_eq!(
            registry.record_root(id, signed(&peer, 1, 10), t0),
            RootOutcome::Unchanged
        );
        let later = t0 + Duration::from_secs(5);
        assert_eq!(
            registry.record_root(id, signed(&peer, 2, 11), later),
            RootOutcome::Accepted
        );

        let entry = registry.get(&id).unwrap();
        assert_eq!(entry.root.as_ref().unwrap().version, 11);
        assert_eq!(entry.first_seen, Some(later));
        assert!(entry.flags.is_empty());
    }

    #[test]
    fn bad_signature_and_rollback_keep_the_old_root() {
        let peer = key(1);
        let id = peer.public();
        let mut registry = RootRegistry::default();
        let now = Instant::now();
        registry.record_root(id, signed(&peer, 1, 10), now);

        // Signed by someone else, but claimed for this peer.
        assert_eq!(
            registry.record_root(id, signed(&key(2), 9, 99), now),
            RootOutcome::BadSignature
        );
        assert_eq!(
            registry.record_root(id, signed(&peer, 3, 9), now),
            RootOutcome::Rollback
        );

        let entry = registry.get(&id).unwrap();
        assert_eq!(entry.root.as_ref().unwrap().version, 10);
        assert!(entry.has_flag(FLAG_BAD_SIGNATURE) && entry.has_flag(FLAG_ROLLBACK));
        assert!(entry.evidence.is_empty());
    }

    #[test]
    fn same_version_with_other_content_is_equivocation_with_evidence() {
        let peer = key(1);
        let id = peer.public();
        let mut registry = RootRegistry::default();
        let now = Instant::now();
        let first = signed(&peer, 1, 10);
        let second = signed(&peer, 2, 10);
        registry.record_root(id, first.clone(), now);

        assert_eq!(
            registry.record_root(id, second.clone(), now),
            RootOutcome::Equivocation
        );
        assert_eq!(
            registry.record_root(id, second.clone(), now),
            RootOutcome::Equivocation
        );

        let entry = registry.get(&id).unwrap();
        assert_eq!(entry.root.as_ref(), Some(&first));
        assert_eq!(entry.flags, vec![FLAG_EQUIVOCATION.to_string()]);
        // Both are signed by the peer: a third party can check them too.
        assert_eq!(entry.evidence, vec![first.clone(), second.clone()]);
        assert!(entry.evidence.iter().all(|r| r.verify(&id)));
    }

    #[test]
    fn untried_and_recent_peers_are_asked_failing_ones_are_not() {
        let mut registry = RootRegistry::default();
        let stale = Duration::from_secs(180);
        let t0 = Instant::now();
        let (alive, dead, untried) = (key(1).public(), key(2).public(), key(3).public());

        registry.record_ok(alive, t0);
        registry.record_failure(dead, "timeout".into(), t0);
        assert!(registry.should_ask(&alive, t0, stale));
        assert!(!registry.should_ask(&dead, t0, stale));
        assert!(registry.should_ask(&untried, t0, stale));

        // Silent for longer than `stale`: skipped until it answers again.
        let much_later = t0 + Duration::from_secs(181);
        assert!(!registry.should_ask(&alive, much_later, stale));
        registry.record_ok(alive, much_later);
        assert!(registry.should_ask(&alive, much_later, stale));

        // A failure after recent contact skips the peer at once...
        let failed = much_later + Duration::from_secs(1);
        registry.record_failure(alive, "timeout".into(), failed);
        assert!(!registry.should_ask(&alive, failed, stale));
        // ...until it answers again, e.g. to the next background refresh.
        let refreshed = failed + Duration::from_secs(60);
        registry.record_ok(alive, refreshed);
        assert!(registry.should_ask(&alive, refreshed, stale));
    }

    #[test]
    fn status_lists_targets_and_known_peers() {
        let peer = key(1);
        let mut registry = RootRegistry::default();
        let t0 = Instant::now();
        registry.record_root(peer.public(), signed(&peer, 0xab, 7), t0);
        registry.record_failure(key(2).public(), "timed out".into(), t0);
        let untried = key(3).public();

        let now = t0 + Duration::from_secs(30);
        let lines = registry.status(&[untried], now, Duration::from_secs(180));
        assert_eq!(lines.len(), 3);
        let line = |id: EndpointId| {
            lines
                .iter()
                .find(|l| l.endpoint_id == id.to_string())
                .unwrap()
        };

        let ok = line(peer.public());
        assert_eq!(
            (ok.status, ok.asked_in_queries, ok.version),
            ("ok", true, Some(7))
        );
        assert_eq!(ok.root.as_deref(), Some("abababababababab"));
        assert_eq!(
            (ok.first_seen_secs_ago, ok.last_ok_secs_ago),
            (Some(30), Some(30))
        );

        let failing = line(key(2).public());
        assert_eq!(
            (failing.status, failing.asked_in_queries),
            ("failing", false)
        );
        assert_eq!(failing.last_error.as_deref(), Some("timed out"));

        assert_eq!(line(untried).status, "untried");

        // An equivocating peer is shown as not asked, as queries refuse it.
        registry.add_flag(peer.public(), FLAG_EQUIVOCATION);
        let lines = registry.status(&[], now, Duration::from_secs(180));
        let refused = lines
            .iter()
            .find(|l| l.endpoint_id == peer.public().to_string())
            .unwrap();
        assert!(!refused.asked_in_queries);
        assert!(line(untried).asked_in_queries);
    }
}
