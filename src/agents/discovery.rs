//! Agent discovery (M1).
//!
//! An agent publishes an `a2a.card` provenance record (card hash, version, or a
//! revocation) which travels over gossip like every other record; the card
//! document itself is fetched from the agent over `slakshna/agent/1` or the
//! HTTPS gateway. The directory only lists an agent when
//!   * the record is validly signed by its node,
//!   * the node delegated that agent key with scope `a2a.card` (and it is live),
//!   * the fetched card hashes to the record's `card_hash`, is signed by the
//!     agent key, and has not been revoked.
//!
//! Capability search then runs over verified cards.
//!
//! Scope: card *announcements* ride the federation gossip; node addresses are
//! resolved by the existing mDNS/DHT address lookup. Cards are not themselves
//! published into the DHT.

use crate::agents::a2a::AgentCard;
use crate::agents::delegation::DelegationRegistry;
use crate::agents::records::{A2aCard, KIND_A2A_CARD};
use crate::history::{RecordKind, UpdateRecord};
use std::collections::{BTreeMap, BTreeSet};

pub const SCOPE_CARD: &str = "a2a.card";

#[derive(Debug, Clone)]
struct Announced {
    record_hash: String,
    version: u32,
    card_hash: String,
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub node_id: String,
    pub agent_key: String,
    pub card: AgentCard,
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub version: u32,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CardRejected {
    #[error("no announcement for this agent")]
    NotAnnounced,
    #[error("agent card was revoked")]
    Revoked,
    #[error("card does not match the announced hash")]
    HashMismatch,
    #[error("card is not signed by the announcing agent key")]
    BadSignature,
    #[error("agent key is not delegated for cards by its node: {0}")]
    NotDelegated(String),
    #[error("invalid card: {0}")]
    Invalid(String),
}

#[derive(Debug, Default)]
pub struct Directory {
    delegations: DelegationRegistry,
    announced: BTreeMap<(String, String), Announced>, // (node, agent key) -> latest
    revoked: BTreeSet<(String, String)>,
    entries: BTreeMap<(String, String), Entry>,
}

impl Directory {
    pub fn from_records<'a>(records: impl IntoIterator<Item = &'a UpdateRecord> + Clone) -> Self {
        let mut d = Directory { delegations: DelegationRegistry::from_records(records.clone()), ..Default::default() };
        for r in records {
            d.ingest(r);
        }
        d
    }

    /// Feed a record; key delegations update the registry, card records the announcements.
    pub fn ingest(&mut self, r: &UpdateRecord) {
        self.delegations.add(r);
        let RecordKind::Provenance { kind, body } = &r.kind else { return };
        if kind != KIND_A2A_CARD || r.verify().is_err() {
            return;
        }
        let Ok(c) = serde_json::from_value::<A2aCard>(body.clone()) else { return };
        let key = (r.node_id.clone(), c.agent_public_key.clone());
        if c.revoked {
            self.revoked.insert(key.clone());
            self.entries.remove(&key);
            return;
        }
        let newer = self.announced.get(&key).map_or(true, |a| (c.version, &r.hash) > (a.version, &a.record_hash));
        if newer {
            self.announced.insert(key.clone(), Announced { record_hash: r.hash.clone(), version: c.version, card_hash: c.card_hash });
            // a newer announcement invalidates the previously attached document
            self.entries.remove(&key);
        }
    }

    /// Attach a fetched card document, verifying it against the announcement.
    pub fn attach_card(&mut self, node_id: &str, card: AgentCard, now: i64) -> Result<(), CardRejected> {
        card.validate().map_err(CardRejected::Invalid)?;
        let agent_key = card.verified_key().ok_or(CardRejected::BadSignature)?;
        let key = (node_id.to_string(), agent_key.clone());
        if self.revoked.contains(&key) {
            return Err(CardRejected::Revoked);
        }
        let ann = self.announced.get(&key).ok_or(CardRejected::NotAnnounced)?;
        if card.content_hash() != ann.card_hash {
            return Err(CardRejected::HashMismatch);
        }
        self.delegations.check(node_id, &agent_key, SCOPE_CARD, now).map_err(|e| CardRejected::NotDelegated(e.to_string()))?;
        let version = ann.version;
        self.entries.insert(key, Entry { node_id: node_id.into(), agent_key, card, version });
        Ok(())
    }

    /// Entries still valid at `now` (delegation may have expired or been revoked since).
    pub fn entries(&self, now: i64) -> Vec<&Entry> {
        self.entries
            .iter()
            .filter(|((n, k), _)| !self.revoked.contains(&(n.clone(), k.clone())) && self.delegations.check(n, k, SCOPE_CARD, now).is_ok())
            .map(|(_, e)| e)
            .collect()
    }

    /// Capability search: match a skill id or tag (case-insensitive).
    pub fn search(&self, query: &str, now: i64) -> Vec<&Entry> {
        let q = query.to_lowercase();
        self.entries(now)
            .into_iter()
            .filter(|e| e.card.skills.iter().any(|s| s.id.to_lowercase() == q || s.tags.iter().any(|t| t.to_lowercase() == q)))
            .collect()
    }

    /// QUIC endpoint id of a verified agent's node, from its card.
    pub fn endpoint_of(&self, node_id: &str, now: i64) -> Option<String> {
        self.entries(now).into_iter().find(|e| e.node_id == node_id).and_then(|e| e.card.slakshna_endpoint_id.clone())
    }

    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn is_revoked(&self, node: &str, agent_key: &str) -> bool {
        self.revoked.contains(&(node.to_string(), agent_key.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::a2a::*;
    use crate::agents::delegation::delegation_record;
    use crate::agents::graph::make_record;
    use crate::identity::Keypair;

    struct Setup { node: Keypair, nid: String, agent: Keypair, log: Vec<UpdateRecord> }

    impl Setup {
        fn new() -> Self {
            let node = Keypair::generate();
            let nid = node.node_id().to_string();
            let agent = Keypair::generate();
            let mut d = delegation_record(&nid, &"0".repeat(64), &agent.public_key_hex(), &[SCOPE_CARD], 0, 10_000, false);
            d.sign(&node);
            Setup { node, nid, agent, log: vec![d] }
        }

        fn card(&self, skills: &[(&str, &[&str])]) -> AgentCard {
            let mut c = AgentCard {
                name: "agent".into(), description: "d".into(), version: "1.0".into(),
                supported_interfaces: vec![AgentInterface { url: "https://h/a2a".into(), protocol_binding: "JSONRPC".into(), protocol_version: "1.0".into(), tenant: None }],
                capabilities: AgentCapabilities::default(), security_schemes: None, security_requirements: vec![],
                default_input_modes: vec!["text/plain".into()], default_output_modes: vec!["text/plain".into()],
                skills: skills.iter().map(|(id, tags)| AgentSkill { id: id.to_string(), name: id.to_string(), description: "d".into(), tags: tags.iter().map(|t| t.to_string()).collect(), examples: vec![], input_modes: vec![], output_modes: vec![] }).collect(),
                signatures: vec![], slakshna_node_id: Some(self.nid.clone()), slakshna_endpoint_id: None, slakshna_prices: Default::default(),
            };
            c.sign(&self.agent);
            c
        }

        fn announce(&mut self, card: &AgentCard, version: u32, revoked: bool) {
            let prev = self.log.last().unwrap().hash.clone();
            let mut r = make_record(&self.nid, &prev, KIND_A2A_CARD, &A2aCard { agent_public_key: self.agent.public_key_hex(), card_hash: card.content_hash(), version, revoked });
            r.sign(&self.node);
            self.log.push(r);
        }
    }

    #[test]
    fn announced_signed_delegated_card_is_discoverable_by_skill_and_tag() {
        let mut s = Setup::new();
        let card = s.card(&[("evaluate", &["private-data", "Medical"])]);
        s.announce(&card, 1, false);
        let mut d = Directory::from_records(&s.log);
        d.attach_card(&s.nid, card, 100).unwrap();
        assert_eq!(d.search("evaluate", 100).len(), 1);
        assert_eq!(d.search("medical", 100).len(), 1, "tags match case-insensitively");
        assert!(d.search("translate", 100).is_empty());
    }

    #[test]
    fn card_must_match_announcement_and_be_signed_by_the_delegated_key() {
        let mut s = Setup::new();
        let card = s.card(&[("evaluate", &[])]);
        // never announced
        let mut d = Directory::from_records(&s.log);
        assert_eq!(d.attach_card(&s.nid, card.clone(), 1), Err(CardRejected::NotAnnounced));
        s.announce(&card, 1, false);
        let mut d = Directory::from_records(&s.log);
        // tampered document
        let mut tampered = card.clone();
        tampered.description = "malicious".into();
        assert_eq!(d.attach_card(&s.nid, tampered, 1), Err(CardRejected::BadSignature));
        // re-signed by an undelegated key: announcement is for another key
        let rogue = Keypair::generate();
        let mut other = card.clone();
        other.sign(&rogue);
        assert_eq!(d.attach_card(&s.nid, other, 1), Err(CardRejected::NotAnnounced));
        // a different (validly signed) document than the announced one
        let mut diff = s.card(&[("evaluate", &[]), ("extra", &[])]);
        diff.sign(&s.agent);
        assert_eq!(d.attach_card(&s.nid, diff, 1), Err(CardRejected::HashMismatch));
        // claiming another node's id does not work either
        assert_eq!(d.attach_card("other-node", card, 1), Err(CardRejected::NotAnnounced));
    }

    #[test]
    fn undelegated_agent_keys_are_not_listed() {
        let mut s = Setup::new();
        s.log.clear(); // no delegation record at all
        let card = s.card(&[("evaluate", &[])]);
        s.log.push({ let mut r = make_record(&s.nid, &"0".repeat(64), KIND_A2A_CARD, &A2aCard { agent_public_key: s.agent.public_key_hex(), card_hash: card.content_hash(), version: 1, revoked: false }); r.sign(&s.node); r });
        let mut d = Directory::from_records(&s.log);
        assert!(matches!(d.attach_card(&s.nid, card, 1), Err(CardRejected::NotDelegated(_))));
    }

    #[test]
    fn revocation_removes_the_agent_and_is_sticky() {
        let mut s = Setup::new();
        let card = s.card(&[("evaluate", &[])]);
        s.announce(&card, 1, false);
        let mut d = Directory::from_records(&s.log);
        d.attach_card(&s.nid, card.clone(), 1).unwrap();
        assert_eq!(d.entries(1).len(), 1);
        s.announce(&card, 2, true);
        let mut rev = s.log.clone();
        d.ingest(rev.last().unwrap());
        assert!(d.entries(1).is_empty() && d.is_revoked(&s.nid, &s.agent.public_key_hex()));
        assert_eq!(d.attach_card(&s.nid, card, 1), Err(CardRejected::Revoked));
        // order independence: revocation seen first, announcement later
        rev.reverse();
        let d2 = Directory::from_records(&rev);
        assert!(d2.is_revoked(&s.nid, &s.agent.public_key_hex()));
    }

    #[test]
    fn newer_version_replaces_older_and_requires_its_own_document() {
        let mut s = Setup::new();
        let v1 = s.card(&[("evaluate", &[])]);
        s.announce(&v1, 1, false);
        let mut d = Directory::from_records(&s.log);
        d.attach_card(&s.nid, v1.clone(), 1).unwrap();
        let v2 = s.card(&[("evaluate", &[]), ("chat", &[])]);
        s.announce(&v2, 2, false);
        d.ingest(s.log.last().unwrap());
        assert!(d.entries(1).is_empty(), "stale document dropped until the new one is fetched");
        assert_eq!(d.attach_card(&s.nid, v1, 1), Err(CardRejected::HashMismatch));
        d.attach_card(&s.nid, v2, 1).unwrap();
        assert_eq!(d.search("chat", 1).len(), 1);
        assert_eq!(d.entries(1)[0].version, 2);
    }

    #[test]
    fn expired_delegation_hides_the_agent() {
        let mut s = Setup::new();
        let card = s.card(&[("evaluate", &[])]);
        s.announce(&card, 1, false);
        let mut d = Directory::from_records(&s.log);
        d.attach_card(&s.nid, card, 100).unwrap();
        assert_eq!(d.entries(9_999).len(), 1);
        assert!(d.entries(10_000).is_empty());
    }
}
