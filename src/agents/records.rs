//! Typed bodies for the provenance record kinds of Steps 4-5.
//!
//! Every body is carried inside `RecordKind::Provenance { kind, body }`, so it
//! is hash-chained and Ed25519-signed with the rest of the node's log. A body
//! that does not deserialize into its schema (unknown fields included) or that
//! carries a malformed hash reference is rejected by `UpdateRecord::verify`.
//!
//! Object identity: an object's id is the hash of the record that carries it.
//! That hash commits to the author, the kind and the whole body, which makes
//! every object content-addressed and immutable. Cross-references
//! (`program`, `parents`, `run`, ...) are record hashes.

use serde::{Deserialize, Serialize};

pub const KIND_OBJECTIVE: &str = "research.objective";
pub const KIND_AGENDA: &str = "research.agenda";
pub const KIND_CANDIDATE: &str = "research.candidate";
pub const KIND_RUN: &str = "research.run";
pub const KIND_FINDING: &str = "research.finding";
pub const KIND_DECISION: &str = "research.decision";
/// Task-market claim lifecycle (A3). Not in the handoff's kind table; added because
/// claims, leases and heartbeats must be signed and attributable like everything else.
pub const KIND_CLAIM: &str = "research.claim";
/// A planner's proposal for a generation, carrying its VRF committee ticket (A5).
pub const KIND_PROPOSAL: &str = "research.proposal";
/// Blackboard post of a collaboration (M3), likewise an addition to the table.
pub const KIND_POST: &str = "coord.post";
pub const KIND_KEY_DELEGATION: &str = "agent.key_delegation";
pub const KIND_A2A_CARD: &str = "a2a.card";
pub const KIND_A2A_TASK: &str = "a2a.task";
pub const KIND_A2A_DELEGATION: &str = "a2a.delegation";
pub const KIND_A2A_ACCEPTANCE: &str = "a2a.acceptance";
pub const KIND_POLICY_AUDIT: &str = "policy.audit";

pub const ALL_KINDS: &[&str] = &[
    KIND_OBJECTIVE,
    KIND_AGENDA,
    KIND_CANDIDATE,
    KIND_RUN,
    KIND_FINDING,
    KIND_DECISION,
    KIND_CLAIM,
    KIND_PROPOSAL,
    KIND_POST,
    KIND_KEY_DELEGATION,
    KIND_A2A_CARD,
    KIND_A2A_TASK,
    KIND_A2A_DELEGATION,
    KIND_A2A_ACCEPTANCE,
    KIND_POLICY_AUDIT,
];

/// True for a 64-char lowercase hex string (SHA-256 / record hash).
pub fn is_hash(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn check_hash(field: &str, s: &str) -> Result<(), String> {
    if is_hash(s) { Ok(()) } else { Err(format!("{field} is not a 64-char lowercase hex hash")) }
}

fn check_hashes(field: &str, v: &[String]) -> Result<(), String> {
    v.iter().try_for_each(|h| check_hash(field, h))
}

fn check_nonempty(field: &str, s: &str) -> Result<(), String> {
    if s.trim().is_empty() { Err(format!("{field} must not be empty")) } else { Ok(()) }
}

fn check_finite(field: &str, v: f64) -> Result<(), String> {
    if v.is_finite() { Ok(()) } else { Err(format!("{field} must be finite")) }
}

// --------------------------------------------------------------------------
// Step 4: research loop
// --------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Maximize,
    Minimize,
}

/// Maturity rules, preregistered in the Objective so every node derives the
/// same lanes from the same evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaturityRules {
    /// Distinct evaluators that must have produced a finding before a
    /// candidate leaves the incubator.
    pub min_evaluators: u32,
    /// Candidates in the frontier lane are the best `frontier_size` mature ones.
    pub frontier_size: u32,
    /// A mature candidate whose score is at least this good is a Gem.
    pub gem_threshold: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Objective {
    pub goal: String,
    pub metric: String,
    pub direction: Direction,
    pub maturity: MaturityRules,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Requirements {
    #[serde(default)]
    pub gpu_class: Option<String>,
    #[serde(default)]
    pub data_access: Option<String>,
    /// Upper bound on GPU-seconds an item may use.
    #[serde(default)]
    pub gpu_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgendaItem {
    pub id: String,
    pub description: String,
    #[serde(default)]
    pub requirements: Requirements,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Agenda {
    pub program: String,
    pub generation: u64,
    /// Hash of the evidence set the planner saw (see `EvidenceGraph::evidence_root`).
    pub evidence_root: String,
    pub committee: Vec<String>,
    pub items: Vec<AgendaItem>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposedItem {
    pub id: String,
    pub description: String,
    pub tag: String,
    #[serde(default)]
    pub requirements: Requirements,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Proposal {
    pub program: String,
    pub generation: u64,
    pub evidence_root: String,
    /// VRF public key and proof proving this node's committee ticket for (program, generation, evidence_root).
    pub ticket_public_key: String,
    pub ticket_proof: String,
    pub items: Vec<ProposedItem>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub program: String,
    /// SHA-256 of the diff against the parents (blob fetched over iroh-blobs).
    pub diff_hash: String,
    pub parents: Vec<String>,
    #[serde(default)]
    pub agenda_item: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Run {
    pub candidate: String,
    pub container_hash: String,
    pub input_hashes: Vec<String>,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Finding {
    pub run: String,
    pub candidate: String,
    pub metric: String,
    pub value: f64,
    pub evaluator_code_hash: String,
    /// Merkle root over the evaluation data; the data itself never leaves its owner.
    pub data_merkle_root: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DecisionAction {
    /// Remove a candidate from every lane.
    Retire { candidate: String },
    /// Approve or refuse a human-gated change (objective, budget, innovation gate).
    Gate { gate: String, approved: bool },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Decision {
    pub program: String,
    pub generation: u64,
    pub action: DecisionAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimAction {
    Claim,
    Heartbeat,
    Complete,
    Release,
}

/// One step of an agenda item's claim lifecycle (A3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimEvent {
    /// Record hash of the Agenda that posted the item.
    pub agenda: String,
    pub item_id: String,
    pub action: ClaimAction,
    /// Unix seconds claimed by the author.
    pub at: i64,
    /// Lease length (Claim only).
    #[serde(default)]
    pub lease_secs: u64,
    /// Record hash of the Claim this event refers to (Heartbeat/Complete/Release).
    #[serde(default)]
    pub claim: Option<String>,
    /// Result artifact hash (Complete only).
    #[serde(default)]
    pub result: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PostKind {
    Proposal,
    Critique,
    Support,
    Result,
}

/// A signed blackboard post (M3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Post {
    pub collaboration: String,
    pub kind: PostKind,
    /// Record hash of the post being answered, if any.
    #[serde(default)]
    pub reply_to: Option<String>,
    pub content_hash: String,
    #[serde(default)]
    pub text: String,
}

// --------------------------------------------------------------------------
// Agent keys and A2A
// --------------------------------------------------------------------------

/// A node delegates a scoped, expiring signing key to one of its agents. The
/// parent signature is the record signature itself. A later record with
/// `revoked = true` for the same key revokes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyDelegation {
    pub agent_public_key: String,
    pub scope: Vec<String>,
    pub issued_at: i64,
    pub expires_at: i64,
    #[serde(default)]
    pub revoked: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct A2aCard {
    pub agent_public_key: String,
    pub card_hash: String,
    pub version: u32,
    #[serde(default)]
    pub revoked: bool,
}

pub use crate::agents::a2a::TaskState;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct A2aTask {
    pub task_id: String,
    pub contract_hash: String,
    pub state: TaskState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct A2aDelegation {
    pub parent_task: String,
    pub sub_task: String,
    pub scope: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct A2aAcceptance {
    pub task_id: String,
    pub result_hash: String,
    pub accepted: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyAudit {
    pub action: String,
    pub approved: bool,
    pub decided_by: String,
}

fn parse<T: serde::de::DeserializeOwned>(body: &serde_json::Value) -> Result<T, String> {
    serde_json::from_value(body.clone()).map_err(|e| e.to_string())
}

/// Validate `body` against the schema of `kind`.
pub fn validate_body(kind: &str, body: &serde_json::Value) -> Result<(), String> {
    match kind {
        KIND_OBJECTIVE => {
            let o: Objective = parse(body)?;
            check_nonempty("goal", &o.goal)?;
            check_nonempty("metric", &o.metric)?;
            check_finite("gem_threshold", o.maturity.gem_threshold)?;
            if o.maturity.min_evaluators == 0 {
                return Err("min_evaluators must be at least 1".into());
            }
            if o.maturity.frontier_size == 0 {
                return Err("frontier_size must be at least 1".into());
            }
        }
        KIND_AGENDA => {
            let a: Agenda = parse(body)?;
            check_hash("program", &a.program)?;
            check_hash("evidence_root", &a.evidence_root)?;
            if a.committee.is_empty() {
                return Err("committee must not be empty".into());
            }
            let mut ids = std::collections::HashSet::new();
            for item in &a.items {
                check_nonempty("item id", &item.id)?;
                if !ids.insert(&item.id) {
                    return Err(format!("duplicate agenda item id {}", item.id));
                }
            }
        }
        KIND_CANDIDATE => {
            let c: Candidate = parse(body)?;
            check_hash("program", &c.program)?;
            check_hash("diff_hash", &c.diff_hash)?;
            check_hashes("parents", &c.parents)?;
        }
        KIND_RUN => {
            let r: Run = parse(body)?;
            check_hash("candidate", &r.candidate)?;
            check_hash("container_hash", &r.container_hash)?;
            check_hashes("input_hashes", &r.input_hashes)?;
        }
        KIND_FINDING => {
            let f: Finding = parse(body)?;
            check_hash("run", &f.run)?;
            check_hash("candidate", &f.candidate)?;
            check_hash("evaluator_code_hash", &f.evaluator_code_hash)?;
            check_hash("data_merkle_root", &f.data_merkle_root)?;
            check_nonempty("metric", &f.metric)?;
            check_finite("value", f.value)?;
        }
        KIND_DECISION => {
            let d: Decision = parse(body)?;
            check_hash("program", &d.program)?;
            if let DecisionAction::Retire { candidate } = &d.action {
                check_hash("candidate", candidate)?;
            }
        }
        KIND_PROPOSAL => {
            let p: Proposal = parse(body)?;
            check_hash("program", &p.program)?;
            check_hash("evidence_root", &p.evidence_root)?;
            check_pubkey(&p.ticket_public_key)?;
            if hex::decode(&p.ticket_proof).map(|b| b.len()) != Ok(80) {
                return Err("ticket_proof must be 80 bytes of hex".into());
            }
            if p.items.len() > 64 {
                return Err("too many items in a proposal".into());
            }
            let mut ids = std::collections::HashSet::new();
            for it in &p.items {
                check_nonempty("item id", &it.id)?;
                if !ids.insert(&it.id) {
                    return Err(format!("duplicate item id {}", it.id));
                }
            }
        }
        KIND_CLAIM => {
            let c: ClaimEvent = parse(body)?;
            check_hash("agenda", &c.agenda)?;
            check_nonempty("item_id", &c.item_id)?;
            if let Some(h) = &c.claim { check_hash("claim", h)?; }
            if let Some(h) = &c.result { check_hash("result", h)?; }
            match c.action {
                ClaimAction::Claim if c.lease_secs == 0 => return Err("a claim needs a lease".into()),
                ClaimAction::Heartbeat | ClaimAction::Complete | ClaimAction::Release if c.claim.is_none() => {
                    return Err("event must reference its claim".into())
                }
                ClaimAction::Complete if c.result.is_none() => return Err("complete needs a result hash".into()),
                _ => {}
            }
        }
        KIND_POST => {
            let p: Post = parse(body)?;
            check_nonempty("collaboration", &p.collaboration)?;
            check_hash("content_hash", &p.content_hash)?;
            if let Some(h) = &p.reply_to { check_hash("reply_to", h)?; }
            if p.text.len() > 16 * 1024 { return Err("post text too long".into()); }
        }
        KIND_KEY_DELEGATION => {
            let k: KeyDelegation = parse(body)?;
            check_pubkey(&k.agent_public_key)?;
            if k.scope.is_empty() {
                return Err("scope must not be empty".into());
            }
            if k.expires_at <= k.issued_at {
                return Err("expires_at must be after issued_at".into());
            }
        }
        KIND_A2A_CARD => {
            let c: A2aCard = parse(body)?;
            check_pubkey(&c.agent_public_key)?;
            check_hash("card_hash", &c.card_hash)?;
        }
        KIND_A2A_TASK => {
            let t: A2aTask = parse(body)?;
            check_nonempty("task_id", &t.task_id)?;
            check_hash("contract_hash", &t.contract_hash)?;
        }
        KIND_A2A_DELEGATION => {
            let d: A2aDelegation = parse(body)?;
            check_nonempty("parent_task", &d.parent_task)?;
            check_nonempty("sub_task", &d.sub_task)?;
            if d.parent_task == d.sub_task {
                return Err("a task cannot delegate to itself".into());
            }
        }
        KIND_A2A_ACCEPTANCE => {
            let a: A2aAcceptance = parse(body)?;
            check_nonempty("task_id", &a.task_id)?;
            check_hash("result_hash", &a.result_hash)?;
        }
        KIND_POLICY_AUDIT => {
            let p: PolicyAudit = parse(body)?;
            check_nonempty("action", &p.action)?;
            check_nonempty("decided_by", &p.decided_by)?;
        }
        other => return Err(format!("unknown provenance kind '{other}'")),
    }
    Ok(())
}

fn check_pubkey(s: &str) -> Result<(), String> {
    match hex::decode(s) {
        Ok(b) if b.len() == 32 => Ok(()),
        _ => Err("agent_public_key must be 32 bytes of hex".into()),
    }
}

/// Serialize a typed body into a `RecordKind::Provenance`.
pub fn provenance<T: Serialize>(kind: &str, body: &T) -> crate::history::RecordKind {
    crate::history::RecordKind::Provenance {
        kind: kind.to_string(),
        body: serde_json::to_value(body).expect("provenance body serializes"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn h(c: char) -> String { c.to_string().repeat(64) }

    #[test]
    fn rejects_unknown_fields_and_bad_hashes() {
        let ok = json!({"program": h('a'), "diff_hash": h('b'), "parents": []});
        assert!(validate_body(KIND_CANDIDATE, &ok).is_ok());
        let extra = json!({"program": h('a'), "diff_hash": h('b'), "parents": [], "x": 1});
        assert!(validate_body(KIND_CANDIDATE, &extra).is_err());
        let bad = json!({"program": "nothex", "diff_hash": h('b'), "parents": []});
        assert!(validate_body(KIND_CANDIDATE, &bad).is_err());
        assert!(validate_body("research.bogus", &ok).is_err());
    }

    #[test]
    fn finding_must_be_finite() {
        let mut f = Finding {
            run: h('a'), candidate: h('b'), metric: "acc".into(), value: 0.5,
            evaluator_code_hash: h('c'), data_merkle_root: h('d'),
        };
        let v = serde_json::to_value(&f).unwrap();
        assert!(validate_body(KIND_FINDING, &v).is_ok());
        f.value = f64::NAN;
        // NaN serializes to JSON null, which fails to deserialize as f64.
        assert!(validate_body(KIND_FINDING, &serde_json::to_value(&f).unwrap()).is_err());
    }

    #[test]
    fn decision_roundtrips_tagged_action() {
        let d = Decision {
            program: h('a'), generation: 2,
            action: DecisionAction::Gate { gate: "budget".into(), approved: true },
        };
        let v = serde_json::to_value(&d).unwrap();
        assert_eq!(v["action"]["type"], "gate");
        assert!(validate_body(KIND_DECISION, &v).is_ok());
        assert_eq!(parse::<Decision>(&v).unwrap(), d);
    }

    #[test]
    fn delegation_needs_future_expiry_and_scope() {
        let k = json!({"agent_public_key": "ab".repeat(32), "scope": [], "issued_at": 1, "expires_at": 5});
        assert!(validate_body(KIND_KEY_DELEGATION, &k).is_err());
        let k = json!({"agent_public_key": "ab".repeat(32), "scope": ["x"], "issued_at": 5, "expires_at": 5});
        assert!(validate_body(KIND_KEY_DELEGATION, &k).is_err());
    }
}
