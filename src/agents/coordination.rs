//! Coordination without an orchestrator (M3).
//!
//! * Contract-net: a task is announced to every agent, agents bid, the award
//!   is a deterministic function of the bids (so any observer can check it),
//!   and the winner may sub-contract by running the same protocol one level
//!   down. If the winner fails, the next-best bid is awarded.
//! * Blackboard: a collaboration's signed `coord.post` records, viewed as
//!   proposals with critiques and support; scored with reputation weights.
//! * Consensus: trust-weighted pooling of answers, and a critique/debate loop
//!   that repeats until the pool reaches quorum or the rounds run out.
//! * Team templates: researcher, coder, reviewer, evaluator roles filled from the
//!   discovery directory, spread across institutions.

use crate::agents::discovery::Directory;
use crate::agents::records::{Post, PostKind, KIND_POST};
use crate::history::{RecordKind, UpdateRecord};
use serde_json::Value;
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// Contract-net
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TaskSpec {
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub id: String,
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub skill: String,
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub input: Value,
    pub max_price: u64,
    pub deadline_secs: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Bid {
    pub bidder: String,
    pub price: u64,
    pub eta_secs: u64,
    /// Bidder's reputation in [0,1] for this skill, as known to the network.
    pub reputation: f64,
}

impl Bid {
    /// Higher is better: reputation, discounted by price and time used.
    pub fn utility(&self, task: &TaskSpec) -> f64 {
        let price_term = 1.0 - self.price as f64 / (task.max_price.max(1)) as f64;
        let time_term = 1.0 - self.eta_secs as f64 / (task.deadline_secs.max(1)) as f64;
        self.reputation.clamp(0.0, 1.0) * (0.25 + price_term.max(0.0)) * (0.25 + time_term.max(0.0))
    }
}

/// Rank admissible bids best-first. Deterministic: ties go to the lowest bidder id.
pub fn rank_bids(task: &TaskSpec, bids: &[Bid]) -> Vec<Bid> {
    let mut ok: Vec<Bid> = bids.iter().filter(|b| b.price <= task.max_price && b.eta_secs <= task.deadline_secs && b.reputation.is_finite()).cloned().collect();
    ok.sort_by(|a, b| b.utility(task).partial_cmp(&a.utility(task)).unwrap().then_with(|| a.bidder.cmp(&b.bidder)));
    ok
}

#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub trait Contractor {
    fn id(&self) -> &str;
    /// None = not interested / cannot do it.
    fn bid(&self, task: &TaskSpec) -> Option<Bid>;
    /// Perform the task, optionally delegating parts through `net`.
    fn execute(&self, task: &TaskSpec, price: u64, net: &ContractNet<'_>) -> Result<Value, String>;
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub struct Outcome {
    pub output: Value,
    pub winner: String,
    pub price: u64,
    /// Contractors who failed before the winner succeeded.
    pub failed: Vec<String>,
}

#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub struct ContractNet<'a> {
    contractors: &'a [Box<dyn Contractor + 'a>],
    /// Agents excluded from bidding (the delegator chain, to avoid loops).
    excluded: Vec<String>,
    depth: usize,
    max_depth: usize,
    messages: &'a Cell<u64>,
}

impl<'a> ContractNet<'a> {
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn new(contractors: &'a [Box<dyn Contractor + 'a>], messages: &'a Cell<u64>, max_depth: usize) -> Self {
        ContractNet { contractors, excluded: vec![], depth: 0, max_depth, messages }
    }

    /// Announce -> bid -> award -> execute, falling back down the ranking on failure.
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub fn delegate(&self, task: &TaskSpec, delegator: &str) -> Result<Outcome, String> {
        if self.depth > self.max_depth {
            return Err("delegation depth limit reached".into());
        }
        let mut excluded = self.excluded.clone();
        excluded.push(delegator.to_string());
        let eligible: Vec<&Box<dyn Contractor + 'a>> = self.contractors.iter().filter(|c| !excluded.iter().any(|e| e == c.id())).collect();
        self.messages.set(self.messages.get() + eligible.len() as u64); // announcements
        let bids: Vec<Bid> = eligible.iter().filter_map(|c| c.bid(task)).collect();
        self.messages.set(self.messages.get() + bids.len() as u64); // bids
        let ranked = rank_bids(task, &bids);
        if ranked.is_empty() {
            return Err(format!("no admissible bid for task {}", task.id));
        }
        let child = ContractNet { contractors: self.contractors, excluded, depth: self.depth + 1, max_depth: self.max_depth, messages: self.messages };
        let mut failed = vec![];
        for bid in ranked {
            let c = eligible.iter().find(|c| c.id() == bid.bidder).expect("bidder is eligible");
            self.messages.set(self.messages.get() + 2); // award + result
            match c.execute(task, bid.price, &child) {
                Ok(output) => return Ok(Outcome { output, winner: bid.bidder, price: bid.price, failed }),
                Err(_) => failed.push(bid.bidder),
            }
        }
        Err(format!("every bidder failed for task {}: {:?}", task.id, failed))
    }
}

// ---------------------------------------------------------------------------
// Blackboard
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BoardPost {
    pub id: String,
    pub author: String,
    pub post: Post,
}

#[derive(Debug, Default)]
pub struct Blackboard {
    collaboration: String,
    posts: BTreeMap<String, BoardPost>,
}

impl Blackboard {
    pub fn from_records<'a>(collaboration: &str, records: impl IntoIterator<Item = &'a UpdateRecord>) -> Self {
        let mut b = Blackboard { collaboration: collaboration.into(), posts: BTreeMap::new() };
        for r in records {
            b.insert(r);
        }
        b
    }

    pub fn insert(&mut self, r: &UpdateRecord) -> bool {
        let RecordKind::Provenance { kind, body } = &r.kind else { return false };
        if kind != KIND_POST || r.verify().is_err() {
            return false;
        }
        let Ok(p) = serde_json::from_value::<Post>(body.clone()) else { return false };
        if p.collaboration != self.collaboration {
            return false;
        }
        self.posts.insert(r.hash.clone(), BoardPost { id: r.hash.clone(), author: r.node_id.clone(), post: p });
        true
    }

    pub fn proposals(&self) -> Vec<&BoardPost> {
        self.posts.values().filter(|p| p.post.kind == PostKind::Proposal).collect()
    }

    pub fn replies(&self, id: &str) -> Vec<&BoardPost> {
        self.posts.values().filter(|p| p.post.reply_to.as_deref() == Some(id)).collect()
    }

    /// Weighted support minus weighted critique, one voice per author per proposal
    /// (an author's latest reply wins, by record-hash order). The author of the
    /// proposal counts as support.
    pub fn score(&self, proposal: &str, weight: &dyn Fn(&str) -> f64) -> f64 {
        let Some(p) = self.posts.get(proposal) else { return 0.0 };
        let mut voice: BTreeMap<&str, f64> = BTreeMap::new();
        voice.insert(&p.author, 1.0);
        for r in self.replies(proposal) {
            match r.post.kind {
                PostKind::Support => { voice.insert(&r.author, 1.0); }
                PostKind::Critique => { voice.insert(&r.author, -1.0); }
                _ => {}
            }
        }
        voice.iter().map(|(a, v)| v * weight(a)).sum()
    }

    /// Posted results, in id order.
    pub fn results(&self) -> Vec<&BoardPost> {
        self.posts.values().filter(|p| p.post.kind == PostKind::Result).collect()
    }
}

// ---------------------------------------------------------------------------
// Consensus
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Pooled {
    pub answer: String,
    pub share: f64,
    pub dissent: Vec<(String, String)>,
}

/// Trust-weighted plurality over discrete answers. Returns Some only if the winning
/// answer holds at least `quorum` of the total weight. Zero/negative weights are ignored.
pub fn pool(votes: &[(String, String)], weight: &dyn Fn(&str) -> f64, quorum: f64) -> Option<Pooled> {
    let mut tally: BTreeMap<&str, f64> = BTreeMap::new();
    let mut total = 0.0;
    for (agent, ans) in votes {
        let w = weight(agent);
        if w > 0.0 && w.is_finite() {
            *tally.entry(ans).or_default() += w;
            total += w;
        }
    }
    let (best, w) = tally.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap().then_with(|| b.0.cmp(a.0)))?;
    let share = w / total;
    if share + 1e-12 < quorum {
        return None;
    }
    let dissent = votes.iter().filter(|(_, a)| a != best).cloned().collect();
    Some(Pooled { answer: best.to_string(), share, dissent })
}

#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub trait Debater {
    fn id(&self) -> &str;
    /// `peers` are the other debaters' answers from the previous round.
    fn answer(&mut self, question: &str, round: usize, peers: &[(String, String)]) -> String;
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub struct Debate {
    pub consensus: Option<Pooled>,
    pub rounds_used: usize,
    pub transcript: Vec<Vec<(String, String)>>,
}

#[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
pub fn debate(question: &str, agents: &mut [Box<dyn Debater>], weight: &dyn Fn(&str) -> f64, quorum: f64, max_rounds: usize) -> Debate {
    let mut transcript: Vec<Vec<(String, String)>> = vec![];
    let mut prev: Vec<(String, String)> = vec![];
    for round in 0..max_rounds.max(1) {
        let answers: Vec<(String, String)> = agents
            .iter_mut()
            .map(|a| {
                let peers: Vec<(String, String)> = prev.iter().filter(|(id, _)| id != a.id()).cloned().collect();
                (a.id().to_string(), a.answer(question, round, &peers))
            })
            .collect();
        transcript.push(answers.clone());
        if let Some(c) = pool(&answers, weight, quorum) {
            return Debate { consensus: Some(c), rounds_used: round + 1, transcript };
        }
        prev = answers;
    }
    Debate { consensus: None, rounds_used: transcript.len(), transcript }
}

// ---------------------------------------------------------------------------
// Team templates
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RoleSpec {
    pub role: &'static str,
    pub skill: &'static str,
    pub count: usize,
}

#[derive(Debug, Clone)]
pub struct TeamTemplate {
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub name: &'static str,
    pub roles: Vec<RoleSpec>,
}

impl TeamTemplate {
    /// researcher + coder + reviewer + evaluator across nodes.
    pub fn research_team() -> Self {
        TeamTemplate {
            name: "research-team",
            roles: vec![
                RoleSpec { role: "researcher", skill: "research", count: 1 },
                RoleSpec { role: "coder", skill: "code", count: 1 },
                RoleSpec { role: "reviewer", skill: "review", count: 2 },
                RoleSpec { role: "evaluator", skill: "evaluate", count: 1 },
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Member {
    pub role: &'static str,
    pub node_id: String,
    pub agent_key: String,
}

/// Fill a template from the directory: for each role, verified cards advertising
/// the skill, best reputation first, never reusing an agent and preferring nodes
/// the team does not yet use. Returns Err naming the first role that cannot be filled.
pub fn form_team(t: &TeamTemplate, dir: &Directory, reputation: &dyn Fn(&str, &str) -> f64, now: i64) -> Result<Vec<Member>, String> {
    let mut team: Vec<Member> = vec![];
    let mut used_agents: BTreeSet<String> = BTreeSet::new();
    for spec in &t.roles {
        for _ in 0..spec.count {
            let used_nodes: BTreeSet<&str> = team.iter().map(|m| m.node_id.as_str()).collect();
            let mut cands: Vec<_> = dir.search(spec.skill, now).into_iter().filter(|e| !used_agents.contains(&e.agent_key)).collect();
            cands.sort_by(|a, b| {
                let fresh = |e: &&crate::agents::discovery::Entry| !used_nodes.contains(e.node_id.as_str());
                fresh(b)
                    .cmp(&fresh(a))
                    .then_with(|| reputation(&b.node_id, &b.agent_key).partial_cmp(&reputation(&a.node_id, &a.agent_key)).unwrap())
                    .then_with(|| a.agent_key.cmp(&b.agent_key))
            });
            let e = cands.first().ok_or_else(|| format!("no agent available for role '{}' (skill '{}')", spec.role, spec.skill))?;
            used_agents.insert(e.agent_key.clone());
            team.push(Member { role: spec.role, node_id: e.node_id.clone(), agent_key: e.agent_key.clone() });
        }
    }
    Ok(team)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::graph::make_record;
    use crate::identity::Keypair;
    use serde_json::json;

    fn task(price: u64) -> TaskSpec {
        TaskSpec { id: "t".into(), skill: "sum".into(), input: json!([1, 2, 3]), max_price: price, deadline_secs: 100 }
    }

    // -- a tiny contractor population --
    struct Worker { id: String, price: u64, rep: f64, fails: bool, subcontract: bool }
    impl Contractor for Worker {
        fn id(&self) -> &str { &self.id }
        fn bid(&self, t: &TaskSpec) -> Option<Bid> {
            (t.skill == "sum" || t.skill == "part").then(|| Bid { bidder: self.id.clone(), price: self.price, eta_secs: 10, reputation: self.rep })
        }
        fn execute(&self, t: &TaskSpec, price: u64, net: &ContractNet<'_>) -> Result<Value, String> {
            if self.fails {
                return Err("crashed".into());
            }
            if self.subcontract && t.skill == "sum" {
                // split the job: sub-contract half of it, with half of our price as budget
                let sub = TaskSpec { id: format!("{}/sub", t.id), skill: "part".into(), input: json!([3]), max_price: price / 2, deadline_secs: 50 };
                let r = net.delegate(&sub, &self.id)?;
                let rest: i64 = [1, 2].iter().sum();
                return Ok(json!(rest + r.output.as_i64().unwrap()));
            }
            let s: i64 = t.input.as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).sum();
            Ok(json!(s))
        }
    }
    fn w(id: &str, price: u64, rep: f64) -> Box<dyn Contractor> { Box::new(Worker { id: id.into(), price, rep, fails: false, subcontract: false }) }

    #[test]
    fn award_prefers_reputation_and_price_and_is_deterministic() {
        let t = task(100);
        let bids = vec![
            Bid { bidder: "cheap-bad".into(), price: 10, eta_secs: 10, reputation: 0.2 },
            Bid { bidder: "good".into(), price: 40, eta_secs: 10, reputation: 0.9 },
            Bid { bidder: "too-pricey".into(), price: 500, eta_secs: 10, reputation: 1.0 },
            Bid { bidder: "too-slow".into(), price: 10, eta_secs: 5000, reputation: 1.0 },
        ];
        let r = rank_bids(&t, &bids);
        assert_eq!(r.iter().map(|b| b.bidder.as_str()).collect::<Vec<_>>(), ["good", "cheap-bad"]);
        let mut rev = bids.clone();
        rev.reverse();
        assert_eq!(rank_bids(&t, &rev), r);
        // identical bids tie-break on id
        let same = vec![Bid { bidder: "b".into(), price: 1, eta_secs: 1, reputation: 0.5 }, Bid { bidder: "a".into(), price: 1, eta_secs: 1, reputation: 0.5 }];
        assert_eq!(rank_bids(&t, &same)[0].bidder, "a");
    }

    #[test]
    fn failed_winner_falls_back_to_next_best_bid() {
        let msgs = Cell::new(0);
        let cs: Vec<Box<dyn Contractor>> = vec![
            Box::new(Worker { id: "star".into(), price: 20, rep: 0.99, fails: true, subcontract: false }),
            w("steady", 30, 0.8),
            w("weak", 30, 0.1),
        ];
        let net = ContractNet::new(&cs, &msgs, 3);
        let out = net.delegate(&task(100), "requester").unwrap();
        assert_eq!((out.winner.as_str(), out.output.clone()), ("steady", json!(6)));
        assert_eq!(out.failed, vec!["star"]);
        assert!(msgs.get() >= 3 + 3 + 4);
    }

    #[test]
    fn winner_can_subcontract_without_looping_back_to_the_delegator_chain() {
        let msgs = Cell::new(0);
        let cs: Vec<Box<dyn Contractor>> = vec![
            Box::new(Worker { id: "prime".into(), price: 60, rep: 0.95, fails: false, subcontract: true }),
            w("sub1", 20, 0.3),
        ];
        let net = ContractNet::new(&cs, &msgs, 3);
        let out = net.delegate(&task(100), "requester").unwrap();
        assert_eq!(out.winner, "prime");
        assert_eq!(out.output, json!(6));
        // 2 announcements + 2 bids + award/result for prime, then the sub-contract round:
        // only sub1 may bid (prime is in the delegator chain) -> 1 + 1 + 2 more.
        assert_eq!(msgs.get(), 2 + 2 + 2 + 1 + 1 + 2);
    }

    #[test]
    fn no_bidders_and_depth_limit_are_errors() {
        let msgs = Cell::new(0);
        let none: Vec<Box<dyn Contractor>> = vec![];
        assert!(ContractNet::new(&none, &msgs, 3).delegate(&task(10), "r").is_err());
        let cs: Vec<Box<dyn Contractor>> = vec![w("a", 1, 0.5)];
        let net = ContractNet::new(&cs, &msgs, 0);
        assert!(net.delegate(&task(10), "r").is_ok());
        // The only contractor is the delegator: nobody left to bid
        assert!(net.delegate(&task(10), "a").is_err());
    }

    // -- blackboard --
    struct Author { kp: Keypair, id: String }
    fn author() -> Author { let kp = Keypair::generate(); let id = kp.node_id().to_string(); Author { kp, id } }
    fn post(a: &Author, collab: &str, kind: PostKind, reply_to: Option<&str>, text: &str, n: u8) -> UpdateRecord {
        let mut r = make_record(&a.id, &format!("{:064x}", n), KIND_POST, &Post { collaboration: collab.into(), kind, reply_to: reply_to.map(String::from), content_hash: format!("{:064x}", n), text: text.into() });
        r.sign(&a.kp);
        r
    }

    #[test]
    fn blackboard_scores_proposals_by_weighted_support_and_critique() {
        let (p, s1, s2, c1) = (author(), author(), author(), author());
        let prop = post(&p, "col", PostKind::Proposal, None, "use lr=3e-4", 1);
        let recs = vec![
            prop.clone(),
            post(&s1, "col", PostKind::Support, Some(&prop.hash), "+1", 2),
            post(&s2, "col", PostKind::Support, Some(&prop.hash), "+1", 3),
            post(&c1, "col", PostKind::Critique, Some(&prop.hash), "diverges", 4),
            post(&c1, "other-collab", PostKind::Proposal, None, "ignored", 5),
        ];
        let b = Blackboard::from_records("col", &recs);
        assert_eq!(b.proposals().len(), 1);
        assert_eq!(b.replies(&prop.hash).len(), 3);
        assert_eq!(b.score(&prop.hash, &|_| 1.0), 2.0); // author + 2 supports - 1 critique
        // a heavily trusted critic outweighs the rest
        let heavy = c1.id.clone();
        assert!(b.score(&prop.hash, &|a| if a == heavy { 5.0 } else { 1.0 }) < 0.0);
        // forged post (bad signature) is dropped
        let mut forged = post(&s1, "col", PostKind::Support, Some(&prop.hash), "evil", 6);
        forged.signature = "00".repeat(64);
        let mut with = recs.clone();
        with.push(forged);
        assert_eq!(Blackboard::from_records("col", &with).replies(&prop.hash).len(), 3);
    }

    #[test]
    fn one_voice_per_author_latest_reply_wins() {
        let (p, v) = (author(), author());
        let prop = post(&p, "c", PostKind::Proposal, None, "x", 1);
        let recs = vec![prop.clone(), post(&v, "c", PostKind::Critique, Some(&prop.hash), "no", 2), post(&v, "c", PostKind::Critique, Some(&prop.hash), "still no", 3), post(&v, "c", PostKind::Critique, Some(&prop.hash), "again", 4)];
        assert_eq!(Blackboard::from_records("c", &recs).score(&prop.hash, &|_| 1.0), 0.0); // 1 - 1, not 1 - 3
    }

    // -- consensus --
    fn v(a: &str, ans: &str) -> (String, String) { (a.into(), ans.into()) }

    #[test]
    fn pooling_is_trust_weighted_with_quorum_and_dissent() {
        let votes = vec![v("a", "yes"), v("b", "yes"), v("c", "no"), v("d", "no"), v("e", "no")];
        let eq = pool(&votes, &|_| 1.0, 0.5).unwrap();
        assert_eq!((eq.answer.as_str(), eq.dissent.len()), ("no", 2));
        // trusted voters flip it
        let heavy = pool(&votes, &|x| if x == "a" || x == "b" { 5.0 } else { 1.0 }, 0.5).unwrap();
        assert_eq!(heavy.answer, "yes");
        // no quorum
        assert!(pool(&votes, &|_| 1.0, 0.7).is_none());
        // zero-weight voters (e.g. flagged colluders) do not count
        let muted = pool(&votes, &|x| if x == "c" || x == "d" || x == "e" { 0.0 } else { 1.0 }, 0.9).unwrap();
        assert_eq!(muted.answer, "yes");
        assert!(pool(&[], &|_| 1.0, 0.5).is_none());
    }

    struct Stubborn(String, String);
    impl Debater for Stubborn {
        fn id(&self) -> &str { &self.0 }
        fn answer(&mut self, _: &str, _: usize, _: &[(String, String)]) -> String { self.1.clone() }
    }
    /// Starts with its own view, adopts the plurality of its peers' answers after seeing them.
    struct Persuadable(String, String);
    impl Debater for Persuadable {
        fn id(&self) -> &str { &self.0 }
        fn answer(&mut self, _: &str, round: usize, peers: &[(String, String)]) -> String {
            if round > 0 {
                let mut c: BTreeMap<&str, usize> = BTreeMap::new();
                for (_, a) in peers { *c.entry(a).or_default() += 1; }
                if let Some((best, _)) = c.iter().max_by_key(|(a, n)| (**n, std::cmp::Reverse(**a))) { self.1 = best.to_string(); }
            }
            self.1.clone()
        }
    }

    #[test]
    fn debate_converges_when_agents_are_persuadable_and_stops_otherwise() {
        let mut team: Vec<Box<dyn Debater>> = vec![
            Box::new(Stubborn("s1".into(), "A".into())),
            Box::new(Stubborn("s2".into(), "A".into())),
            Box::new(Persuadable("p1".into(), "B".into())),
            Box::new(Persuadable("p2".into(), "B".into())),
            Box::new(Persuadable("p3".into(), "C".into())),
        ];
        let d = debate("q", &mut team, &|_| 1.0, 0.8, 5);
        let c = d.consensus.expect("should converge");
        assert_eq!(c.answer, "A");
        assert!(d.rounds_used >= 2, "round 0 had no quorum, persuasion needed a second round");
        assert_eq!(d.transcript.len(), d.rounds_used);

        let mut split: Vec<Box<dyn Debater>> = vec![Box::new(Stubborn("a".into(), "X".into())), Box::new(Stubborn("b".into(), "Y".into()))];
        let d = debate("q", &mut split, &|_| 1.0, 0.8, 3);
        assert!(d.consensus.is_none());
        assert_eq!(d.rounds_used, 3);
    }

    // -- teams --
    #[test]
    fn team_is_filled_by_skill_reputation_and_spread_across_nodes() {
        use crate::agents::a2a::*;
        use crate::agents::delegation::delegation_record;
        use crate::agents::discovery::SCOPE_CARD;
        use crate::agents::records::{A2aCard, KIND_A2A_CARD};
        // three institutions, each with agents offering different skills
        let mut log = vec![];
        let mut cards: Vec<(String, AgentCard)> = vec![];
        let mut reps: BTreeMap<String, f64> = BTreeMap::new();
        let mut node_of: BTreeMap<&str, String> = BTreeMap::new();
        let mut nodes: BTreeMap<&str, Keypair> = BTreeMap::new();
        let mut tails: BTreeMap<&str, String> = BTreeMap::new();
        for (inst, skills, rep) in [
            ("A1", vec!["research"], 0.9), ("A2", vec!["review"], 0.6),
            ("B1", vec!["code", "evaluate"], 0.8), ("B2", vec!["review"], 0.95),
            ("C1", vec!["evaluate"], 0.7), ("C2", vec!["review"], 0.85),
        ] {
            // three institutions (A, B, C), two agents each, sharing one node key per institution
            let letter = &inst[..1];
            let node = nodes.entry(letter).or_insert_with(Keypair::generate).clone();
            let nid = node.node_id().to_string();
            node_of.insert(inst, nid.clone());
            let agent = Keypair::generate();
            let prev = tails.get(letter).cloned().unwrap_or("0".repeat(64));
            let mut d = delegation_record(&nid, &prev, &agent.public_key_hex(), &[SCOPE_CARD], 0, 1000, false);
            d.sign(&node);
            let mut card = AgentCard {
                name: inst.into(), description: "d".into(), version: "1".into(),
                supported_interfaces: vec![AgentInterface { url: "q".into(), protocol_binding: "SLAKSHNA-QUIC".into(), protocol_version: "1.0".into(), tenant: None }],
                capabilities: AgentCapabilities::default(), security_schemes: None, security_requirements: vec![],
                default_input_modes: vec![], default_output_modes: vec![],
                skills: skills.iter().map(|s| AgentSkill { id: s.to_string(), name: s.to_string(), description: "d".into(), tags: vec![], examples: vec![], input_modes: vec![], output_modes: vec![] }).collect(),
                signatures: vec![], slakshna_node_id: Some(nid.clone()), slakshna_endpoint_id: None, slakshna_prices: Default::default(),
            };
            card.sign(&agent);
            let mut a = make_record(&nid, &d.hash, KIND_A2A_CARD, &A2aCard { agent_public_key: agent.public_key_hex(), card_hash: card.content_hash(), version: 1, revoked: false });
            a.sign(&node);
            tails.insert(letter, a.hash.clone());
            log.push(d);
            log.push(a);
            reps.insert(agent.public_key_hex(), rep);
            cards.push((nid, card));
        }
        let mut dir = Directory::from_records(&log);
        for (nid, c) in cards { dir.attach_card(&nid, c, 10).unwrap(); }
        let team = form_team(&TeamTemplate::research_team(), &dir, &|_, k| reps[k], 10).unwrap();
        assert_eq!(team.len(), 5);
        let roles: Vec<_> = team.iter().map(|m| m.role).collect();
        assert_eq!(roles, ["researcher", "coder", "reviewer", "reviewer", "evaluator"]);
        assert_eq!(team.iter().map(|m| &m.agent_key).collect::<BTreeSet<_>>().len(), 5, "no agent holds two roles");
        let nodes: BTreeSet<_> = team.iter().map(|m| &m.node_id).collect();
        assert_eq!(nodes.len(), 3, "the team spans all three institutions");
        // first reviewer: the best-reputed reviewer on a node not yet in the team (C), i.e. C2, not B2 (0.95 but node B is used)
        assert_eq!(team[2].node_id, node_of["C2"]);
        // second reviewer: every node used, so plain best reputation among the rest: B2
        assert_eq!(team[3].node_id, node_of["B2"]);
        // a missing skill fails loudly instead of returning a partial team
        let mut t = TeamTemplate::research_team();
        t.roles.push(RoleSpec { role: "translator", skill: "translate", count: 1 });
        assert!(form_team(&t, &dir, &|_, k| reps[k], 10).unwrap_err().contains("translator"));
    }
}
