//! Steps 4-5: federated agents (A1-A7) and A2A multi-agent systems (M1-M7).
//!
//! Built so far (see docs/AGENTS_A2A.md for status against the plan):
//! provenance record kinds and schemas, agent key delegation, budgets, the
//! evidence graph with deterministic lanes, the sandbox executor, the
//! persistent Python worker, the settlement-chain trait, and the
//! `slakshna/agent/1` ALPN.

pub mod a2a;
pub mod agent_loop;
pub mod a2a_client;
pub mod audit;
pub mod blobs;
pub mod budget;
pub mod coordination;
pub mod credit;
pub mod delegate;
pub mod delegation;
pub mod discovery;
pub mod economy;
pub mod evaluator;
pub mod gateway;
pub mod graph;
pub mod llm_bench;
pub mod llm_roles;
pub mod market;
pub mod merkle;
pub mod planning;
pub mod protocol;
pub mod records;
pub mod reputation;
pub mod research;
pub mod runtime;
pub mod safety;
pub mod service;
pub mod skills;
pub mod testbed;
pub mod tools;
pub mod sandbox;
pub mod settlement;
pub mod vrf;
pub mod worker;
