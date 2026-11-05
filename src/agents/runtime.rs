//! Starts the agent layer on a node: service, skills, card, gateway, mailbox,
//! directory refresh and program-topic watcher. Called from `main` when
//! `[agents] enabled = true`.

use crate::agents::a2a_client::{A2aClient, RemoteAgentSkill};
use crate::agents::gateway::{self, GatewayState};
use crate::agents::service::{AgentService, Broadcaster};
use crate::agents::skills::{ChatSkill, EvalRegistry, EvaluateSkill};
use crate::agents::worker::AgentWorker;
use crate::config::Config;
use crate::history::UpdateHistory;
use crate::identity::Keypair;
use crate::network::mesh::MeshNetwork;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, warn};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

pub struct AgentRuntime {
    pub service: Arc<AgentService>,
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub evals: Arc<EvalRegistry>,
}

pub async fn start(
    config: &Config,
    keypair: Keypair,
    history: Arc<RwLock<UpdateHistory>>,
    network: Arc<RwLock<MeshNetwork>>,
) -> Result<AgentRuntime, BoxError> {
    let cfg = config.agents.clone();
    let broadcaster: Arc<dyn Broadcaster> = network.clone();
    let svc = AgentService::new(cfg.clone(), keypair, history, broadcaster, &config.node.data_dir).await?;

    // --- skills ---
    let evals = Arc::new(EvalRegistry::default());
    evals.load_config(&cfg.datasets)?;
    let runtime = crate::agents::sandbox::Runtime::from_name("auto").unwrap_or(crate::agents::sandbox::Runtime::Apptainer);
    if !evals.dataset_names().is_empty() {
        svc.register_skill(Arc::new(crate::agents::audit::AuditSkill { registry: evals.clone() }));
        svc.register_skill(Arc::new(EvaluateSkill {
            registry: evals.clone(),
            allowed_images: cfg.allowed_images.clone(),
            runtime,
            work_root: std::path::Path::new(&config.node.data_dir).join("agents/eval-work"),
        }));
        info!("🧪 evaluate skill on datasets {:?}", evals.dataset_names());
    }
    if let Some(url) = &cfg.serve_url {
        let script = std::env::var("SLAKSHNA_AGENT_WORKER").unwrap_or_else(|_| "agent_worker.py".into());
        let mut wenv = vec![("SLAKSHNA_SERVE_URL".to_string(), url.clone())];
        if let Some(m) = &cfg.serve_model {
            wenv.push(("SLAKSHNA_SERVE_MODEL".to_string(), m.clone()));
        }
        let worker = Arc::new(AgentWorker::new(&cfg.python, &script, wenv));
        svc.register_skill(Arc::new(ChatSkill { worker: worker.clone(), max_tokens: 512 }));
        info!("💬 chat skill via {url}");

        // Tool-using agent: MCP tools from the manifest and allow-listed outside APIs.
        let hub = match &cfg.tools.manifest {
            Some(path) => {
                let m = crate::agents::tools::Manifest::load(path)?;
                Some(Arc::new(crate::agents::tools::ToolHub::from_manifest(&m, svc.token_verifier(), svc.policy.clone()).await.map_err(|e| format!("tool manifest: {e}"))?))
            }
            None => None,
        };
        let provider = if cfg.provider.allowed_hosts.is_empty() {
            None
        } else {
            Some(Arc::new(crate::agents::tools::HttpProvider::new(cfg.provider.allowed_hosts.clone(), cfg.provider.allow_insecure_loopback, svc.token_verifier(), svc.policy.clone()).map_err(|e| format!("provider: {e}"))?))
        };
        if hub.is_some() || provider.is_some() {
            svc.register_skill(Arc::new(crate::agents::agent_loop::AgentLoopSkill {
                svc: Arc::downgrade(&svc), worker, tools: hub, provider, max_steps: cfg.tools.max_steps, max_tokens: 512,
            }));
            info!("🛠️  agent skill (tools: manifest={}, outside APIs={})", cfg.tools.manifest.is_some(), cfg.provider.allowed_hosts.len());
        }
    }
    for r in &cfg.remote {
        let bearer = r.bearer_env.as_ref().and_then(|v| std::env::var(v).ok());
        match A2aClient::new(&r.url, bearer) {
            Ok(client) => {
                svc.register_skill(Arc::new(RemoteAgentSkill {
                    local_id: r.skill_id.clone(),
                    remote_skill: r.remote_skill.clone(),
                    client,
                    timeout: Duration::from_secs(cfg.task_timeout_secs),
                }));
                info!("🔗 remote agent skill '{}' -> {}", r.skill_id, r.url);
            }
            Err(e) => warn!("remote agent '{}' not configured: {e}", r.skill_id),
        }
    }

    // The card carries the QUIC endpoint id, so peers can dial this agent.
    if let Some(ep) = network.read().await.endpoint() {
        svc.set_endpoint_id(&ep.id().to_string());
    }

    // --- identity, card, recovery ---
    svc.ensure_delegation().await;
    svc.publish_card().await;
    let resumed = svc.recover();
    if resumed > 0 {
        info!("♻️  resumed {resumed} unfinished agent task(s)");
    }

    // --- attach to the network ---
    let (slot, endpoint, neighbours) = {
        let n = network.read().await;
        (n.agent_slot(), n.endpoint(), n.neighbours_handle())
    };
    let _ = slot.set(svc.clone());
    svc.spawn_program_watcher();
    if let Some(ep) = endpoint {
        match crate::agents::delegate::Delegator::new(svc.clone(), ep.clone(), Some(std::path::Path::new(&config.node.data_dir).join("agents"))) {
            Ok(d) => svc.set_delegator(d),
            Err(e) => warn!("reputation book unreadable, delegation disabled: {e}"),
        }
        svc.spawn_mail_loop(ep.clone());
        let s = svc.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(30));
            loop {
                tick.tick().await;
                let peers: Vec<iroh::EndpointId> = neighbours.read().await.iter().filter_map(|p| p.parse().ok()).collect();
                s.refresh_directory(&ep, peers).await;
            }
        });
    }

    // --- research loop (Step 4) ---
    if cfg.research.enabled {
        match build_research(config, &svc, &evals, &network).await {
            Ok(node) => {
                svc.set_research(node.clone());
                let tick = Duration::from_secs(cfg.research.tick_secs.max(1));
                tokio::spawn(async move {
                    let mut t = tokio::time::interval(tick);
                    loop {
                        t.tick().await;
                        match node.tick(chrono::Utc::now().timestamp()).await {
                            Ok(r) if r != crate::agents::research::TickReport::default() => info!("🔬 research tick: {r:?}"),
                            Ok(_) => {}
                            Err(e) => warn!("research tick failed: {e}"),
                        }
                    }
                });
            }
            Err(e) => warn!("research loop not started: {e}"),
        }
    }

    // --- A2A gateway for outside agents ---
    if cfg.gateway_port != 0 {
        let addr = format!("{}:{}", cfg.gateway_bind, cfg.gateway_port);
        let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| format!("A2A gateway bind {addr}: {e}"))?;
        let scheme = if cfg.tls_cert.is_some() { "https" } else { "http" };
        if cfg.public_url.is_none() {
            svc.set_public_url(&format!("{scheme}://{addr}"));
            svc.publish_card().await;
        }
        let st = GatewayState::new(svc.clone());
        tokio::spawn(async move {
            if let Err(e) = gateway::serve(st, listener).await {
                warn!("A2A gateway stopped: {e}");
            }
        });
        info!("🌐 A2A gateway on {scheme}://{addr}");
    }
    info!("🤖 agents enabled: skills {:?}, agent key {}", svc.skill_ids(), svc.agent_public_key());
    Ok(AgentRuntime { service: svc, evals })
}

async fn build_research(
    config: &Config,
    svc: &Arc<AgentService>,
    evals: &Arc<EvalRegistry>,
    network: &Arc<RwLock<MeshNetwork>>,
) -> Result<Arc<crate::agents::research::ResearchNode>, BoxError> {
    use crate::agents::budget::Budget;
    use crate::agents::llm_roles::{CallBudget, LlmCandidateMaker, LlmProposer};
    use crate::agents::market::CapabilityAd;
    use crate::agents::research::{ArtifactStore, ResearchNode, SandboxRunner};
    use crate::agents::skills::PatchEvaluator;

    let r = &config.agents.research;
    let has = |role: &str| r.roles.iter().any(|x| x == role);
    let project = std::path::PathBuf::from(r.project.clone().ok_or("agents.research.project is required")?);
    let data_dir = std::path::Path::new(&config.node.data_dir).join("agents");

    // Artifacts move over iroh-blobs; providers are the current gossip neighbours.
    let (store, endpoint, neighbours) = {
        let n = network.read().await;
        (n.blob_store(), n.endpoint(), n.neighbours_handle())
    };
    let (store, endpoint) = (store.ok_or("blob store unavailable")?, endpoint.ok_or("endpoint unavailable")?);
    let providers: Arc<dyn Fn() -> Vec<iroh::EndpointId> + Send + Sync> = Arc::new(move || {
        neighbours.try_read().map(|p| p.iter().filter_map(|s| s.parse().ok()).collect()).unwrap_or_default()
    });
    let blobs: Arc<dyn ArtifactStore> = Arc::new(crate::agents::blobs::IrohArtifacts::new(store, &endpoint, providers));

    let caps = CapabilityAd { node_id: svc.node_id.clone(), gpu_class: r.gpu_class.clone(), data_access: r.data_access.clone(), gpu_seconds: r.gpu_seconds };
    let mut node = ResearchNode::new(svc.clone(), caps, blobs, project.clone(), data_dir.join("research-work"));
    node.committee_size = r.committee_size;
    node.min_proposals = r.committee_size;
    node.lease_secs = r.lease_secs;
    node.capacity = r.capacity;
    node.max_generations = r.max_generations;
    node.wait_ticks = r.wait_ticks;
    node.max_item_failures = r.max_item_failures;
    node.merge_cfg.min_supporters = r.min_supporters;
    node.merge_cfg.solo_per_planner = r.solo_per_planner;
    node.merge_cfg.max_items = r.max_agenda_items;
    if !r.allowed_paths.is_empty() {
        node.allowed_paths = Some(r.allowed_paths.clone());
    }
    let guidance = match &r.guidance_file {
        Some(f) => std::fs::read_to_string(f).map_err(|e| format!("guidance_file {f}: {e}"))?,
        None => String::new(),
    };
    // Hard GPU ceiling (halting the node stops it too).
    let gpu = Arc::new(crate::agents::budget::BudgetTracker::new(Budget { tokens: u64::MAX / 4, gpu_seconds: r.gpu_budget_seconds, wall_clock_secs: 10 * 365 * 86_400 }));
    svc.kill.register_budget(&gpu);
    node.gpu_budget = Some(gpu);

    let runtime = crate::agents::sandbox::Runtime::from_name(&r.runtime).ok_or_else(|| format!("no container runtime available for '{}'", r.runtime))?;
    let image = r.image.clone();
    let profile_for = |image: &str| -> Result<crate::agents::sandbox::SandboxProfile, BoxError> {
        let mut p = crate::agents::sandbox::SandboxProfile::new(runtime, image);
        p.gpu = r.gpu;
        p.gpu_devices = r.gpu_devices.clone();
        p.cpus = r.cpus;
        p.memory_mb = r.memory_mb;
        for m in &r.mounts {
            let (host, name) = m.split_once(':').ok_or_else(|| format!("mount '{m}' must be host_path:name"))?;
            p.mounts.push((std::path::PathBuf::from(host), name.to_string()));
        }
        for e in &r.env {
            let (k, v) = e.split_once('=').ok_or_else(|| format!("env '{e}' must be KEY=VALUE"))?;
            p.env.push((k.to_string(), v.to_string()));
        }
        Ok(p)
    };
    if has("executor") || has("planner") {
        let url = config.agents.serve_url.clone().ok_or("planner/executor roles need agents.serve_url")?;
        let script = std::env::var("SLAKSHNA_AGENT_WORKER").unwrap_or_else(|_| "agent_worker.py".into());
        let mut wenv = vec![("SLAKSHNA_SERVE_URL".to_string(), url)];
        if let Some(m) = &config.agents.serve_model {
            wenv.push(("SLAKSHNA_SERVE_MODEL".to_string(), m.clone()));
        }
        let worker = Arc::new(AgentWorker::new(&config.agents.python, &script, wenv));
        let b = &config.agents.task_budget;
        let cb = CallBudget { budget: Budget { tokens: b.tokens, gpu_seconds: b.gpu_seconds, wall_clock_secs: b.wall_clock_secs }, kill: Some(svc.kill.clone()) };
        let cb2 = cb.clone();
        let budget: Arc<dyn Fn() -> Arc<crate::agents::budget::BudgetTracker> + Send + Sync> = Arc::new(move || cb2.tracker());
        if has("planner") {
            node.proposer = Some(Arc::new(LlmProposer { guidance: guidance.clone(), worker: worker.clone(), budget: budget.clone(), max_tokens: 512 }));
        }
        if has("executor") {
            let image = image.clone().ok_or("executor role needs agents.research.image")?;
            if r.run_command.is_empty() {
                return Err("executor role needs agents.research.run_command".into());
            }
            let mut maker = LlmCandidateMaker::new(worker, budget);
            maker.guidance = guidance.clone();
            node.maker = Some(Arc::new(maker));
            node.runner = Some(Arc::new(SandboxRunner { profile: profile_for(&image)?, command: r.run_command.clone(), wall_clock: Duration::from_secs(r.run_wall_clock_secs) }));
        }
    }
    if has("evaluator") {
        let name = r.dataset.clone().ok_or("evaluator role needs agents.research.dataset")?;
        let dataset = evals.dataset(&name).ok_or_else(|| format!("dataset '{name}' is not configured"))?;
        let image = image.ok_or("evaluator role needs agents.research.image")?;
        if r.predict_command.is_empty() {
            return Err("evaluator role needs agents.research.predict_command".into());
        }
        node.evaluator = Some(Arc::new(PatchEvaluator {
            dataset, profile: profile_for(&image)?, command: r.predict_command.clone(), base_project: project, work_root: data_dir.join("eval-work"),
            wall_clock: Duration::from_secs(r.run_wall_clock_secs),
        }));
    }
    if has("auditor") {
        node.check_consensus = true;
        let image = r.image.clone().ok_or("auditor role needs agents.research.image")?;
        if r.predict_command.is_empty() {
            return Err("auditor role needs agents.research.predict_command".into());
        }
        node.auditor = Some(Arc::new(crate::agents::audit::NetworkAuditor {
            svc: svc.clone(), endpoint: endpoint.clone(), profile: profile_for(&image)?, command: r.predict_command.clone(),
            base_project: node.base_project.clone(), work_root: data_dir.join("audit-work"), sample_size: 48, tolerance: 0.02,
        }));
        if let Some(d) = svc.delegator() {
            node.on_audit = Some(Arc::new(move |evaluator, ok| d.record_evaluator_audit(evaluator, ok)));
        }
    }
    info!("🔬 research roles {:?}", r.roles);
    Ok(Arc::new(node))
}
