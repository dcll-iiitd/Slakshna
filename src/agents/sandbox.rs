//! Rootless sandbox executor (A1, A7).
//!
//! Candidates run in Podman (Apptainer under SLURM) with: no network, a
//! read-only root filesystem, all capabilities dropped, no-new-privileges,
//! CPU/memory/pid caps, one private worktree mounted read-write, inputs
//! mounted read-only, and a hard wall-clock kill. Command construction is a
//! pure function so the isolation flags are unit-tested even on hosts where
//! no container runtime is installed.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Runtime {
    Podman,
    Apptainer,
    /// Docker (rootful daemon): the container runs as the calling user, not root.
    Docker,
}

impl Runtime {
    pub fn binary(self) -> &'static str {
        match self {
            Runtime::Podman => "podman",
            Runtime::Apptainer => "apptainer",
            Runtime::Docker => "docker",
        }
    }

    /// Pick a runtime by name ("auto" prefers Podman, then Docker, then Apptainer).
    pub fn from_name(name: &str) -> Option<Runtime> {
        match name {
            "podman" => Some(Runtime::Podman),
            "docker" => Some(Runtime::Docker),
            "apptainer" => Some(Runtime::Apptainer),
            "auto" | "" => [Runtime::Podman, Runtime::Docker, Runtime::Apptainer].into_iter().find(|r| r.available()),
            _ => None,
        }
    }

    pub fn available(self) -> bool {
        std::env::var_os("PATH")
            .map(|p| std::env::split_paths(&p).any(|d| d.join(self.binary()).is_file()))
            .unwrap_or(false)
    }
}

/// How an operator runs candidate code: runtime, image, GPUs, read-only mounts, environment, caps.
/// Shared by the executor, evaluator and auditor roles so all of them sandbox identically.
#[derive(Debug, Clone)]
pub struct SandboxProfile {
    pub runtime: Runtime,
    pub image: String,
    pub gpu: bool,
    pub gpu_devices: Option<String>,
    /// Read-only mounts: (host path, name under /inputs).
    pub mounts: Vec<(PathBuf, String)>,
    pub env: Vec<(String, String)>,
    pub cpus: u32,
    pub memory_mb: u64,
}

impl SandboxProfile {
    pub fn new(runtime: Runtime, image: &str) -> Self {
        SandboxProfile { runtime, image: image.into(), gpu: false, gpu_devices: None, mounts: vec![], env: vec![], cpus: 4, memory_mb: 16 * 1024 }
    }

    pub fn spec(&self, worktree: &Path, command: Vec<String>, wall_clock: Duration) -> SandboxSpec {
        let mut s = SandboxSpec::new(self.runtime, &self.image, worktree, command, wall_clock);
        s.inputs = self.mounts.clone();
        s.env = self.env.clone();
        s.gpu = self.gpu;
        s.gpu_devices = self.gpu_devices.clone();
        s.cpus = self.cpus;
        s.memory_mb = self.memory_mb;
        s
    }
}

#[derive(Debug, Clone)]
pub struct SandboxSpec {
    pub runtime: Runtime,
    /// Image reference (Podman) or .sif path (Apptainer).
    pub image: String,
    /// Private worktree of this candidate, mounted read-write at /work.
    pub worktree: PathBuf,
    /// Read-only inputs: (host path, path inside /inputs).
    pub inputs: Vec<(PathBuf, String)>,
    pub command: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cpus: u32,
    pub memory_mb: u64,
    pub pids: u32,
    pub gpu: bool,
    /// Specific GPU indices (e.g. "0,1"); with `gpu` and None, all GPUs are visible.
    pub gpu_devices: Option<String>,
    /// Egress is off unless a policy engine explicitly turns it on.
    pub allow_network: bool,
    pub wall_clock: Duration,
}

impl SandboxSpec {
    pub fn new(runtime: Runtime, image: &str, worktree: &Path, command: Vec<String>, wall_clock: Duration) -> Self {
        SandboxSpec {
            runtime,
            image: image.to_string(),
            worktree: worktree.to_path_buf(),
            inputs: vec![],
            command,
            env: vec![],
            cpus: 2,
            memory_mb: 4096,
            pids: 256,
            gpu: false,
            gpu_devices: None,
            allow_network: false,
            wall_clock,
        }
    }

    /// Full argv (binary first).
    pub fn argv(&self, container_name: &str) -> Vec<String> {
        let s = |x: &str| x.to_string();
        let mut a: Vec<String> = Vec::new();
        match self.runtime {
            Runtime::Podman => {
                a.extend([s("podman"), s("run"), s("--rm"), s("--name"), container_name.to_string()]);
                if !self.allow_network {
                    a.push(s("--network=none"));
                }
                a.extend([
                    s("--read-only"),
                    s("--cap-drop=ALL"),
                    s("--security-opt=no-new-privileges"),
                    format!("--cpus={}", self.cpus),
                    format!("--memory={}m", self.memory_mb),
                    format!("--pids-limit={}", self.pids),
                    s("--tmpfs=/tmp:rw,noexec,nosuid,size=256m"),
                    s("--workdir=/work"),
                    format!("--volume={}:/work:rw", self.worktree.display()),
                ]);
                for (host, inside) in &self.inputs {
                    a.push(format!("--volume={}:/inputs/{}:ro", host.display(), inside));
                }
                if self.gpu {
                    a.push(s("--device=nvidia.com/gpu=all"));
                }
                for (k, v) in &self.env {
                    a.push(format!("--env={k}={v}"));
                }
                a.push(self.image.clone());
                a.extend(self.command.iter().cloned());
            }
            Runtime::Docker => {
                a.extend([s("docker"), s("run"), s("--rm"), s("--name"), container_name.to_string()]);
                if !self.allow_network {
                    a.push(s("--network=none"));
                }
                a.extend([
                    s("--read-only"),
                    s("--cap-drop=ALL"),
                    s("--security-opt=no-new-privileges"),
                    format!("--cpus={}", self.cpus),
                    format!("--memory={}m", self.memory_mb),
                    format!("--pids-limit={}", self.pids),
                    s("--tmpfs=/tmp:rw,noexec,nosuid,size=256m"),
                    s("--workdir=/work"),
                    format!("--user={}", current_uid_gid()),
                    format!("--volume={}:/work:rw", self.worktree.display()),
                ]);
                for (host, inside) in &self.inputs {
                    a.push(format!("--volume={}:/inputs/{}:ro", host.display(), inside));
                }
                if self.gpu {
                    a.push(match &self.gpu_devices {
                        Some(d) => format!("--gpus=\"device={d}\""),
                        None => s("--gpus=all"),
                    });
                }
                for (k, v) in &self.env {
                    a.push(format!("--env={k}={v}"));
                }
                a.push(self.image.clone());
                a.extend(self.command.iter().cloned());
            }
            Runtime::Apptainer => {
                a.extend([s("apptainer"), s("exec"), s("--containall"), s("--no-home"), s("--cleanenv")]);
                if !self.allow_network {
                    // Drop all namespaces' network access.
                    a.extend([s("--net"), s("--network=none")]);
                }
                a.extend([s("--pwd"), s("/work"), s("--bind"), format!("{}:/work:rw", self.worktree.display())]);
                for (host, inside) in &self.inputs {
                    a.extend([s("--bind"), format!("{}:/inputs/{}:ro", host.display(), inside)]);
                }
                if self.gpu {
                    a.push(s("--nv"));
                }
                for (k, v) in &self.env {
                    a.extend([s("--env"), format!("{k}={v}")]);
                }
                a.push(self.image.clone());
                a.extend(self.command.iter().cloned());
            }
        }
        a
    }
}

#[derive(Debug)]
pub struct SandboxResult {
    pub exit_code: Option<i32>,
    #[allow(dead_code)] // library surface: used by tests and integrators, or pending the settlement-chain decision
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
    /// Killed for writing more than the capture cap to stdout/stderr.
    pub flooded: bool,
}

/// Cap on captured output per stream, so a hostile candidate cannot exhaust
/// host memory by printing.
const MAX_CAPTURE: u64 = 4 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error("container runtime '{0}' not found on PATH")]
    RuntimeMissing(&'static str),
    #[error("spawn failed: {0}")]
    Spawn(std::io::Error),
}

/// Run the spec to completion or until the wall clock expires, killing the
/// container (by name, since killing the client alone would leave it running).
pub async fn run(spec: &SandboxSpec, container_name: &str) -> Result<SandboxResult, SandboxError> {
    if !spec.runtime.available() {
        return Err(SandboxError::RuntimeMissing(spec.runtime.binary()));
    }
    let argv = spec.argv(container_name);
    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(SandboxError::Spawn)?;

    // Read each stream up to the cap. A candidate that keeps writing past it is flooding us:
    // it is killed on the spot rather than left blocked on a full pipe until the wall clock.
    let overflow = std::sync::Arc::new(tokio::sync::Notify::new());
    let reader = |mut stream: Box<dyn tokio::io::AsyncRead + Unpin + Send>, overflow: std::sync::Arc<tokio::sync::Notify>| {
        tokio::spawn(async move {
            let mut buf = Vec::new();
            let mut chunk = vec![0u8; 64 * 1024];
            loop {
                match stream.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        buf.extend_from_slice(&chunk[..n]);
                        if buf.len() as u64 > MAX_CAPTURE {
                            buf.truncate(MAX_CAPTURE as usize);
                            overflow.notify_one();
                            break;
                        }
                    }
                }
            }
            buf
        })
    };
    let out_h = reader(Box::new(child.stdout.take().unwrap()), overflow.clone());
    let err_h = reader(Box::new(child.stderr.take().unwrap()), overflow.clone());

    enum End {
        Exited(Option<i32>),
        TimedOut,
        Flooded,
    }
    let end = tokio::select! {
        st = child.wait() => End::Exited(st.ok().and_then(|s| s.code())),
        _ = tokio::time::sleep(spec.wall_clock) => End::TimedOut,
        _ = overflow.notified() => End::Flooded,
    };
    if !matches!(end, End::Exited(_)) {
        kill_container(spec.runtime, container_name).await;
        let _ = child.start_kill();
        let _ = child.wait().await;
    }
    let stdout = out_h.await.unwrap_or_default();
    let stderr = err_h.await.unwrap_or_default();
    Ok(match end {
        End::Exited(code) => SandboxResult { exit_code: code, stdout: String::from_utf8_lossy(&stdout).into_owned(), stderr: String::from_utf8_lossy(&stderr).into_owned(), timed_out: false, flooded: false },
        End::TimedOut => SandboxResult { exit_code: None, stdout: String::new(), stderr: String::new(), timed_out: true, flooded: false },
        End::Flooded => SandboxResult { exit_code: None, stdout: String::from_utf8_lossy(&stdout).into_owned(), stderr: String::from_utf8_lossy(&stderr).into_owned(), timed_out: false, flooded: true },
    })
}

async fn kill_container(rt: Runtime, name: &str) {
    match rt {
        Runtime::Podman => { let _ = Command::new("podman").args(["kill", name]).output().await; }
        Runtime::Docker => { let _ = Command::new("docker").args(["kill", name]).output().await; }
        Runtime::Apptainer => {}
    }
}

/// `uid:gid` of this process, so files the container writes into its worktree belong to us.
fn current_uid_gid() -> String {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |key: &str| status.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)).unwrap_or("1000").to_string();
    format!("{}:{}", field("Uid:"), field("Gid:"))
}

/// One private worktree per candidate: a `git worktree` when the base is a
/// git repository, otherwise a plain recursive copy. The directory is named by
/// the candidate id so a retry reuses a clean tree.
pub fn create_worktree(base: &Path, root: &Path, candidate_id: &str) -> std::io::Result<PathBuf> {
    if candidate_id.is_empty() || !candidate_id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "unsafe candidate id"));
    }
    std::fs::create_dir_all(root)?;
    let dest = root.join(candidate_id);
    if dest.exists() {
        std::fs::remove_dir_all(&dest)?;
    }
    if base.join(".git").exists() {
        let st = std::process::Command::new("git")
            .arg("-C").arg(base)
            .args(["worktree", "add", "--detach"]).arg(&dest)
            .stdout(Stdio::null()).stderr(Stdio::null())
            .status()?;
        if st.success() {
            return Ok(dest);
        }
        let _ = std::process::Command::new("git").arg("-C").arg(base).args(["worktree", "prune"]).status();
    }
    copy_dir(base, &dest)?;
    Ok(dest)
}

fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == ".git" {
            continue;
        }
        let ty = entry.file_type()?;
        let to = dst.join(&name);
        if ty.is_dir() {
            copy_dir(&entry.path(), &to)?;
        } else if ty.is_file() {
            std::fs::copy(entry.path(), &to)?;
        } // symlinks are skipped: they could point outside the tree
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(rt: Runtime) -> SandboxSpec {
        let mut s = SandboxSpec::new(rt, "img:1", Path::new("/wt/c1"), vec!["python".into(), "run.py".into()], Duration::from_secs(5));
        s.inputs.push((PathBuf::from("/data/in"), "in".into()));
        s.env.push(("SEED".into(), "7".into()));
        s
    }

    #[test]
    fn podman_is_locked_down_by_default() {
        let a = spec(Runtime::Podman).argv("n");
        for flag in ["--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges"] {
            assert!(a.iter().any(|x| x == flag), "missing {flag}: {a:?}");
        }
        assert!(a.contains(&"--volume=/wt/c1:/work:rw".to_string()));
        assert!(a.contains(&"--volume=/data/in:/inputs/in:ro".to_string()));
        // image comes right before the command, after every option
        let img = a.iter().position(|x| x == "img:1").unwrap();
        assert_eq!(&a[img + 1..], &["python", "run.py"]);
        assert!(a[..img].iter().all(|x| x.starts_with('-') || x == "podman" || x == "run" || x == "n"));
    }

    #[test]
    fn network_only_when_policy_allows_and_gpu_opt_in() {
        let mut s = spec(Runtime::Podman);
        s.allow_network = true;
        assert!(!s.argv("n").iter().any(|x| x.starts_with("--network")));
        assert!(!s.argv("n").iter().any(|x| x.contains("nvidia")));
        s.gpu = true;
        assert!(s.argv("n").iter().any(|x| x.contains("nvidia")));
    }

    #[test]
    fn apptainer_is_contained_and_inputs_readonly() {
        let a = spec(Runtime::Apptainer).argv("n");
        for flag in ["--containall", "--no-home", "--cleanenv", "--network=none"] {
            assert!(a.iter().any(|x| x == flag), "missing {flag}");
        }
        assert!(a.contains(&"/data/in:/inputs/in:ro".to_string()));
    }

    #[test]
    fn worktree_is_private_per_candidate_and_rejects_bad_ids() {
        let tmp = tempdir();
        let base = tmp.join("base");
        std::fs::create_dir_all(base.join("sub")).unwrap();
        std::fs::write(base.join("sub/a.txt"), "hello").unwrap();
        let root = tmp.join("wts");
        let w1 = create_worktree(&base, &root, "cand1").unwrap();
        let w2 = create_worktree(&base, &root, "cand2").unwrap();
        std::fs::write(w1.join("sub/a.txt"), "changed").unwrap();
        assert_eq!(std::fs::read_to_string(w2.join("sub/a.txt")).unwrap(), "hello");
        assert_eq!(std::fs::read_to_string(base.join("sub/a.txt")).unwrap(), "hello");
        assert!(create_worktree(&base, &root, "../escape").is_err());
        assert!(create_worktree(&base, &root, "").is_err());
    }

    #[test]
    fn git_base_uses_git_worktree() {
        let tmp = tempdir();
        let base = tmp.join("repo");
        std::fs::create_dir_all(&base).unwrap();
        let git = |args: &[&str]| std::process::Command::new("git").arg("-C").arg(&base).args(args).output().unwrap();
        git(&["init", "-q"]);
        std::fs::write(base.join("f.txt"), "x").unwrap();
        git(&["add", "."]);
        git(&["-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "i"]);
        let w = create_worktree(&base, &tmp.join("wts"), "c1").unwrap();
        assert_eq!(std::fs::read_to_string(w.join("f.txt")).unwrap(), "x");
        assert!(w.join(".git").exists());
    }

    #[tokio::test]
    async fn missing_runtime_is_reported_not_faked() {
        // Whatever the host has, the contract is: run() either really runs the
        // container or says the runtime is missing. It never pretends.
        let s = spec(Runtime::Apptainer);
        if !Runtime::Apptainer.available() {
            assert!(matches!(run(&s, "n").await, Err(SandboxError::RuntimeMissing("apptainer"))));
        }
    }

    // ---- real containers (Docker). Skipped, loudly, where Docker or the image is missing. ----

    fn docker_image_present(image: &str) -> bool {
        Runtime::Docker.available()
            && std::process::Command::new("docker").args(["image", "inspect", image]).stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false)
    }

    macro_rules! need_image {
        ($img:expr) => {
            if !docker_image_present($img) {
                eprintln!("SKIPPED: docker image {} not present on this host", $img);
                return;
            }
        };
    }

    fn dspec(image: &str, wt: &Path, cmd: &[&str], secs: u64) -> SandboxSpec {
        SandboxSpec::new(Runtime::Docker, image, wt, cmd.iter().map(|s| s.to_string()).collect(), Duration::from_secs(secs))
    }

    #[tokio::test]
    async fn docker_container_runs_as_the_caller_with_only_its_worktree_writable() {
        need_image!("alpine:latest");
        let wt = tempdir();
        let inp = tempdir();
        std::fs::write(inp.join("data.txt"), "secret-input").unwrap();
        let mut spec = dspec("alpine:latest", &wt, &["sh", "-c", "id -u > /work/uid; cat /inputs/in/data.txt > /work/copy; touch /inputs/in/x 2>/dev/null && echo INPUT_WRITABLE >> /work/flags; touch /rootfs-test 2>/dev/null && echo ROOT_WRITABLE >> /work/flags; touch /tmp/ok && echo TMP_OK >> /work/flags; echo done"], 30);
        spec.inputs.push((inp.clone(), "in".into()));
        let r = run(&spec, &format!("slk-t-{}", rand::random::<u32>())).await.unwrap();
        assert_eq!((r.exit_code, r.timed_out), (Some(0), false), "{r:?}");
        assert_eq!(r.stdout.trim(), "done");
        let my_uid = current_uid_gid().split(':').next().unwrap().to_string();
        assert_eq!(std::fs::read_to_string(wt.join("uid")).unwrap().trim(), my_uid, "container is not root");
        assert_eq!(std::fs::read_to_string(wt.join("copy")).unwrap(), "secret-input", "read-only inputs are readable");
        let flags = std::fs::read_to_string(wt.join("flags")).unwrap_or_default();
        assert!(!flags.contains("INPUT_WRITABLE") && !flags.contains("ROOT_WRITABLE"), "isolation broken: {flags}");
        assert!(flags.contains("TMP_OK"));
        // files it wrote are ours (removable without sudo)
        std::fs::remove_dir_all(&wt).unwrap();
    }

    #[tokio::test]
    async fn docker_container_has_no_network_unless_policy_allows() {
        need_image!("alpine:latest");
        let wt = tempdir();
        let probe = ["sh", "-c", "wget -q -T 4 -O /dev/null http://1.1.1.1/ && echo REACHED || echo BLOCKED"];
        let r = run(&dspec("alpine:latest", &wt, &probe, 30), &format!("slk-n-{}", rand::random::<u32>())).await.unwrap();
        assert_eq!(r.stdout.trim(), "BLOCKED", "egress must be off by default: {r:?}");
        let mut open = dspec("alpine:latest", &wt, &probe, 30);
        open.allow_network = true;
        let r = run(&open, &format!("slk-n2-{}", rand::random::<u32>())).await.unwrap();
        assert_eq!(r.stdout.trim(), "REACHED", "the same probe reaches out once policy enables the network: {r:?}");
    }

    #[tokio::test]
    async fn docker_wall_clock_kills_the_container_not_just_the_client() {
        need_image!("alpine:latest");
        let wt = tempdir();
        let name = format!("slk-k-{}", rand::random::<u32>());
        let t0 = std::time::Instant::now();
        let r = run(&dspec("alpine:latest", &wt, &["sleep", "120"], 3), &name).await.unwrap();
        assert!(r.timed_out && r.exit_code.is_none());
        assert!(t0.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(500)).await;
        let ps = std::process::Command::new("docker").args(["ps", "-q", "--filter", &format!("name={name}")]).output().unwrap();
        assert!(String::from_utf8_lossy(&ps.stdout).trim().is_empty(), "container must be gone after the timeout");
        // failing commands report their exit code
        let r = run(&dspec("alpine:latest", &wt, &["sh", "-c", "echo oops >&2; exit 7"], 30), &format!("slk-e-{}", rand::random::<u32>())).await.unwrap();
        assert_eq!(r.exit_code, Some(7));
        assert!(r.stderr.contains("oops"));
    }

    #[tokio::test]
    async fn docker_output_is_capped_so_a_noisy_candidate_cannot_exhaust_memory() {
        need_image!("alpine:latest");
        let wt = tempdir();
        let t0 = std::time::Instant::now();
        let r = run(&dspec("alpine:latest", &wt, &["sh", "-c", "yes AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA | head -c 200000000"], 60), &format!("slk-c-{}", rand::random::<u32>())).await.unwrap();
        assert!(t0.elapsed() < Duration::from_secs(30), "a flooding candidate is killed promptly, not at the wall clock");
        assert!(r.flooded && r.exit_code.is_none(), "{r:?}");
        assert!(r.stdout.len() as u64 <= MAX_CAPTURE, "captured {} bytes", r.stdout.len());
    }

    /// The point of the whole exercise: a sandboxed candidate can use the GPUs.
    #[tokio::test]
    async fn docker_container_sees_exactly_the_gpus_it_is_given() {
        need_image!("nvidia/cuda:12.1.1-base-ubuntu22.04");
        let wt = tempdir();
        let mut all = dspec("nvidia/cuda:12.1.1-base-ubuntu22.04", &wt, &["nvidia-smi", "-L"], 60);
        all.gpu = true;
        let r = run(&all, &format!("slk-g-{}", rand::random::<u32>())).await.unwrap();
        let n_all = r.stdout.lines().filter(|l| l.starts_with("GPU ")).count();
        assert!(n_all >= 1, "no GPU visible with gpu=true: {r:?}");
        let mut one = dspec("nvidia/cuda:12.1.1-base-ubuntu22.04", &wt, &["nvidia-smi", "-L"], 60);
        one.gpu = true;
        one.gpu_devices = Some("0".into());
        let r = run(&one, &format!("slk-g1-{}", rand::random::<u32>())).await.unwrap();
        assert_eq!(r.stdout.lines().filter(|l| l.starts_with("GPU ")).count(), 1, "device pinning: {r:?}");
        // and with gpu=false the container sees none
        let none = dspec("nvidia/cuda:12.1.1-base-ubuntu22.04", &wt, &["sh", "-c", "nvidia-smi -L || echo NO_GPU"], 60);
        let r = run(&none, &format!("slk-g0-{}", rand::random::<u32>())).await.unwrap();
        assert!(r.stdout.contains("NO_GPU") || !r.stdout.contains("GPU 0"), "{r:?}");
        println!("GPUs visible in container with gpu=true: {n_all}");
    }

    pub fn tempdir() -> PathBuf {
        let p = std::env::temp_dir().join(format!("slk-{}-{}", std::process::id(), rand::random::<u64>()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
