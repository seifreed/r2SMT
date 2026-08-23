//! Isolated subprocess driver for authoritative sample analysis.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use r2smt_core::AnalysisResult;
use r2smt_slicer::SliceLimits;
use r2smt_smt::SolveOptions;
use serde::{Deserialize, Serialize};

use crate::args::SolverArg;
use crate::support::{analyze_provider_with_caps, attach_pseudocode, open_provider};

pub(crate) const WALL_CLOCK_MS: u64 = 120_000;
pub(crate) const MEMORY_MIB: u64 = 2_048;
pub(crate) const MAX_FUNCTIONS: usize = 100_000;
pub(crate) const MAX_BRANCHES: usize = 1_000_000;
const MAX_LOG_BYTES: u64 = 1_048_576;
const MAX_RESULT_BYTES: u64 = 64 * 1_048_576;
const POLL_MS: u64 = 10;
const MEMORY_POLL_MS: u64 = 100;

pub(crate) fn sandbox_status() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        if Path::new("/usr/bin/sandbox-exec").exists() {
            "sandbox-exec (network denied, read-only host)"
        } else {
            "unavailable"
        }
    }
    #[cfg(target_os = "linux")]
    {
        if ["/usr/bin/bwrap", "/bin/bwrap"]
            .into_iter()
            .any(|path| Path::new(path).exists())
        {
            "bubblewrap (network namespace, read-only host)"
        } else {
            "unavailable (install bubblewrap)"
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        "unavailable on this platform"
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AnalysisWorkerRequest {
    pub(crate) file: PathBuf,
    pub(crate) deep: bool,
    pub(crate) at: Option<String>,
    pub(crate) function_filter: Option<String>,
    pub(crate) limits: SliceLimits,
    pub(crate) options: SolveOptions,
    pub(crate) solver: SolverArg,
    pub(crate) with_decompiler: bool,
    pub(crate) ir_pcode: bool,
}

impl AnalysisWorkerRequest {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        file: &Path,
        deep: bool,
        at: Option<&str>,
        function_filter: Option<&str>,
        limits: &SliceLimits,
        options: SolveOptions,
        solver: SolverArg,
        with_decompiler: bool,
        ir_pcode: bool,
    ) -> Result<Self> {
        Ok(Self {
            file: file.canonicalize()?,
            deep,
            at: at.map(str::to_string),
            function_filter: function_filter.map(str::to_string),
            limits: *limits,
            options,
            solver,
            with_decompiler,
            ir_pcode,
        })
    }
}

pub(crate) fn run_isolated(request: &AnalysisWorkerRequest) -> Result<AnalysisResult> {
    let dir = tempfile::tempdir().context("creating isolated analysis directory")?;
    let request_path = dir.path().join("request.json");
    let result_path = dir.path().join("result.json");
    let stdout_path = dir.path().join("stdout.log");
    let stderr_path = dir.path().join("stderr.log");
    write_json_sync(&request_path, request)?;

    let executable = std::env::current_exe().context("locating r2smt executable")?;
    let args = vec![
        "__analysis-worker".to_string(),
        "--request".to_string(),
        request_path.display().to_string(),
        "--result".to_string(),
        result_path.display().to_string(),
    ];
    let mut command = sandboxed_command(&executable, &args, dir.path())?;
    command
        .current_dir(dir.path())
        .env("HOME", dir.path())
        .env("TMPDIR", dir.path())
        .stdin(Stdio::null())
        .stdout(Stdio::from(File::create(&stdout_path)?))
        .stderr(Stdio::from(File::create(&stderr_path)?));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        command.process_group(0);
    }

    let mut child = command.spawn().context("spawning analysis worker")?;
    let status = wait_worker(
        &mut child,
        &stdout_path,
        &stderr_path,
        WALL_CLOCK_MS,
        MEMORY_MIB,
    )?;
    if !status.success() {
        anyhow::bail!(
            "analysis worker exited with {status}: {}",
            read_bounded(&stderr_path, MAX_LOG_BYTES)?
        );
    }
    if file_len(&result_path) > MAX_RESULT_BYTES {
        anyhow::bail!("analysis worker result exceeded {MAX_RESULT_BYTES} bytes");
    }
    let bytes = fs::read(&result_path).context("reading analysis worker result")?;
    serde_json::from_slice(&bytes).context("parsing analysis worker result")
}

fn wait_worker(
    child: &mut std::process::Child,
    stdout_path: &Path,
    stderr_path: &Path,
    wall_clock_ms: u64,
    memory_mib: u64,
) -> Result<std::process::ExitStatus> {
    let process_group = child.id();
    let deadline = Instant::now() + Duration::from_millis(wall_clock_ms);
    let mut next_memory_check = Instant::now();
    loop {
        if let Some(status) = child.try_wait().context("polling analysis worker")? {
            return Ok(status);
        }
        if file_len(stdout_path) > MAX_LOG_BYTES || file_len(stderr_path) > MAX_LOG_BYTES {
            kill_process_tree(child);
            anyhow::bail!("analysis worker exceeded the stdout/stderr limit");
        }
        if Instant::now() >= next_memory_check {
            let rss_kib = process_group_rss_kib(process_group)?;
            if rss_kib > memory_mib * 1024 {
                kill_process_tree(child);
                anyhow::bail!("analysis worker exceeded the {memory_mib} MiB memory limit");
            }
            next_memory_check = Instant::now() + Duration::from_millis(MEMORY_POLL_MS);
        }
        if Instant::now() >= deadline {
            kill_process_tree(child);
            anyhow::bail!("analysis worker exceeded the {wall_clock_ms} ms wall-clock limit");
        }
        std::thread::sleep(Duration::from_millis(POLL_MS));
    }
}

pub(crate) fn worker_entry(request_path: &Path, result_path: &Path) -> Result<()> {
    if file_len(request_path) > MAX_LOG_BYTES {
        anyhow::bail!("analysis worker request is too large");
    }
    let request: AnalysisWorkerRequest = serde_json::from_slice(&fs::read(request_path)?)
        .context("parsing analysis worker request")?;
    let mut provider = open_provider(&request.file, request.deep)?;
    provider.set_attach_pcode(request.ir_pcode);
    let mut result = analyze_provider_with_caps(
        &mut provider,
        request.at.as_deref(),
        request.function_filter.as_deref(),
        &request.limits,
        request.options,
        request.solver,
        MAX_FUNCTIONS,
        MAX_BRANCHES,
    )?;
    if request.with_decompiler {
        attach_pseudocode(&mut provider, &mut result.findings);
    }
    write_json_sync(result_path, &result)
}

fn write_json_sync(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    let mut file = File::create(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

fn file_len(path: &Path) -> u64 {
    fs::metadata(path).map_or(0, |metadata| metadata.len())
}

fn read_bounded(path: &Path, max: u64) -> Result<String> {
    let mut bytes = Vec::new();
    File::open(path)?.take(max).read_to_end(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).trim().to_string())
}

#[cfg(unix)]
fn kill_process_tree(child: &mut std::process::Child) {
    let process_group = format!("-{}", child.id());
    let _ = Command::new("/bin/kill")
        .args(["-KILL", &process_group])
        .status();
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(not(unix))]
fn kill_process_tree(child: &mut std::process::Child) {
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(target_os = "macos")]
fn sandboxed_command(executable: &Path, args: &[String], temp: &Path) -> Result<Command> {
    let sandbox = Path::new("/usr/bin/sandbox-exec");
    if !sandbox.exists() {
        anyhow::bail!("sandbox-exec is required for network/filesystem isolation");
    }
    let escape = |path: &Path| {
        path.display()
            .to_string()
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
    };
    let writable = escape(temp);
    let canonical = escape(&temp.canonicalize().unwrap_or_else(|_| temp.to_path_buf()));
    let policy = format!(
        "(version 1)(allow default)(deny network*)(deny file-write*)(allow file-write* (subpath \"{writable}\") (subpath \"{canonical}\"))"
    );
    let mut command = Command::new(sandbox);
    command
        .args(["-p", &policy, &executable.display().to_string()])
        .args(args);
    Ok(command)
}

#[cfg(target_os = "linux")]
fn sandboxed_command(executable: &Path, args: &[String], temp: &Path) -> Result<Command> {
    let bwrap = ["/usr/bin/bwrap", "/bin/bwrap"]
        .into_iter()
        .map(Path::new)
        .find(|path| path.exists())
        .ok_or_else(|| {
            anyhow::anyhow!("bubblewrap is required for network/filesystem isolation")
        })?;
    let mut wrapped = vec![
        "--die-with-parent".to_string(),
        "--unshare-net".to_string(),
        "--ro-bind".to_string(),
        "/".to_string(),
        "/".to_string(),
        "--bind".to_string(),
        temp.display().to_string(),
        temp.display().to_string(),
        "--chdir".to_string(),
        temp.display().to_string(),
        executable.display().to_string(),
    ];
    wrapped.extend_from_slice(args);
    let mut command = Command::new(bwrap);
    command.args(wrapped);
    Ok(command)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn sandboxed_command(_executable: &Path, _args: &[String], _temp: &Path) -> Result<Command> {
    anyhow::bail!("analysis worker isolation is unsupported on this platform")
}

#[cfg(unix)]
fn process_group_rss_kib(process_group: u32) -> Result<u64> {
    let output = Command::new("/bin/ps")
        .args(["-o", "rss=", "-g", &process_group.to_string()])
        .output()
        .context("reading analysis worker RSS")?;
    if !output.status.success() {
        return Ok(0);
    }
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .try_fold(0u64, |total, raw| {
            let value = raw
                .parse::<u64>()
                .with_context(|| format!("parsing worker RSS '{raw}'"))?;
            total
                .checked_add(value)
                .ok_or_else(|| anyhow::anyhow!("worker RSS overflow"))
        })
}

#[cfg(not(unix))]
fn process_group_rss_kib(_process_group: u32) -> Result<u64> {
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_reader_never_returns_more_than_the_cap() -> Result<()> {
        let file = tempfile::NamedTempFile::new()?;
        fs::write(file.path(), b"0123456789")?;
        assert_eq!(read_bounded(file.path(), 4)?, "0123");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn watchdog_kills_a_slow_process_group() -> Result<()> {
        use std::os::unix::process::CommandExt as _;

        let dir = tempfile::tempdir()?;
        let stdout = dir.path().join("stdout");
        let stderr = dir.path().join("stderr");
        File::create(&stdout)?;
        File::create(&stderr)?;
        let mut command = Command::new("/bin/sh");
        command.arg("-c").arg("sleep 30 & wait").process_group(0);
        let mut child = command.spawn()?;
        let Err(error) = wait_worker(&mut child, &stdout, &stderr, 50, MEMORY_MIB) else {
            anyhow::bail!("slow worker did not time out");
        };
        assert!(error.to_string().contains("wall-clock limit"));
        Ok(())
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn sandbox_denies_network_sockets() -> Result<()> {
        let python = Path::new("/usr/bin/python3");
        if !python.exists() {
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let mut command = sandboxed_command(
            python,
            &["-c".into(), "import socket; socket.socket()".into()],
            dir.path(),
        )?;
        assert!(!command.status()?.success());
        Ok(())
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn sandbox_denies_writes_outside_worker_directory() -> Result<()> {
        let python = Path::new("/usr/bin/python3");
        if !python.exists() {
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let outside = tempfile::NamedTempFile::new()?;
        let script = format!("open({:?}, 'wb').write(b'changed')", outside.path());
        let mut command = sandboxed_command(python, &["-c".into(), script], dir.path())?;
        assert!(!command.status()?.success());
        assert_eq!(fs::read(outside.path())?, Vec::<u8>::new());
        Ok(())
    }
}
