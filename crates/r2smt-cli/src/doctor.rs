//! Runtime dependency diagnostics and radare2 capability gates.

use std::process::Command;
use std::sync::OnceLock;

use anyhow::Result;
use serde_json::Value;

use crate::args::SolverArg;

const MIN_ESIL_FLAGS_VERSION: (u64, u64, u64) = (6, 2, 0);

#[derive(Debug)]
struct Radare2Probe {
    version: Option<String>,
    detail: String,
}

static RADARE2: OnceLock<Radare2Probe> = OnceLock::new();
static R2GHIDRA: OnceLock<String> = OnceLock::new();
static CVC5: OnceLock<String> = OnceLock::new();
static BITWUZLA: OnceLock<String> = OnceLock::new();
static PORTFOLIO: OnceLock<String> = OnceLock::new();

fn output(program: &str, args: &[&str]) -> std::result::Result<String, String> {
    let result = Command::new(program)
        .args(args)
        .output()
        .map_err(|err| err.to_string())?;
    let stdout = String::from_utf8_lossy(&result.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&result.stderr).trim().to_string();
    if result.status.success() {
        Ok(if stdout.is_empty() { stderr } else { stdout })
    } else {
        Err(if stderr.is_empty() { stdout } else { stderr })
    }
}

fn first_line(value: &str) -> &str {
    value.lines().next().unwrap_or(value).trim()
}

fn parse_radare2_version(value: &str) -> Option<(u64, u64, u64)> {
    let raw = value
        .split_whitespace()
        .find(|token| token.chars().next().is_some_and(|c| c.is_ascii_digit()))?;
    let mut parts = raw.split('.');
    Some((
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
    ))
}

fn radare2() -> &'static Radare2Probe {
    RADARE2.get_or_init(|| match output("radare2", &["-v"]) {
        Ok(detail) => Radare2Probe {
            version: parse_radare2_version(&detail)
                .map(|(major, minor, patch)| format!("{major}.{minor}.{patch}")),
            detail: first_line(&detail).to_string(),
        },
        Err(detail) => Radare2Probe {
            version: None,
            detail,
        },
    })
}

pub(crate) fn radare2_version() -> &'static str {
    radare2().version.as_deref().unwrap_or("unavailable")
}

pub(crate) fn esil_flags_supported() -> bool {
    radare2()
        .version
        .as_deref()
        .and_then(parse_radare2_version)
        .is_some_and(|version| version >= MIN_ESIL_FLAGS_VERSION)
}

pub(crate) fn r2ghidra_version() -> &'static str {
    R2GHIDRA
        .get_or_init(|| {
            let Ok(raw) = output("r2", &["-q", "-N", "-c", "Lcj;q", "malloc://1"]) else {
                return "unavailable".to_string();
            };
            serde_json::from_str::<Vec<Value>>(&raw)
                .ok()
                .and_then(|plugins| {
                    plugins.into_iter().find_map(|plugin| {
                        let name = plugin.get("name")?.as_str()?;
                        name.contains("ghidra").then(|| {
                            plugin
                                .get("version")
                                .and_then(Value::as_str)
                                .unwrap_or("installed (version unavailable)")
                                .to_string()
                        })
                    })
                })
                .unwrap_or_else(|| "unavailable".to_string())
        })
        .as_str()
}

fn executable_version(program: &str) -> String {
    output(program, &["--version"]).map_or_else(
        |_| "unavailable".to_string(),
        |value| first_line(&value).to_string(),
    )
}

pub(crate) fn solver_version(solver: SolverArg) -> &'static str {
    match solver {
        SolverArg::Z3 => r2smt_smt::z3_version(),
        SolverArg::Cvc5 => CVC5.get_or_init(|| executable_version("cvc5")).as_str(),
        SolverArg::Bitwuzla => BITWUZLA
            .get_or_init(|| executable_version("bitwuzla"))
            .as_str(),
        SolverArg::Portfolio => PORTFOLIO
            .get_or_init(|| {
                format!(
                    "z3 {}; cvc5 {}; bitwuzla {}",
                    solver_version(SolverArg::Z3),
                    solver_version(SolverArg::Cvc5),
                    solver_version(SolverArg::Bitwuzla)
                )
            })
            .as_str(),
    }
}

pub(crate) fn doctor() -> Result<()> {
    let r2 = radare2();
    println!("r2smt      {}", env!("CARGO_PKG_VERSION"));
    println!("radare2    {}", r2.detail);
    println!("r2ghidra   {}", r2ghidra_version());
    println!("z3         {}", solver_version(SolverArg::Z3));
    println!("cvc5       {}", solver_version(SolverArg::Cvc5));
    println!("bitwuzla   {}", solver_version(SolverArg::Bitwuzla));
    println!("worker     {}", crate::worker::sandbox_status());
    println!(
        "esil flags {} (requires radare2 >= 6.2.0)",
        if esil_flags_supported() {
            "enabled"
        } else {
            "disabled"
        }
    );
    if r2.version.is_none() {
        anyhow::bail!("radare2 is unavailable or its version could not be parsed");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MIN_ESIL_FLAGS_VERSION, parse_radare2_version};

    #[test]
    fn parses_and_orders_radare2_versions_numerically() {
        assert_eq!(
            parse_radare2_version("radare2 6.2.0 +0 abi:132"),
            Some(MIN_ESIL_FLAGS_VERSION)
        );
        assert!(parse_radare2_version("radare2 6.10.0") > Some(MIN_ESIL_FLAGS_VERSION));
        assert_eq!(parse_radare2_version("not installed"), None);
    }
}
