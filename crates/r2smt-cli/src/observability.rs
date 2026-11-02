//! Process-level logging configuration and safety policy.

use std::borrow::Cow;
use std::io::IsTerminal as _;

use anyhow::{Context as _, Result, bail};
use tracing_subscriber::EnvFilter;

const DEFAULT_ENVIRONMENT: &str = "development";
const MAX_LOG_VALUE_CHARS: usize = 4_096;
const MAX_CONTEXT_CHARS: usize = 128;
const REDACTED: &str = "[REDACTED]";
const SENSITIVE_MARKERS: [&str; 10] = [
    "password",
    "passwd",
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "bearer ",
    "cookie",
    "private_key",
    "secret",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LogFormat {
    Compact,
    Json,
}

/// Validated process context attached to the root command span.
pub(crate) struct ProcessContext {
    pub(crate) environment: String,
    pub(crate) correlation_id: Option<String>,
}

struct LogConfig {
    filter: EnvFilter,
    format: LogFormat,
    environment: String,
    correlation_id: Option<String>,
}

impl LogConfig {
    fn from_env(verbosity: u8) -> Result<Self> {
        Self::new(
            verbosity,
            std::env::var("RUST_LOG").ok().as_deref(),
            std::env::var("R2SMT_LOG_FORMAT").ok().as_deref(),
            std::env::var("R2SMT_ENVIRONMENT").ok().as_deref(),
            std::env::var("R2SMT_CORRELATION_ID").ok().as_deref(),
        )
    }

    fn new(
        verbosity: u8,
        rust_log: Option<&str>,
        format: Option<&str>,
        environment: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<Self> {
        let default_level = match verbosity {
            0 => "info",
            1 => "debug",
            _ => "trace",
        };
        let filter = EnvFilter::try_new(rust_log.unwrap_or(default_level))
            .context("invalid RUST_LOG filter")?;
        let format = match format.unwrap_or("compact") {
            "compact" => LogFormat::Compact,
            "json" => LogFormat::Json,
            other => bail!("invalid R2SMT_LOG_FORMAT '{other}'; expected compact or json"),
        };
        let environment = validate_context(
            "R2SMT_ENVIRONMENT",
            environment.unwrap_or(DEFAULT_ENVIRONMENT),
        )?;
        let correlation_id = correlation_id
            .map(|value| validate_context("R2SMT_CORRELATION_ID", value))
            .transpose()?;
        Ok(Self {
            filter,
            format,
            environment,
            correlation_id,
        })
    }
}

/// Install the only global subscriber in the application composition root.
pub(crate) fn init(verbosity: u8) -> Result<ProcessContext> {
    let config = LogConfig::from_env(verbosity)?;
    let ansi = std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    match config.format {
        LogFormat::Compact => tracing_subscriber::fmt()
            .compact()
            .with_ansi(ansi)
            .with_env_filter(config.filter)
            .with_target(true)
            .with_thread_ids(true)
            .with_writer(std::io::stderr)
            .try_init(),
        LogFormat::Json => tracing_subscriber::fmt()
            .json()
            .flatten_event(true)
            .with_current_span(true)
            .with_span_list(true)
            .with_ansi(false)
            .with_env_filter(config.filter)
            .with_target(true)
            .with_thread_ids(true)
            .with_writer(std::io::stderr)
            .try_init(),
    }
    .map_err(|error| anyhow::anyhow!("tracing init failed: {error}"))?;
    Ok(ProcessContext {
        environment: config.environment,
        correlation_id: config.correlation_id,
    })
}

/// Redact likely credentials, bound field size, and neutralize log injection.
pub(crate) fn safe_value(value: &str) -> Cow<'_, str> {
    let lower = value.to_ascii_lowercase();
    if SENSITIVE_MARKERS
        .iter()
        .any(|marker| lower.contains(marker))
    {
        return Cow::Borrowed(REDACTED);
    }
    if value.chars().count() <= MAX_LOG_VALUE_CHARS && !value.chars().any(char::is_control) {
        return Cow::Borrowed(value);
    }

    let mut output = String::with_capacity(value.len().min(MAX_LOG_VALUE_CHARS));
    for character in value.chars().take(MAX_LOG_VALUE_CHARS) {
        match character {
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            control if control.is_control() => output.extend(control.escape_default()),
            other => output.push(other),
        }
    }
    if value.chars().count() > MAX_LOG_VALUE_CHARS {
        output.push_str("…[truncated]");
    }
    Cow::Owned(output)
}

fn validate_context(name: &str, value: &str) -> Result<String> {
    if value.is_empty()
        || value.chars().count() > MAX_CONTEXT_CHARS
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "-._:/".contains(character))
    {
        bail!("{name} must be 1-{MAX_CONTEXT_CHARS} ASCII letters, digits, or one of -._:/");
    }
    Ok(value.to_owned())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn test_config_rejects_unknown_format() {
        let error = LogConfig::new(0, None, Some("xml"), None, None)
            .err()
            .expect("unknown format must fail");
        assert!(error.to_string().contains("R2SMT_LOG_FORMAT"));
    }

    #[test]
    fn test_config_rejects_invalid_filter() {
        assert!(LogConfig::new(0, Some("["), None, None, None).is_err());
    }

    #[test]
    fn test_context_accepts_w3c_compatible_correlation_value() -> Result<()> {
        let config = LogConfig::new(
            0,
            None,
            Some("json"),
            Some("production-eu"),
            Some("00-4bf92f3577b34da6a3ce929d0e0e4736"),
        )?;
        assert_eq!(
            config.correlation_id.as_deref(),
            Some("00-4bf92f3577b34da6a3ce929d0e0e4736")
        );
        Ok(())
    }

    #[test]
    fn test_safe_value_redacts_secret_bearing_text() {
        assert_eq!(safe_value("authorization: Bearer abc123"), REDACTED);
    }

    #[test]
    fn test_safe_value_escapes_log_injection() {
        assert_eq!(
            safe_value("bad\n2026 ERROR forged\r\n"),
            "bad\\n2026 ERROR forged\\r\\n"
        );
    }

    #[test]
    fn test_safe_value_bounds_untrusted_text() {
        let value = "x".repeat(MAX_LOG_VALUE_CHARS + 1);
        assert!(safe_value(&value).ends_with("…[truncated]"));
    }
}
