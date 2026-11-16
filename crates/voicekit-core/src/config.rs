use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub vad: VadConfig,
    pub stt: SttConfig,
    pub llm: LlmConfig,
    pub tts: TtsConfig,
    pub session_policy: SessionPolicy,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VadConfig {
    /// Registry key for the VAD model (e.g. "silero-vad").
    pub model: String,
    pub threshold: f32,
    pub silence_ms: u64,
    pub chunk_samples: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SttConfig {
    pub provider: String,
    /// Registry key for the STT model (e.g. "moonshine-tiny").
    pub model: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String,
    /// Registry key for the LLM model (e.g. "gemma3-1b-q4").
    pub model: String,
    pub system_prompt: String,
    pub context_size: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TtsConfig {
    pub provider: String,
    /// Registry key for the TTS model (e.g. "paradee-int8").
    pub model: String,
    pub voice: String,
    pub sample_rate: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct SessionPolicy {
    pub allow_system_prompt_override: bool,
    pub allow_voice_override: bool,
    pub allow_vad_override: bool,
}

impl Config {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("reading config {:?}", path.as_ref()))?;
        toml::from_str(&raw).context("parsing config.toml")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fixture_config() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/config.toml");
        let cfg = Config::from_file(path).expect("config should parse");
        assert_eq!(cfg.stt.provider, "moonshine");
        assert_eq!(cfg.stt.model, "moonshine-tiny");
        assert_eq!(cfg.llm.provider, "llamacpp");
        assert_eq!(cfg.llm.model, "gemma3-1b-q4");
        assert_eq!(cfg.tts.provider, "paradee");
        assert_eq!(cfg.tts.model, "paradee-int8");
        assert_eq!(cfg.server.port, 8080);
    }
}
