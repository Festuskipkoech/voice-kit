pub mod llm;
pub mod stt;
pub mod tts;

use std::path::Path;
use std::sync::OnceLock;

use anyhow::Result;
use ort::environment::Environment;
use tokio::sync::mpsc::Sender;

static ORT_ENV: OnceLock<std::result::Result<Environment, String>> = OnceLock::new();

pub(crate) fn ort_env() -> Result<&'static Environment> {
    ORT_ENV
        .get_or_init(|| ort::init().build().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| anyhow::anyhow!("ort environment init: {e}"))
}

pub trait SttEngine: Send + Sync {
    fn transcribe(&self, audio: &[f32]) -> Result<String>;
}

pub trait LlmEngine: Send + Sync {
    fn generate(&self, prompt: &str, tx: Sender<String>) -> Result<()>;
}

pub trait TtsEngine: Send + Sync {
    fn synthesize(&self, text: &str, voice: &str) -> Result<Vec<f32>>;
}

use crate::config::{LlmConfig, SttConfig, TtsConfig, VadConfig};
use crate::vad::{Vad, VadEngine};

pub fn build_stt(cfg: &SttConfig, model_path: &Path) -> Result<Box<dyn SttEngine>> {
    match cfg.provider.as_str() {
        "moonshine" => Ok(Box::new(stt::moonshine::MoonshineEngine::new(model_path)?)),
        other => anyhow::bail!("unknown stt provider: {other}"),
    }
}

pub fn build_llm(cfg: &LlmConfig, model_path: &Path) -> Result<Box<dyn LlmEngine>> {
    match cfg.provider.as_str() {
        "llamacpp" => Ok(Box::new(llm::llamacpp::LlamaCppEngine::new(
            model_path, cfg,
        )?)),
        other => anyhow::bail!("unknown llm provider: {other}"),
    }
}

pub fn build_tts(cfg: &TtsConfig, model_path: &Path) -> Result<Box<dyn TtsEngine>> {
    match cfg.provider.as_str() {
        "paradee" => Ok(Box::new(tts::paradee::ParadeeEngine::new(model_path, cfg)?)),
        other => anyhow::bail!("unknown tts provider: {other}"),
    }
}

pub fn build_vad(cfg: &VadConfig, model_path: &Path) -> Result<Box<dyn VadEngine>> {
    Ok(Box::new(Vad::new(
        model_path,
        cfg.threshold,
        cfg.silence_ms,
    )?))
}

pub fn build_vad_with_overrides(
    model_path: &Path,
    threshold: f32,
    silence_ms: u64,
) -> Result<Box<dyn VadEngine>> {
    Ok(Box::new(Vad::new(model_path, threshold, silence_ms)?))
}
