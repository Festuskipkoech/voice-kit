use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

use voicekit_core::config::Config;
use voicekit_core::registry::{ModelEntry, ModelRegistry};
use voicekit_core::server::ResolvedPaths;

const REGISTRY_PATH: &str = "/etc/voicekit/models.toml";
const MODELS_ROOT: &str = "/models";

#[derive(Parser)]
#[command(name = "voicekit-server", version, about = "Voicekit WebSocket voice-agent server")]
struct Cli {
    #[arg(long, default_value = "/data/config.toml")]
    config: String,

    #[arg(long, default_value = REGISTRY_PATH)]
    registry: String,

    #[arg(long, default_value = MODELS_ROOT)]
    models_dir: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "voicekit_server=info,voicekit_core=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    let config = Arc::new(Config::from_file(&cli.config).context("loading config")?);
    info!(path = %cli.config, "config loaded");

    let registry = ModelRegistry::from_file(&cli.registry).context("loading model registry")?;
    let models_root = Path::new(&cli.models_dir);

    let paths = ResolvedPaths {
        // STT and TTS engines receive a directory; they join sub-paths internally.
        stt: resolve_dir(&registry, &config.stt.model, models_root, "stt")?,
        tts: resolve_dir(&registry, &config.tts.model, models_root, "tts")?,
        // LLM and VAD engines receive the specific model file, not the directory.
        llm: resolve_file(&registry, &config.llm.model, models_root, "llm")?,
        vad: resolve_file(&registry, &config.vad.model, models_root, "vad")?,
    };

    info!(
        stt = %paths.stt.display(),
        llm = %paths.llm.display(),
        tts = %paths.tts.display(),
        vad = %paths.vad.display(),
        "model paths resolved"
    );

    voicekit_core::server::run(config, paths).await
}

/// Resolve a registry entry whose engine takes the model directory (STT, TTS).
fn resolve_dir(
    registry: &ModelRegistry,
    key: &str,
    models_root: &Path,
    label: &str,
) -> Result<PathBuf> {
    let path = registry
        .resolve(key, models_root)
        .with_context(|| format!("resolving {label} model '{key}' in registry"))?;
    anyhow::ensure!(
        path.exists(),
        "{label} model directory not found: {}\n\
         Check that '{key}' is correct in config.toml and models have been downloaded.",
        path.display()
    );
    Ok(path)
}

/// Resolve a registry entry whose engine takes a single model file (LLM, VAD).
/// Returns `dir.join(files[0])` from the registry entry.
fn resolve_file(
    registry: &ModelRegistry,
    key: &str,
    models_root: &Path,
    label: &str,
) -> Result<PathBuf> {
    let entry: &ModelEntry = registry
        .models
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("{label} model '{key}' not found in registry"))?;

    let file = entry
        .files
        .first()
        .ok_or_else(|| anyhow::anyhow!("{label} model '{key}' has no files listed in registry"))?;

    let path = entry.abs_path(models_root).join(file);

    anyhow::ensure!(
        path.exists(),
        "{label} model file not found: {}\n\
         Check that '{key}' is correct in config.toml and models have been downloaded.",
        path.display()
    );

    Ok(path)
}