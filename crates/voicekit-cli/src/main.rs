// voicekit-cli/src/main.rs
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;

use voicekit_core::config::Config;
use voicekit_core::registry::ModelRegistry;
use voicekit_core::server::ResolvedPaths;

const CONFIG_FILE: &str = "voicekit.toml";
const CONFIG_TEMPLATE: &str = "config.template.toml";
const REGISTRY_FILE: &str = "models.toml";
const MODELS_DIR: &str = "models";

#[derive(Parser)]
#[command(name = "voicekit", version, about = "Voicekit voice-agent server")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create voicekit.toml and exit - edit it before starting
    Setup,

    /// Start the server (reads voicekit.toml + models.toml)
    Start {
        /// Write default config and start immediately without editing
        #[arg(long)]
        defaults: bool,
    },
}

#[tokio::main]
async fn main() {
    let result = match Cli::parse().command {
        Commands::Setup => cmd_setup(),
        Commands::Start { defaults } => cmd_start(defaults).await,
    };

    if let Err(e) = result {
        eprintln!("{} {:#}", "error:".red().bold(), e);
        std::process::exit(1);
    }
}

fn cmd_setup() -> Result<()> {
    let dst = PathBuf::from(CONFIG_FILE);

    if dst.exists() {
        eprint!(
            "{} voicekit.toml already exists. Overwrite? [y/N] ",
            "warn:".yellow().bold()
        );
        let mut answer = String::new();
        std::io::stdin()
            .read_line(&mut answer)
            .context("failed to read stdin")?;
        if !answer.trim().eq_ignore_ascii_case("y") {
            println!("{} Keeping existing config.", "info:".cyan().bold());
            return Ok(());
        }
    }

    copy_template(&dst)?;

    println!("{} Created voicekit.toml from config.template.toml", "ok:".green().bold());
    println!("    1. Edit voicekit.toml - set your preferred voice, VAD threshold, etc.");
    println!("    2. Run:  voicekit start");

    Ok(())
}

async fn cmd_start(defaults: bool) -> Result<()> {
    let config_path = PathBuf::from(CONFIG_FILE);
    let registry_path = PathBuf::from(REGISTRY_FILE);
    let models_root = PathBuf::from(MODELS_DIR);

    if defaults {
        copy_template(&config_path)?;
        println!("{} Wrote default voicekit.toml from config.template.toml", "ok:".green().bold());
    }

    if !config_path.exists() {
        bail!("voicekit.toml not found - run `voicekit setup` to create one.");
    }

    if !registry_path.exists() {
        bail!(
            "models.toml not found - place it alongside voicekit.toml \
             or run `scripts/download_models.py` to fetch models and generate it."
        );
    }

    let config: Config =
        toml::from_str(&std::fs::read_to_string(&config_path).context("failed to read voicekit.toml")?)
            .context("voicekit.toml is invalid")?;

    let registry = ModelRegistry::from_file(&registry_path)?;

    let paths = resolve_paths(&config, &registry, &models_root)?;

    println!(
        "{} Starting on {}:{}",
        "ok:".green().bold(),
        config.server.host,
        config.server.port
    );

    voicekit_core::server::run(Arc::new(config), paths).await
}

/// Resolve all four model paths from the registry, verify they exist on disk,
/// and return a `ResolvedPaths` ready to pass to `server::run`.
///
/// STT and TTS engines take a directory; LLM and VAD engines take a specific
/// file (the first entry in the registry's `files` list).
fn resolve_paths(
    cfg: &Config,
    registry: &ModelRegistry,
    models_root: &Path,
) -> Result<ResolvedPaths> {
    let stt = {
        let path = registry
            .resolve(&cfg.stt.model, models_root)
            .with_context(|| format!("resolving stt model '{}'", cfg.stt.model))?;
        anyhow::ensure!(
            path.exists(),
            "stt model directory not found: {} \
             (run scripts/download_models.py to fetch it)",
            path.display()
        );
        path
    };

    let tts = {
        let path = registry
            .resolve(&cfg.tts.model, models_root)
            .with_context(|| format!("resolving tts model '{}'", cfg.tts.model))?;
        anyhow::ensure!(
            path.exists(),
            "tts model directory not found: {} \
             (run scripts/download_models.py to fetch it)",
            path.display()
        );
        path
    };

    let llm = resolve_single_file(registry, &cfg.llm.model, models_root, "llm")?;
    let vad = resolve_single_file(registry, &cfg.vad.model, models_root, "vad")?;

    Ok(ResolvedPaths { stt, llm, tts, vad })
}

/// Resolve a registry entry that represents a single model file (LLM, VAD).
/// Returns `dir.join(files[0])` and asserts it exists on disk.
fn resolve_single_file(
    registry: &ModelRegistry,
    key: &str,
    models_root: &Path,
    label: &str,
) -> Result<PathBuf> {
    let entry = registry
        .models
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("{label} model '{key}' not found in models.toml"))?;

    let file = entry
        .files
        .first()
        .ok_or_else(|| anyhow::anyhow!("{label} model '{key}' has no files listed in models.toml"))?;

    let path = entry.abs_path(models_root).join(file);

    anyhow::ensure!(
        path.exists(),
        "{label} model file not found: {} \
         (run scripts/download_models.py to fetch it)",
        path.display()
    );

    Ok(path)
}

/// Copy config.template.toml to `dst`, giving the user an editable starting point.
fn copy_template(dst: &Path) -> Result<()> {
    let src = PathBuf::from(CONFIG_TEMPLATE);
    anyhow::ensure!(
        src.exists(),
        "{CONFIG_TEMPLATE} not found - it should live alongside the voicekit binary"
    );
    std::fs::copy(&src, dst)
        .with_context(|| format!("copying {CONFIG_TEMPLATE} to {}", dst.display()))?;
    Ok(())
}