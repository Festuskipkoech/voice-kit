#![cfg(feature = "integration")]

use std::path::PathBuf;
use voicekit_core::config::Config;
use voicekit_core::engines::build_tts;
use voicekit_core::registry::ModelRegistry;

fn env_path(var: &str, default: &str) -> PathBuf {
    std::env::var(var)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(default))
}

#[test]
fn synthesizes_sentence() {
    let cfg = Config::from_file(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/config.toml"))
        .expect("config");

    let registry   = ModelRegistry::from_file(env_path("VOICEKIT_REGISTRY", "models.toml")).expect("registry");
    let models_dir = env_path("VOICEKIT_MODELS_DIR", "/models");
    let model_path = registry.resolve(&cfg.tts.model, &models_dir).expect("resolve tts path");

    let engine = build_tts(&cfg.tts, &model_path).expect("build_tts");

    let audio = engine.synthesize("Hello, world.", &cfg.tts.voice).expect("synthesize");

    assert!(!audio.is_empty(), "expected non-empty audio, got nothing");
    // Paradee outputs 24kHz - a minimal sentence should be at least 0.3s
    assert!(
        audio.len() > 24000 / 3,
        "audio suspiciously short: {} samples",
        audio.len()
    );
}