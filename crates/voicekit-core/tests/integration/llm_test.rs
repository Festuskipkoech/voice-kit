#![cfg(feature = "integration")]

use std::path::PathBuf;
use tokio::sync::mpsc;
use voicekit_core::config::Config;
use voicekit_core::engines::build_llm;
use voicekit_core::registry::ModelRegistry;

fn env_path(var: &str, default: &str) -> PathBuf {
    std::env::var(var)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(default))
}

#[test]
fn generates_answer_to_simple_question() {
    let cfg = Config::from_file(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/config.toml"))
        .expect("config");

    let registry =
        ModelRegistry::from_file(env_path("VOICEKIT_REGISTRY", "models.toml")).expect("registry");
    let models_dir = env_path("VOICEKIT_MODELS_DIR", "/models");
    let model_path = registry
        .resolve(&cfg.llm.model, &models_dir)
        .expect("resolve llm path");

    let engine = build_llm(&cfg.llm, &model_path).expect("build_llm");

    let (tx, mut rx) = mpsc::channel::<String>(256);

    let handle = std::thread::spawn(move || engine.generate("What is ai", tx).expect("generate"));

    let mut response = String::new();
    while let Some(piece) = rx.blocking_recv() {
        response.push_str(&piece);
    }

    handle.join().expect("thread panicked");

    assert!(
        !response.trim().is_empty(),
        "expected non-empty response, got nothing"
    );
}
