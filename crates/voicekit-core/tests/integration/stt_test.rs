#![cfg(feature = "integration")]

use std::path::PathBuf;
use voicekit_core::config::Config;
use voicekit_core::engines::build_stt;
use voicekit_core::registry::ModelRegistry;

fn env_path(var: &str, default: &str) -> PathBuf {
    std::env::var(var)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(default))
}

fn read_wav(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open wav");
    assert_eq!(reader.spec().sample_rate, 16000, "fixture must be 16 kHz");
    reader
        .samples::<i16>()
        .map(|s| s.expect("sample") as f32 / i16::MAX as f32)
        .collect()
}

#[test]
fn transcribes_hello_fixture() {
    let cfg = Config::from_file(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/config.toml"))
        .expect("config");

    let registry   = ModelRegistry::from_file(env_path("VOICEKIT_REGISTRY", "models.toml")).expect("registry");
    let models_dir = env_path("VOICEKIT_MODELS_DIR", "/models");
    let model_path = registry.resolve(&cfg.stt.model, &models_dir).expect("resolve stt path");

    let engine = build_stt(&cfg.stt, &model_path).expect("build_stt");

    let audio = read_wav(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/hello.wav"));
    let transcript = engine.transcribe(&audio).expect("transcribe");
    let lower = transcript.to_lowercase();

    assert!(
        lower.contains("hello") || lower.contains("hey"),
        "expected 'hello' in transcript, got: {transcript:?}"
    );
}