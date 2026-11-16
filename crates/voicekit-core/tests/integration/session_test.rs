//! Integration tests for the full session pipeline.
//!
//! Requires real model files. Run with:
//!   VOICEKIT_INTEGRATION=1 cargo test -p voicekit-core \
//!       --test session_integration_test --features integration

#![cfg(feature = "integration")]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, connect_async};
use tokio_tungstenite::tungstenite::Message;

use voicekit_core::config::Config;
use voicekit_core::engines::{build_llm, build_stt, build_tts, build_vad, LlmEngine, SttEngine, TtsEngine};
use voicekit_core::registry::ModelRegistry;
use voicekit_core::session::{bytes_to_pcm, pcm_to_bytes, run_session};

fn env_path(var: &str, default: &str) -> PathBuf {
    std::env::var(var)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(default))
}

struct Engines {
    stt:      Arc<dyn SttEngine>,
    llm:      Arc<dyn LlmEngine>,
    tts:      Arc<dyn TtsEngine>,
    vad_path: Arc<PathBuf>,
}

// llama.cpp backend and ORT environment are process-wide singletons.
// Initialize once and share across all tests in this binary.
static ENGINES: OnceLock<Engines> = OnceLock::new();

fn fixture_config() -> Config {
    Config::from_file(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/config.toml"))
        .expect("fixtures/config.toml missing or invalid")
}

fn shared_engines() -> &'static Engines {
    ENGINES.get_or_init(|| {
        let config     = fixture_config();
        let registry   = ModelRegistry::from_file(env_path("VOICEKIT_REGISTRY", "models.toml")).expect("registry");
        let models_dir = env_path("VOICEKIT_MODELS_DIR", "/models");

        let vad_path = Arc::new(registry.resolve(&config.vad.model, &models_dir).expect("resolve vad path"));
        let stt_path = registry.resolve(&config.stt.model, &models_dir).expect("resolve stt path");
        let llm_path = registry.resolve(&config.llm.model, &models_dir).expect("resolve llm path");
        let tts_path = registry.resolve(&config.tts.model, &models_dir).expect("resolve tts path");

        Engines {
            stt:      Arc::from(build_stt(&config.stt, &stt_path).expect("build_stt")),
            llm:      Arc::from(build_llm(&config.llm, &llm_path).expect("build_llm")),
            tts:      Arc::from(build_tts(&config.tts, &tts_path).expect("build_tts")),
            vad_path,
        }
    })
}

fn hello_wav_pcm() -> Vec<f32> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/hello.wav");
    let mut reader = hound::WavReader::open(path).expect("fixtures/hello.wav missing");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16000, "hello.wav must be 16 kHz");
    assert_eq!(spec.channels, 1, "hello.wav must be mono");
    reader
        .samples::<i16>()
        .map(|s| s.expect("bad sample") as f32 / i16::MAX as f32)
        .collect()
}

async fn handshake(client: &mut tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>) {
    client
        .send(Message::Text(r#"{"type":"session.config"}"#.to_string().into()))
        .await
        .expect("send session.config");

    loop {
        match client.next().await {
            Some(Ok(Message::Text(t))) => {
                let v: serde_json::Value = serde_json::from_str(&t).expect("bad json");
                match v["type"].as_str() {
                    Some("session.ready") => return,
                    Some("session.error") => panic!("session.error: {}", v["message"]),
                    _ => continue,
                }
            }
            other => panic!("unexpected frame during handshake: {other:?}"),
        }
    }
}

async fn spawn_server(config: Arc<Config>) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let engines  = shared_engines();
    let stt      = Arc::clone(&engines.stt);
    let llm      = Arc::clone(&engines.llm);
    let tts      = Arc::clone(&engines.tts);
    let vad_path = Arc::clone(&engines.vad_path);

    tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let ws = accept_async(tcp).await.unwrap();

        // VAD is per-session: it holds mutable chunk state.
        let vad = build_vad(&config.vad, &vad_path).expect("build_vad");

        run_session(ws, stt, llm, tts, vad, vad_path, config)
            .await
            .expect("run_session");
    });

    addr
}

#[tokio::test]
async fn real_pipeline_produces_audio() {
    let config = Arc::new(fixture_config());
    let addr = spawn_server(Arc::clone(&config)).await;

    let (mut client, _) = connect_async(format!("ws://{addr}"))
        .await
        .expect("connect failed");

    handshake(&mut client).await;

    let pcm = hello_wav_pcm();
    let chunk_size = config.vad.chunk_samples;

    for chunk in pcm.chunks(chunk_size) {
        let mut padded = chunk.to_vec();
        padded.resize(chunk_size, 0.0);
        client
            .send(Message::Binary(Bytes::from(pcm_to_bytes(&padded))))
            .await
            .expect("send failed");
    }

    // hello.wav has only ~17 trailing silence chunks but the VAD needs
    // silence_ms worth (~25 chunks at 800 ms) to dispatch the utterance.
    // Send explicit silence to guarantee the threshold is crossed.
    let silence_pad = vec![0.0_f32; chunk_size];
    for _ in 0..30 {
        client
            .send(Message::Binary(Bytes::from(pcm_to_bytes(&silence_pad))))
            .await
            .expect("silence pad failed");
    }

    let audio_frame = tokio::time::timeout(Duration::from_secs(120), async {
        loop {
            match client.next().await {
                Some(Ok(Message::Binary(b))) => return Some(b),
                Some(Ok(_)) => continue,
                _ => return None,
            }
        }
    })
    .await
    .expect("timed out waiting for audio response (120s)");

    let audio_frame = audio_frame.expect("connection closed without audio");
    let samples = bytes_to_pcm(&audio_frame);

    assert!(!samples.is_empty(), "received empty audio frame");

    let energy: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
    assert!(energy > 1e-6, "audio frame is silent (energy = {energy:.2e})");

    client.send(Message::Close(None)).await.ok();
}

#[tokio::test]
async fn real_pipeline_handles_silence() {
    // Send pure silence - VAD should not dispatch, session stays alive.
    let config = Arc::new(fixture_config());
    let addr = spawn_server(Arc::clone(&config)).await;

    let (mut client, _) = connect_async(format!("ws://{addr}"))
        .await
        .expect("connect failed");

    handshake(&mut client).await;

    let silence = vec![0.0_f32; config.vad.chunk_samples];
    for _ in 0..10 {
        client
            .send(Message::Binary(Bytes::from(pcm_to_bytes(&silence))))
            .await
            .expect("send failed");
    }

    let no_audio = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            match client.next().await {
                Some(Ok(Message::Binary(_))) => return false,
                Some(Ok(_)) => continue,
                _ => return true,
            }
        }
    })
    .await
    .unwrap_or(true);

    assert!(no_audio, "session sent audio for silent input");

    client.send(Message::Close(None)).await.ok();
}