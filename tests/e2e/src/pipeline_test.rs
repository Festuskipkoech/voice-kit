//! End-to-end pipeline test.
//!
//! Requires real model files and is gated behind the `integration` feature.
//! Set VOICEKIT_INTEGRATION=1 to allow this test to compile and run.
//!
//! Run with:
//!   VOICEKIT_INTEGRATION=1 cargo test -p e2e --features integration -- --nocapture
//!
//! Environment variables:
//!   VOICEKIT_CONFIG      path to config.toml      (default: "config.local.toml")
//!   VOICEKIT_REGISTRY    path to models.toml       (default: "models.toml")
//!   VOICEKIT_MODELS_DIR  root directory of models  (default: "/models")
//!   VOICEKIT_HELLO       path to hello.wav          (default: "fixtures/hello.wav")

#![cfg(feature = "integration")]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{accept_async, connect_async};

use voicekit_core::config::Config;
use voicekit_core::engines::{build_llm, build_stt, build_tts, build_vad};
use voicekit_core::registry::ModelRegistry;
use voicekit_core::session::{bytes_to_pcm, pcm_to_bytes, run_session};

// Helpers

fn env_path(var: &str, default: &str) -> PathBuf {
    std::env::var(var)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(default))
}

/// Load a WAV file and return 16 kHz mono f32 samples.
///
/// Only supports PCM-16 or PCM-32 WAV files. Stereo files are accepted but
/// only the left channel is used.
///
/// # Panics
///
/// Panics if the file is not 16 kHz - VAD and STT both require 16 kHz input.
/// A fixture recorded at the wrong sample rate would silently produce wrong
/// chunk timing and cause the test to hang rather than fail clearly.
fn load_wav_f32(path: &PathBuf) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open WAV: {}", path.display()))?;
    let spec = reader.spec();

    // Validate sample rate - VAD chunks are 512 samples at 16 kHz (32 ms).
    // A 44.1 kHz file would make each chunk 11.6 ms, breaking VAD timing.
    assert_eq!(
        spec.sample_rate, 16_000,
        "fixture WAV must be 16 kHz (got {} Hz) - VAD and STT both expect 16 kHz input",
        spec.sample_rate
    );

    // Tolerate stereo by taking left channel only.
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .step_by(channels)
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect::<std::result::Result<_, _>>()
            .context("error reading i16 PCM samples")?,
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .step_by(channels)
            .collect::<std::result::Result<_, _>>()
            .context("error reading f32 PCM samples")?,
    };

    Ok(samples)
}

/// Chunk a flat sample buffer into `chunk_size`-sized frames suitable for VAD.
fn chunk_samples(samples: &[f32], chunk_size: usize) -> Vec<Bytes> {
    samples
        .chunks(chunk_size)
        .map(|c| {
            let mut padded = c.to_vec();
            padded.resize(chunk_size, 0.0_f32);
            pcm_to_bytes(&padded)
        })
        .collect()
}

/// Perform the session handshake as the client side.
/// Sends `session.config` and waits up to 5 s for `session.ready`.
async fn client_handshake(
    client: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> Result<()> {
    client
        .send(Message::Text(
            r#"{"type":"session.config"}"#.to_string().into(),
        ))
        .await
        .context("failed to send session.config")?;

    let reply = tokio::time::timeout(Duration::from_secs(5), client.next())
        .await
        .context("timed out waiting for session.ready")?
        .context("server closed connection during handshake")?
        .context("WebSocket error during handshake")?;

    match reply {
        Message::Text(text) => {
            let v: serde_json::Value =
                serde_json::from_str(&text).context("server sent invalid JSON during handshake")?;
            match v["type"].as_str() {
                Some("session.ready") => Ok(()),
                Some("session.error") => {
                    anyhow::bail!("server rejected session: {}", v["message"])
                }
                other => anyhow::bail!("unexpected handshake frame type: {:?}", other),
            }
        }
        other => anyhow::bail!("expected text frame for session.ready, got: {:?}", other),
    }
}

// E2E test

/// Full pipeline: real VAD + STT + LLM + TTS.
///
/// Sends `hello.wav` over a WebSocket, waits up to 60 s for at least one
/// non-empty audio frame back. Verifies the returned PCM deserializes cleanly.
#[tokio::test]
async fn pipeline_hello_returns_audio() -> Result<()> {
    // Load config
    let config_path = env_path("VOICEKIT_CONFIG", "config.local.toml");
    let config = Config::from_file(&config_path)
        .with_context(|| format!("config not found at {}", config_path.display()))?;
    let config = Arc::new(config);

    // Load model registry and resolve paths
    let registry_path = env_path("VOICEKIT_REGISTRY", "models.toml");
    let models_dir = env_path("VOICEKIT_MODELS_DIR", "/models");
    let registry = ModelRegistry::from_file(&registry_path)
        .with_context(|| format!("registry not found at {}", registry_path.display()))?;
    let models_root: &Path = &models_dir;

    let vad_path = Arc::new(
        registry
            .resolve("silero-vad", models_root)
            .context("resolving VAD path")?,
    );
    let stt_path = registry
        .resolve("moonshine-tiny", models_root)
        .context("resolving STT path")?;
    let llm_path = registry
        .resolve("gemma3-1b-q4", models_root)
        .context("resolving LLM path")?;
    let tts_path = registry
        .resolve("paradee-int8", models_root)
        .context("resolving TTS path")?;

    // Build engines once - shared across all sessions via Arc
    let stt = Arc::from(build_stt(&config.stt, &stt_path).context("failed to build STT engine")?);
    let llm = Arc::from(build_llm(&config.llm, &llm_path).context("failed to build LLM engine")?);
    let tts = Arc::from(build_tts(&config.tts, &tts_path).context("failed to build TTS engine")?);

    // Load hello.wav
    let wav_path = env_path("VOICEKIT_HELLO", "fixtures/hello.wav");
    let samples = load_wav_f32(&wav_path)
        .with_context(|| format!("failed to load hello.wav at {}", wav_path.display()))?;

    let chunk_size = config.vad.chunk_samples;
    let frames = chunk_samples(&samples, chunk_size);
    eprintln!(
        "[TEST] loaded {} samples, {} frames",
        samples.len(),
        frames.len()
    );

    // Bind a random-port server
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    let url = format!("ws://127.0.0.1:{port}");

    let server_config = config.clone();
    let server_vad_path = Arc::clone(&vad_path);
    let server_task = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await?;
        let ws = accept_async(tcp).await?;
        let vad = build_vad(&server_config.vad, &server_vad_path)?;
        run_session(ws, stt, llm, tts, vad, server_vad_path, server_config).await
    });

    tokio::time::sleep(Duration::from_millis(100)).await;
    if server_task.is_finished() {
        let result = server_task.await?;
        eprintln!("[TEST] server task exited early: {:?}", result);
        panic!("server task died before client connected");
    }

    // Connect as client and complete handshake before sending audio
    let (mut client, _) = connect_async(&url).await?;
    client_handshake(&mut client)
        .await
        .context("session handshake failed")?;
    eprintln!("[TEST] handshake complete, sending audio frames");

    // Send all audio frames
    for frame in &frames {
        client.send(Message::Binary(frame.clone())).await?;
    }
    eprintln!(
        "[TEST] all {} frames sent, waiting for response...",
        frames.len()
    );

    if server_task.is_finished() {
        let result = server_task.await?;
        eprintln!("[TEST] server task exited after frames: {:?}", result);
        panic!("server task died");
    }

    // Collect audio response.
    // 60-second timeout: the full STT + LLM + TTS pipeline takes up to ~10 s on
    // CPU under load; 60 s gives headroom without letting a hung test block CI forever.
    let audio_frames: Vec<Bytes> = tokio::time::timeout(Duration::from_secs(60), async {
        let mut collected = Vec::new();
        loop {
            match client.next().await {
                Some(Ok(Message::Binary(b))) => {
                    collected.push(b);
                    // One frame is enough to prove the pipeline ran end-to-end.
                    if !collected.is_empty() {
                        break;
                    }
                }
                Some(Ok(Message::Close(_))) | None => break,
                Some(Ok(_)) => {}
                Some(Err(e)) => {
                    eprintln!("client receive error: {e}");
                    break;
                }
            }
        }
        collected
    })
    .await
    .context("timed out after 60 s waiting for audio response")?;

    // Close cleanly
    let _ = client.send(Message::Close(None)).await;
    drop(client);

    // Assertions
    assert!(
        !audio_frames.is_empty(),
        "expected at least one audio frame from the pipeline"
    );

    for (i, frame) in audio_frames.iter().enumerate() {
        assert!(!frame.is_empty(), "frame {i} is empty");

        // Each f32 PCM sample is 4 bytes (little-endian). A frame whose byte
        // length is not a multiple of 4 means bytes_to_pcm would silently drop
        // trailing bytes - that indicates a corrupt or truncated server frame.
        assert!(
            frame.len() % 4 == 0,
            "frame {i}: byte length {} is not a multiple of 4 - \
             each f32 PCM sample is 4 bytes; trailing bytes indicate \
             a corrupt or truncated frame from the server",
            frame.len()
        );

        let samples = bytes_to_pcm(frame);
        assert!(
            !samples.is_empty(),
            "frame {i}: no samples after deserialisation"
        );

        // All TTS output should be non-silent (at least some non-zero energy).
        let energy: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
        assert!(
            energy > 0.0,
            "frame {i}: TTS audio is completely silent (energy = 0)"
        );
    }

    // Wait for server task
    let _ = tokio::time::timeout(Duration::from_secs(5), server_task).await;

    Ok(())
}
