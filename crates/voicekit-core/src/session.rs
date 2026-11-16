//! WebSocket session: handshake → VAD → STT → LLM → TTS → audio out.
//! Stale connections: Ping every HEARTBEAT_INTERVAL; idle exit after VAD_IDLE_TIMEOUT.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, trace, warn};

use crate::config::Config;
use crate::engines::{LlmEngine, SttEngine, TtsEngine, build_vad_with_overrides};
use crate::sentence::SentenceBuffer;
use crate::text_clean;
use crate::vad::VadEngine;

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
const VAD_IDLE_TIMEOUT: Duration = Duration::from_secs(45);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct Turn {
    pub role: Role,
    pub content: String,
}

pub fn build_prompt(history: &[Turn], system_prompt: &str) -> String {
    let mut prompt = String::new();

    if !system_prompt.is_empty() {
        prompt.push_str(system_prompt);
        prompt.push_str("\n\n");
    }

    for turn in history {
        match turn.role {
            Role::User => {
                prompt.push_str("User: ");
                prompt.push_str(&turn.content);
                prompt.push('\n');
            }
            Role::Assistant => {
                prompt.push_str("Assistant: ");
                prompt.push_str(&turn.content);
                prompt.push('\n');
            }
        }
    }

    prompt.push_str("Assistant:");
    prompt
}

#[derive(Debug, Deserialize)]
struct ClientSessionConfig {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    vad_threshold: Option<f32>,
    #[serde(default)]
    silence_ms: Option<u64>,
}

#[derive(Debug, Serialize)]
struct ServerFrame {
    #[serde(rename = "type")]
    msg_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ResolvedSessionConfig {
    pub system_prompt: String,
    pub voice: String,
    pub vad_threshold: f32,
    pub silence_ms: u64,
}

/// Returns `Ok(Some(cfg))` on a successful handshake.
/// Returns `Ok(None)` when the client times out or disconnects before sending
/// session.config — these are stale-connection cases that warrant a clean exit,
/// not an error.
/// Returns `Err` only for genuine protocol violations (bad JSON, policy denial, etc.).
async fn perform_handshake<S>(
    ws: &mut WebSocketStream<S>,
    config: &Arc<Config>,
) -> Result<Option<ResolvedSessionConfig>>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let raw_frame = match tokio::time::timeout(HANDSHAKE_TIMEOUT, ws.next()).await {
        Err(_) => {
            // Client connected but never sent session.config — stale, close cleanly.
            let _ = ws
                .send(error_frame("handshake timeout: no session.config received"))
                .await;
            debug!("handshake timeout — no session.config received");
            return Ok(None);
        }
        Ok(None) | Ok(Some(Err(_))) => {
            debug!("client disconnected during handshake");
            return Ok(None);
        }
        Ok(Some(Ok(Message::Text(text)))) => text,
        Ok(Some(Ok(_))) => {
            let _ = ws
                .send(error_frame(
                    "expected a JSON text frame for session.config, got binary or other",
                ))
                .await;
            anyhow::bail!("unexpected frame type during handshake");
        }
    };

    let client_cfg: ClientSessionConfig = match serde_json::from_str(&raw_frame) {
        Ok(c) => c,
        Err(e) => {
            let _ = ws
                .send(error_frame(&format!("invalid session.config JSON: {e}")))
                .await;
            anyhow::bail!("invalid handshake JSON: {e}");
        }
    };

    if client_cfg.msg_type != "session.config" {
        let _ = ws
            .send(error_frame(&format!(
                "expected type=session.config, got type={}",
                client_cfg.msg_type
            )))
            .await;
        anyhow::bail!("wrong handshake frame type: {}", client_cfg.msg_type);
    }

    let policy = &config.session_policy;

    if client_cfg.system_prompt.is_some() && !policy.allow_system_prompt_override {
        let _ = ws
            .send(error_frame(
                "system_prompt override is not permitted on this server",
            ))
            .await;
        anyhow::bail!("policy violation: system_prompt override denied");
    }

    if client_cfg.voice.is_some() && !policy.allow_voice_override {
        let _ = ws
            .send(error_frame(
                "voice override is not permitted on this server",
            ))
            .await;
        anyhow::bail!("policy violation: voice override denied");
    }

    if (client_cfg.vad_threshold.is_some() || client_cfg.silence_ms.is_some())
        && !policy.allow_vad_override
    {
        let _ = ws
            .send(error_frame(
                "VAD parameter override is not permitted on this server",
            ))
            .await;
        anyhow::bail!("policy violation: vad override denied");
    }

    let resolved = ResolvedSessionConfig {
        system_prompt: client_cfg
            .system_prompt
            .unwrap_or_else(|| config.llm.system_prompt.clone()),
        voice: client_cfg.voice.unwrap_or_else(|| config.tts.voice.clone()),
        vad_threshold: client_cfg.vad_threshold.unwrap_or(config.vad.threshold),
        silence_ms: client_cfg.silence_ms.unwrap_or(config.vad.silence_ms),
    };

    ws.send(Message::Text(
        serde_json::to_string(&ServerFrame {
            msg_type: "session.ready",
            message: None,
        })
        .unwrap()
        .into(),
    ))
    .await?;

    info!(
        voice         = %resolved.voice,
        silence_ms    = resolved.silence_ms,
        vad_threshold = resolved.vad_threshold,
        prompt_len    = resolved.system_prompt.len(),
        "handshake complete"
    );

    Ok(Some(resolved))
}

fn error_frame(msg: &str) -> Message {
    Message::Text(
        serde_json::to_string(&ServerFrame {
            msg_type: "session.error",
            message: Some(msg.to_string()),
        })
        .unwrap()
        .into(),
    )
}

pub async fn run_session<S>(
    ws_stream: WebSocketStream<S>,
    stt: Arc<dyn SttEngine>,
    llm: Arc<dyn LlmEngine>,
    tts: Arc<dyn TtsEngine>,
    vad: Box<dyn VadEngine>,
    vad_path: Arc<PathBuf>,
    config: Arc<Config>,
) -> Result<()>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let mut ws = ws_stream;

    let session_cfg = match perform_handshake(&mut ws, &config).await? {
        Some(cfg) => cfg,
        None => {
            // Stale connection: timeout or disconnect before session.config.
            return Ok(());
        }
    };

    let mut vad = if session_cfg.vad_threshold != config.vad.threshold
        || session_cfg.silence_ms != config.vad.silence_ms
    {
        build_vad_with_overrides(&vad_path, session_cfg.vad_threshold, session_cfg.silence_ms)?
    } else {
        vad
    };

    let mut history: Vec<Turn> = Vec::new();

    loop {
        let utterance = match vad_loop(&mut ws, &mut vad).await {
            VadOutcome::Utterance(utt) => utt,
            VadOutcome::Disconnected => {
                debug!("client disconnected");
                return Ok(());
            }
            VadOutcome::IdleTimeout => {
                info!("idle timeout - closing stale connection");
                return Ok(());
            }
            VadOutcome::Error(e) => {
                error!("VAD error: {e}");
                return Err(e);
            }
        };

        debug!(samples = utterance.len(), "VAD dispatched utterance");

        let stt_ref = Arc::clone(&stt);
        let transcript =
            match tokio::task::spawn_blocking(move || stt_ref.transcribe(&utterance)).await {
                Ok(Ok(t)) => t,
                Ok(Err(e)) => {
                    warn!("STT error, skipping utterance: {e}");
                    continue;
                }
                Err(e) => return Err(anyhow::anyhow!("STT task panicked: {e}")),
            };

        let transcript = transcript.trim().to_string();
        if transcript.is_empty() {
            debug!("empty STT result, discarding");
            continue;
        }
        info!(transcript = %transcript, "utterance transcribed");

        history.push(Turn {
            role: Role::User,
            content: transcript.clone(),
        });

        let prompt = build_prompt(&history, &session_cfg.system_prompt);
        debug!(turns = history.len(), "LLM generating");

        let cancel = CancellationToken::new();
        let (tx, mut rx) = mpsc::channel::<String>(64);

        let llm_handle = {
            let llm_ref = Arc::clone(&llm);
            let cancel_check = cancel.clone();
            tokio::task::spawn_blocking(move || {
                if cancel_check.is_cancelled() {
                    return Ok(());
                }
                llm_ref.generate(&prompt, tx)
            })
        };

        let mut sentence_buf = SentenceBuffer::new();
        let mut full_response = String::new();
        let mut socket_broken = false;
        let mut token_count = 0usize;

        while let Some(token) = rx.recv().await {
            token_count += 1;
            if token_count == 1 {
                trace!("LLM first token received");
            }
            full_response.push_str(&token);

            for sentence in sentence_buf.push(&token) {
                trace!(sentence = %sentence, "TTS dispatch");
                if let Err(e) =
                    synthesize_and_send(&tts, &sentence, &session_cfg.voice, &mut ws, &cancel).await
                {
                    warn!("TTS/send failed: {e}");
                    cancel.cancel();
                    socket_broken = true;
                    break;
                }
                trace!("TTS audio chunk sent");
            }

            if socket_broken {
                break;
            }
        }

        info!(response = %full_response, "LLM raw response");

        match llm_handle.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                warn!("LLM error on turn, skipping: {e}");
                continue;
            }
            Err(e) => return Err(anyhow::anyhow!("LLM task panicked: {e}")),
        }

        debug!(tokens = token_count, "LLM generation complete");

        if socket_broken {
            return Ok(());
        }

        if let Some(tail) = sentence_buf.flush() {
            debug!(tail = %tail, "TTS flush remainder");
            if !cancel.is_cancelled() {
                if let Err(e) =
                    synthesize_and_send(&tts, &tail, &session_cfg.voice, &mut ws, &cancel).await
                {
                    warn!("final TTS/send failed: {e}");
                    return Ok(());
                }
                trace!("final TTS audio chunk sent");
            }
        }

        history.push(Turn {
            role: Role::Assistant,
            content: full_response,
        });
    }
}

enum VadOutcome {
    Utterance(Vec<f32>),
    Disconnected,
    IdleTimeout,
    Error(anyhow::Error),
}

async fn vad_loop<S>(ws: &mut WebSocketStream<S>, vad: &mut Box<dyn VadEngine>) -> VadOutcome
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send,
{
    let (mut sink, mut stream) = ws.split();

    let mut heartbeat = tokio::time::interval(HEARTBEAT_INTERVAL);
    heartbeat.set_missed_tick_behavior(MissedTickBehavior::Skip);
    heartbeat.tick().await;

    let outcome = loop {
        tokio::select! {
            biased;

            read_result = tokio::time::timeout(VAD_IDLE_TIMEOUT, stream.next()) => {
                match read_result {
                    Err(_) => {
                        warn!(secs = VAD_IDLE_TIMEOUT.as_secs(), "idle timeout");
                        break VadOutcome::IdleTimeout;
                    }
                    Ok(None) => break VadOutcome::Disconnected,
                    Ok(Some(Err(e))) => {
                        debug!("socket read error: {e}");
                        break VadOutcome::Disconnected;
                    }
                    Ok(Some(Ok(Message::Binary(bytes)))) => {
                        let samples = bytes_to_pcm(&bytes);
                        if samples.len() != crate::vad::CHUNK_SAMPLES {
                            warn!(
                                got      = samples.len(),
                                expected = crate::vad::CHUNK_SAMPLES,
                                "wrong chunk size"
                            );
                            continue;
                        }
                        match vad.process_chunk(&samples) {
                            Ok(Some(utt)) => break VadOutcome::Utterance(utt),
                            Ok(None)      => continue,
                            Err(e)        => break VadOutcome::Error(e),
                        }
                    }
                    Ok(Some(Ok(Message::Close(_)))) => {
                        debug!("close frame received");
                        break VadOutcome::Disconnected;
                    }
                    Ok(Some(Ok(Message::Ping(p)))) => {
                        if let Err(e) = sink.send(Message::Pong(p)).await {
                            debug!("pong send failed: {e}");
                            break VadOutcome::Disconnected;
                        }
                    }
                    Ok(Some(Ok(Message::Pong(_)))) => { trace!("heartbeat pong received"); }
                    Ok(Some(Ok(_))) => {}
                }
            }

            _ = heartbeat.tick() => {
                if let Err(e) = sink.send(Message::Ping(Bytes::new())).await {
                    debug!("heartbeat failed: {e}");
                    break VadOutcome::Disconnected;
                }
            }
        }
    };

    outcome
}

async fn synthesize_and_send<S>(
    tts: &Arc<dyn TtsEngine>,
    sentence: &str,
    voice: &str,
    ws: &mut WebSocketStream<S>,
    cancel: &CancellationToken,
) -> Result<()>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send,
{
    if cancel.is_cancelled() {
        return Ok(());
    }

    let cleaned = text_clean::clean(sentence);
    debug!(raw = %sentence, cleaned = %cleaned, "text_clean");

    if cleaned.trim().is_empty() {
        trace!("sentence empty after cleaning, skipping TTS");
        return Ok(());
    }

    let tts_ref = Arc::clone(tts);
    let voice = voice.to_string();
    let audio = tokio::task::spawn_blocking(move || tts_ref.synthesize(&cleaned, &voice))
        .await
        .map_err(|e| anyhow::anyhow!("TTS task panicked: {e}"))??;

    if cancel.is_cancelled() {
        return Ok(());
    }

    ws.send(Message::Binary(pcm_to_bytes(&audio)))
        .await
        .map_err(|e| anyhow::anyhow!("socket send failed: {e}"))
}

pub fn bytes_to_pcm(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

pub fn pcm_to_bytes(samples: &[f32]) -> Bytes {
    let mut buf = Vec::with_capacity(samples.len() * 4);
    for &s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    Bytes::from(buf)
}
