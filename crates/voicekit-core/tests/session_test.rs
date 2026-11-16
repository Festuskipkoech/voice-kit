//! Unit tests for session-layer pure functions and mock-engine WebSocket behaviour.
//! No model files, no feature gate - CI-safe.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{accept_async, connect_async};

use voicekit_core::config::{
    Config, LlmConfig, ServerConfig, SessionPolicy, SttConfig, TtsConfig, VadConfig,
};
use voicekit_core::engines::{LlmEngine, SttEngine, TtsEngine};
use voicekit_core::session::{Role, Turn, build_prompt, bytes_to_pcm, pcm_to_bytes, run_session};
use voicekit_core::vad::{CHUNK_SAMPLES, VadEngine};

// build_prompt

#[test]
fn build_prompt_empty_history_no_system_prompt() {
    assert_eq!(build_prompt(&[], ""), "Assistant:");
}

#[test]
fn build_prompt_system_prompt_prepended() {
    assert_eq!(
        build_prompt(&[], "You are a helpful voice assistant."),
        "You are a helpful voice assistant.\n\nAssistant:"
    );
}

#[test]
fn build_prompt_single_user_turn() {
    let h = vec![Turn {
        role: Role::User,
        content: "hello".to_string(),
    }];
    assert_eq!(build_prompt(&h, ""), "User: hello\nAssistant:");
}

#[test]
fn build_prompt_multi_turn_interleaving() {
    let h = vec![
        Turn {
            role: Role::User,
            content: "hello".to_string(),
        },
        Turn {
            role: Role::Assistant,
            content: "Hi there.".to_string(),
        },
        Turn {
            role: Role::User,
            content: "what time is it".to_string(),
        },
    ];
    assert_eq!(
        build_prompt(&h, "Be brief."),
        "Be brief.\n\nUser: hello\nAssistant: Hi there.\nUser: what time is it\nAssistant:"
    );
}

#[test]
fn build_prompt_empty_system_prompt_leaves_no_blank_line() {
    let result = build_prompt(&[], "");
    assert!(
        !result.starts_with('\n'),
        "prompt must not start with a blank line"
    );
}

#[test]
fn build_prompt_empty_turn_content_does_not_panic() {
    let h = vec![Turn {
        role: Role::User,
        content: String::new(),
    }];
    assert_eq!(build_prompt(&h, ""), "User: \nAssistant:");
}

// PCM conversion

#[test]
fn pcm_roundtrip_preserves_values() {
    let original: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.123_456];
    let recovered = bytes_to_pcm(&pcm_to_bytes(&original));
    assert_eq!(original.len(), recovered.len());
    for (a, b) in original.iter().zip(recovered.iter()) {
        assert!((a - b).abs() < 1e-6, "roundtrip mismatch: {a:.8} vs {b:.8}");
    }
}

#[test]
fn pcm_to_bytes_length() {
    assert_eq!(pcm_to_bytes(&vec![0.0_f32; 512]).len(), 512 * 4);
}

#[test]
fn bytes_to_pcm_empty_input() {
    assert!(bytes_to_pcm(&[]).is_empty());
}

#[test]
fn bytes_to_pcm_ignores_incomplete_tail() {
    assert_eq!(bytes_to_pcm(&vec![0u8; 9]).len(), 2);
}

#[test]
fn bytes_to_pcm_little_endian() {
    let samples = bytes_to_pcm(&[0x00, 0x00, 0x80, 0x3F]);
    assert_eq!(samples.len(), 1);
    assert!((samples[0] - 1.0_f32).abs() < 1e-6);
}

#[test]
fn pcm_to_bytes_little_endian() {
    assert_eq!(&pcm_to_bytes(&[1.0_f32])[..], &[0x00, 0x00, 0x80, 0x3F]);
}

// Mock engines

struct MockStt {
    response: &'static str,
}
impl SttEngine for MockStt {
    fn transcribe(&self, _audio: &[f32]) -> Result<String> {
        Ok(self.response.to_string())
    }
}

struct MockLlm {
    tokens: Vec<&'static str>,
}
impl LlmEngine for MockLlm {
    fn generate(&self, _prompt: &str, tx: tokio::sync::mpsc::Sender<String>) -> Result<()> {
        for &tok in &self.tokens {
            if tx.blocking_send(tok.to_string()).is_err() {
                break;
            }
        }
        Ok(())
    }
}

struct FailingLlm;
impl LlmEngine for FailingLlm {
    fn generate(&self, _prompt: &str, _tx: tokio::sync::mpsc::Sender<String>) -> Result<()> {
        Err(anyhow::anyhow!("simulated LLM failure"))
    }
}

struct MockTts {
    n_samples: usize,
}
impl TtsEngine for MockTts {
    fn synthesize(&self, _text: &str, _voice: &str) -> Result<Vec<f32>> {
        Ok(vec![0.0_f32; self.n_samples])
    }
}

struct MockVad {
    dispatch_after: usize,
    received: usize,
    buffer: Vec<f32>,
}
impl MockVad {
    fn new(n: usize) -> Self {
        Self {
            dispatch_after: n,
            received: 0,
            buffer: Vec::new(),
        }
    }
}
impl VadEngine for MockVad {
    fn process_chunk(&mut self, samples: &[f32]) -> Result<Option<Vec<f32>>> {
        self.received += 1;
        self.buffer.extend_from_slice(samples);
        if self.received == self.dispatch_after {
            return Ok(Some(std::mem::take(&mut self.buffer)));
        }
        Ok(None)
    }
    fn flush(&mut self) -> Option<Vec<f32>> {
        if self.buffer.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.buffer))
        }
    }
}

// Helpers

fn test_config() -> Arc<Config> {
    Arc::new(Config {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            max_connections: 100,
        },
        vad: VadConfig {
            // model key is irrelevant for mock sessions - engines are injected directly.
            model: String::new(),
            threshold: 0.5,
            silence_ms: 800,
            chunk_samples: CHUNK_SAMPLES,
        },
        stt: SttConfig {
            provider: "mock".to_string(),
            model: String::new(),
        },
        llm: LlmConfig {
            provider: "mock".to_string(),
            model: String::new(),
            system_prompt: "test".to_string(),
            context_size: 2048,
        },
        tts: TtsConfig {
            provider: "mock".to_string(),
            model: String::new(),
            voice: "af_heart".to_string(),
            sample_rate: 24000,
        },
        session_policy: SessionPolicy::default(),
    })
}

async fn bind_listener() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    (listener, format!("ws://127.0.0.1:{port}"))
}

// Perform the mandatory session handshake from the client side.
// Sends session.config and waits for session.ready.
async fn handshake(
    client: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) {
    client
        .send(Message::Text(r#"{"type":"session.config"}"#.into()))
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            match client.next().await {
                Some(Ok(Message::Text(t))) if t.contains("session.ready") => break,
                Some(Ok(_)) => continue,
                other => panic!("expected session.ready, got {other:?}"),
            }
        }
    })
    .await
    .expect("no session.ready within 2 s");
}

// WebSocket tests

#[tokio::test]
async fn mock_session_returns_audio_frame() {
    let (listener, url) = bind_listener().await;
    let config = test_config();

    tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let ws = accept_async(tcp).await.unwrap();
        run_session(
            ws,
            Arc::new(MockStt {
                response: "hello world",
            }) as Arc<dyn SttEngine>,
            Arc::new(MockLlm {
                tokens: vec!["Hello", " there", "."],
            }) as Arc<dyn LlmEngine>,
            Arc::new(MockTts { n_samples: 1200 }) as Arc<dyn TtsEngine>,
            Box::new(MockVad::new(1)) as Box<dyn VadEngine>,
            Arc::new(std::path::PathBuf::new()),
            config,
        )
        .await
        .unwrap();
    });

    let (mut client, _) = connect_async(&url).await.unwrap();
    handshake(&mut client).await;

    client
        .send(Message::Binary(pcm_to_bytes(&vec![0.0_f32; CHUNK_SAMPLES])))
        .await
        .unwrap();

    let audio = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match client.next().await {
                Some(Ok(Message::Binary(b))) => return b,
                Some(Ok(_)) => continue,
                other => panic!("unexpected: {other:?}"),
            }
        }
    })
    .await
    .expect("no audio frame within 5 s");

    assert_eq!(bytes_to_pcm(&audio).len(), 1200);
}

#[tokio::test]
async fn empty_transcript_is_discarded() {
    let (listener, url) = bind_listener().await;
    let config = test_config();

    tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let ws = accept_async(tcp).await.unwrap();
        run_session(
            ws,
            Arc::new(MockStt { response: "   " }) as Arc<dyn SttEngine>,
            Arc::new(MockLlm { tokens: vec![] }) as Arc<dyn LlmEngine>,
            Arc::new(MockTts { n_samples: 0 }) as Arc<dyn TtsEngine>,
            Box::new(MockVad::new(1)) as Box<dyn VadEngine>,
            Arc::new(std::path::PathBuf::new()),
            config,
        )
        .await
        .unwrap();
    });

    let (mut client, _) = connect_async(&url).await.unwrap();
    handshake(&mut client).await;

    client
        .send(Message::Binary(pcm_to_bytes(&vec![0.0_f32; CHUNK_SAMPLES])))
        .await
        .unwrap();
    client.send(Message::Close(None)).await.unwrap();

    tokio::time::timeout(Duration::from_secs(3), async {
        while let Some(msg) = client.next().await {
            match msg {
                Ok(Message::Binary(_)) => panic!("unexpected audio for empty transcript"),
                Ok(Message::Close(_)) | Err(_) => break,
                _ => {}
            }
        }
    })
    .await
    .expect("session did not close within 3 s");
}

#[tokio::test]
async fn llm_error_does_not_crash_session() {
    let (listener, url) = bind_listener().await;
    let config = test_config();

    let server_task = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let ws = accept_async(tcp).await.unwrap();
        run_session(
            ws,
            Arc::new(MockStt { response: "hello" }) as Arc<dyn SttEngine>,
            Arc::new(FailingLlm) as Arc<dyn LlmEngine>,
            Arc::new(MockTts { n_samples: 0 }) as Arc<dyn TtsEngine>,
            Box::new(MockVad::new(1)) as Box<dyn VadEngine>,
            Arc::new(std::path::PathBuf::new()),
            config,
        )
        .await
    });

    let (mut client, _) = connect_async(&url).await.unwrap();
    handshake(&mut client).await;

    client
        .send(Message::Binary(pcm_to_bytes(&vec![0.0_f32; CHUNK_SAMPLES])))
        .await
        .unwrap();
    client.send(Message::Close(None)).await.unwrap();

    tokio::time::timeout(Duration::from_secs(3), async {
        while let Some(msg) = client.next().await {
            match msg {
                Ok(Message::Binary(_)) => panic!("audio sent despite LLM error"),
                Ok(Message::Close(_)) | Err(_) => break,
                _ => {}
            }
        }
    })
    .await
    .expect("session did not close within 3 s");

    let result = tokio::time::timeout(Duration::from_secs(3), server_task)
        .await
        .expect("server task did not finish")
        .expect("server task panicked");

    assert!(result.is_ok());
}

#[tokio::test]
async fn clean_disconnect_during_vad_loop() {
    let (listener, url) = bind_listener().await;
    let config = test_config();

    let server_task = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let ws = accept_async(tcp).await.unwrap();
        run_session(
            ws,
            Arc::new(MockStt { response: "hello" }) as Arc<dyn SttEngine>,
            Arc::new(MockLlm { tokens: vec![] }) as Arc<dyn LlmEngine>,
            Arc::new(MockTts { n_samples: 0 }) as Arc<dyn TtsEngine>,
            Box::new(MockVad::new(100)) as Box<dyn VadEngine>,
            Arc::new(std::path::PathBuf::new()),
            config,
        )
        .await
    });

    let (mut client, _) = connect_async(&url).await.unwrap();
    handshake(&mut client).await;

    for _ in 0..3 {
        client
            .send(Message::Binary(pcm_to_bytes(&vec![0.0_f32; CHUNK_SAMPLES])))
            .await
            .unwrap();
    }
    client.send(Message::Close(None)).await.unwrap();

    let result = tokio::time::timeout(Duration::from_secs(3), server_task)
        .await
        .expect("server did not finish within 3 s")
        .expect("server task panicked");

    assert!(result.is_ok());
}

#[tokio::test]
async fn disconnect_mid_response_exits_cleanly() {
    let (listener, url) = bind_listener().await;
    let config = test_config();

    let server_task = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let ws = accept_async(tcp).await.unwrap();
        run_session(
            ws,
            Arc::new(MockStt {
                response: "tell me a story",
            }) as Arc<dyn SttEngine>,
            Arc::new(MockLlm {
                tokens: vec!["Once", " upon", " a", " time.", " The", " end."],
            }) as Arc<dyn LlmEngine>,
            Arc::new(MockTts { n_samples: 2400 }) as Arc<dyn TtsEngine>,
            Box::new(MockVad::new(1)) as Box<dyn VadEngine>,
            Arc::new(std::path::PathBuf::new()),
            config,
        )
        .await
    });

    let (mut client, _) = connect_async(&url).await.unwrap();
    handshake(&mut client).await;

    client
        .send(Message::Binary(pcm_to_bytes(&vec![0.0_f32; CHUNK_SAMPLES])))
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match client.next().await {
                Some(Ok(Message::Binary(_))) => break,
                Some(Ok(_)) => continue,
                other => panic!("unexpected: {other:?}"),
            }
        }
    })
    .await
    .expect("first audio frame not received within 5 s");

    drop(client);

    let result = tokio::time::timeout(Duration::from_secs(5), server_task)
        .await
        .expect("server did not finish within 5 s")
        .expect("server task panicked");

    assert!(result.is_ok());
}

#[tokio::test(start_paused = true)]
async fn server_heartbeat_keeps_session_alive() {
    let (listener, url) = bind_listener().await;
    let config = test_config();

    let server_task = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let ws = accept_async(tcp).await.unwrap();
        run_session(
            ws,
            Arc::new(MockStt { response: "hello" }) as Arc<dyn SttEngine>,
            Arc::new(MockLlm {
                tokens: vec!["Hi."],
            }) as Arc<dyn LlmEngine>,
            Arc::new(MockTts { n_samples: 480 }) as Arc<dyn TtsEngine>,
            Box::new(MockVad::new(999)) as Box<dyn VadEngine>,
            Arc::new(std::path::PathBuf::new()),
            config,
        )
        .await
    });

    let (mut client, _) = connect_async(&url).await.unwrap();
    handshake(&mut client).await;

    // Advance to exactly the heartbeat boundary so the tick fires normally.
    // Overshooting (e.g. 16 s) causes MissedTickBehavior::Skip to skip the
    // tick and reschedule it at t=30 s, so the Ping never arrives.
    tokio::time::advance(Duration::from_secs(15)).await;

    // Resume real time before waiting for the Ping.  With the clock still
    // paused, tokio::time::timeout auto-fires via mock-clock auto-advance the
    // moment both tasks are blocked on I/O — before the epoll reactor has a
    // chance to deliver EPOLLIN on the client socket.  Resuming lets the
    // 500 ms timeout run on wall-clock time, which is long enough for the
    // loopback Ping to arrive.
    tokio::time::resume();

    tokio::time::timeout(Duration::from_millis(500), async {
        loop {
            match client.next().await {
                Some(Ok(Message::Ping(payload))) => {
                    client.send(Message::Pong(payload)).await.unwrap();
                    break;
                }
                Some(Ok(_)) => continue,
                other => panic!("unexpected: {other:?}"),
            }
        }
    })
    .await
    .expect("no Ping after heartbeat interval");

    drop(client);
    tokio::task::yield_now().await;

    let result = tokio::time::timeout(Duration::from_secs(5), server_task)
        .await
        .expect("server did not exit after client dropped")
        .expect("server task panicked");

    assert!(result.is_ok());
}

#[tokio::test(start_paused = true)]
async fn idle_timeout_closes_stale_session() {
    let (listener, url) = bind_listener().await;
    let config = test_config();

    let server_task = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let ws = accept_async(tcp).await.unwrap();
        run_session(
            ws,
            Arc::new(MockStt { response: "hello" }) as Arc<dyn SttEngine>,
            Arc::new(MockLlm { tokens: vec![] }) as Arc<dyn LlmEngine>,
            Arc::new(MockTts { n_samples: 0 }) as Arc<dyn TtsEngine>,
            Box::new(MockVad::new(999)) as Box<dyn VadEngine>,
            Arc::new(std::path::PathBuf::new()),
            config,
        )
        .await
    });

    let (_client, _) = connect_async(&url).await.unwrap();
    // intentionally skip handshake - idle timeout fires before session.config arrives

    tokio::time::advance(Duration::from_secs(60)).await;
    tokio::task::yield_now().await;

    let result = tokio::time::timeout(Duration::from_secs(3), server_task)
        .await
        .expect("server did not exit after idle timeout")
        .expect("server task panicked");

    assert!(result.is_ok());
}
