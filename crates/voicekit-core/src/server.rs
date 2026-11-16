use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tokio_tungstenite::accept_async;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::config::Config;
use crate::engines::{LlmEngine, SttEngine, TtsEngine, build_llm, build_stt, build_tts, build_vad};
use crate::session::run_session;
use crate::vad::VadEngine;

pub struct ResolvedPaths {
    pub vad: PathBuf,
    pub stt: PathBuf,
    pub llm: PathBuf,
    pub tts: PathBuf,
}

pub async fn run(config: Arc<Config>, paths: ResolvedPaths) -> Result<()> {
    let stt: Arc<dyn SttEngine> =
        Arc::from(build_stt(&config.stt, &paths.stt).context("building STT")?);
    let llm: Arc<dyn LlmEngine> =
        Arc::from(build_llm(&config.llm, &paths.llm).context("building LLM")?);
    let tts: Arc<dyn TtsEngine> =
        Arc::from(build_tts(&config.tts, &paths.tts).context("building TTS")?);

    // validate VAD path once; each session gets its own stateful instance
    build_vad(&config.vad, &paths.vad).context("validating VAD model")?;
    info!("all engines loaded");

    let vad_path = Arc::new(paths.vad);

    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port)
        .parse()
        .context("parsing server address")?;

    let listener = TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding {addr}"))?;

    info!(addr = %addr, "listening");

    let max_conn = config.server.max_connections;
    let semaphore = Arc::new(Semaphore::new(max_conn));
    info!(max_connections = max_conn, "connection limit configured");

    let shutdown = CancellationToken::new();
    {
        let token = shutdown.clone();
        tokio::spawn(async move {
            wait_for_shutdown_signal().await;
            info!("shutdown signal received - stopping accept loop");
            token.cancel();
        });
    }

    loop {
        tokio::select! {
            biased;

            _ = shutdown.cancelled() => {
                info!("accept loop stopped");
                break;
            }

            result = listener.accept() => {
                let (tcp, peer) = result.context("accept")?;

                let permit = match Arc::clone(&semaphore).try_acquire_owned() {
                    Ok(p)  => p,
                    Err(_) => {
                        warn!(peer = %peer, limit = max_conn, "connection limit reached, dropping");
                        continue;
                    }
                };

                info!(peer = %peer, "connection accepted");

                let vad: Box<dyn VadEngine> = match build_vad(&config.vad, &vad_path) {
                    Ok(v)  => v,
                    Err(e) => {
                        warn!(peer = %peer, "failed to build VAD, dropping connection: {e}");
                        continue;
                    }
                };

                let (stt, llm, tts, vad_path, config) = (
                    Arc::clone(&stt),
                    Arc::clone(&llm),
                    Arc::clone(&tts),
                    Arc::clone(&vad_path),
                    Arc::clone(&config),
                );

                tokio::spawn(async move {
                    let _permit = permit;

                    let ws = match accept_async(tcp).await {
                        Ok(ws) => ws,
                        Err(e) => {
                            warn!(peer = %peer, "WebSocket handshake failed: {e}");
                            return;
                        }
                    };

                    match run_session(ws, stt, llm, tts, vad, Arc::clone(&vad_path), config).await {
                        Ok(())  => info!(peer = %peer, "session closed"),
                        Err(e)  => error!(peer = %peer, "session error: {e}"),
                    }
                });
            }
        }
    }

    info!("draining active sessions (up to 30 s)...");
    match tokio::time::timeout(
        std::time::Duration::from_secs(30),
        semaphore.acquire_many(max_conn as u32),
    )
    .await
    {
        Ok(_) => info!("all sessions closed cleanly"),
        Err(_) => warn!("drain timeout - some sessions may have been cut short"),
    }

    info!("server stopped");
    Ok(())
}

async fn wait_for_shutdown_signal() {
    use tokio::signal;

    #[cfg(unix)]
    {
        use signal::unix::{SignalKind, signal};
        let mut sigterm =
            signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
        tokio::select! {
            _ = sigterm.recv()    => {}
            _ = signal::ctrl_c() => {}
        }
    }

    #[cfg(not(unix))]
    {
        let _ = signal::ctrl_c().await;
    }
}
