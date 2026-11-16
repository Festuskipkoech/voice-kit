"use strict";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SAMPLE_RATE_IN = 16000;
const SAMPLE_RATE_OUT = 24000;
const CHUNK_SAMPLES = 512;
const VAD_FLUSH_CHUNKS = 25; // 25 * 512 / 16000 = 800 ms of silence

const STATE = Object.freeze({
  DISCONNECTED: "DISCONNECTED",
  CONNECTING: "CONNECTING",
  IDLE: "IDLE",
  RECORDING: "RECORDING",
  PROCESSING: "PROCESSING",
  SPEAKING: "SPEAKING",
});

const STATE_LABEL = {
  [STATE.DISCONNECTED]: "Not connected",
  [STATE.CONNECTING]: "Connecting...",
  [STATE.IDLE]: "Press to speak",
  [STATE.RECORDING]: "Listening...",
  [STATE.PROCESSING]: "Thinking...",
  [STATE.SPEAKING]: "Speaking",
};

// AudioWorklet source — bundled as a Blob URL so no separate file is needed.
const WORKLET_SRC = `
class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(${CHUNK_SAMPLES});
    this._pos = 0;
    this._active = false;
    this.port.onmessage = (e) => { this._active = e.data === 'start'; };
  }
  process(inputs) {
    if (!this._active) return true;
    const ch = inputs[0]?.[0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this._buf[this._pos++] = ch[i];
      if (this._pos === ${CHUNK_SAMPLES}) {
        this.port.postMessage(this._buf.buffer, [this._buf.buffer]);
        this._buf = new Float32Array(${CHUNK_SAMPLES});
        this._pos = 0;
      }
    }
    return true;
  }
}
registerProcessor('pcm-capture', PcmCaptureProcessor);
`;

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

const els = {
  statusBadge: $("status-badge"),
  statusText: $("status-text"),
  statusDot: $("status-dot"),
  micWrap: $("mic-wrap"),
  micBtn: $("mic-btn"),
  micIcon: $("mic-icon"),
  stopIcon: $("stop-icon"),
  spinner: $("spinner"),
  waveIcon: $("wave-icon"),
  stateLabel: $("state-label"),
  transcript: $("transcript"),
  clearBtn: $("clear-btn"),
  serverInput: $("server-input"),
  connectBtn: $("connect-btn"),
  canvas: $("canvas"),
};

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

let state = STATE.DISCONNECTED;
let ws = null;
let audioCtx = null;
let analyser = null;
let workletNode = null;
let micStream = null;
let playQueue = []; // { buffer: Float32Array, sampleRate: number }[]
let nextPlayAt = 0;
let animFrame = null;
let canvasCtx = els.canvas.getContext("2d");

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

function transition(next) {
  if (state === next) return;
  state = next;
  render();
}

function render() {
  const s = state;

  // Status badge
  els.statusText.textContent = STATE_LABEL[s] ?? s;
  els.statusBadge.className = "status-badge";
  if (
    s === STATE.IDLE ||
    s === STATE.RECORDING ||
    s === STATE.PROCESSING ||
    s === STATE.SPEAKING
  ) {
    els.statusBadge.classList.add("connected");
  }
  if (s === STATE.RECORDING) els.statusBadge.classList.add("recording");

  // Mic wrap classes
  els.micWrap.className = "mic-wrap";
  if (s === STATE.RECORDING) els.micWrap.classList.add("recording");

  // Mic button class + aria
  els.micBtn.className = "mic-btn";
  if (s === STATE.IDLE) els.micBtn.classList.add("state-idle");
  if (s === STATE.RECORDING) els.micBtn.classList.add("state-recording");
  if (s === STATE.PROCESSING) els.micBtn.classList.add("state-processing");
  if (s === STATE.SPEAKING) els.micBtn.classList.add("state-speaking");

  els.micBtn.disabled = s === STATE.DISCONNECTED || s === STATE.CONNECTING;
  els.micBtn.setAttribute(
    "aria-label",
    s === STATE.RECORDING ? "Stop recording" : "Start recording",
  );

  // Icon visibility
  const show = (el) => el.classList.remove("hidden");
  const hide = (el) => el.classList.add("hidden");

  hide(els.micIcon);
  hide(els.stopIcon);
  hide(els.spinner);
  hide(els.waveIcon);
  if (s === STATE.IDLE) show(els.micIcon);
  else if (s === STATE.RECORDING) show(els.stopIcon);
  else if (s === STATE.PROCESSING || s === STATE.CONNECTING) show(els.spinner);
  else if (s === STATE.SPEAKING) show(els.waveIcon);
  else show(els.micIcon);

  // State label
  els.stateLabel.textContent = STATE_LABEL[s] ?? "";
  els.stateLabel.className = "state-label";
  if (s !== STATE.DISCONNECTED && s !== STATE.CONNECTING) {
    els.stateLabel.classList.add("active");
  }

  // Connect button label
  const isConnected = s !== STATE.DISCONNECTED && s !== STATE.CONNECTING;
  els.connectBtn.textContent = isConnected ? "Disconnect" : "Connect";

  // Server input readonly while connected
  els.serverInput.readOnly = isConnected;
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

function connect() {
  const url = els.serverInput.value.trim();
  if (!url) return;

  transition(STATE.CONNECTING);

  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    transition(STATE.IDLE);
  };

  ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) {
      handleAudio(ev.data);
    }
  };

  ws.onerror = () => {
    appendSystem("Connection error.");
    disconnect();
  };

  ws.onclose = () => {
    if (state !== STATE.DISCONNECTED) {
      appendSystem("Disconnected from server.");
      disconnect();
    }
  };
}

function disconnect() {
  stopRecording();

  if (ws) {
    ws.onclose = null;
    ws.onerror = null;
    ws.close();
    ws = null;
  }

  playQueue = [];
  nextPlayAt = 0;

  transition(STATE.DISCONNECTED);
}

// ---------------------------------------------------------------------------
// Microphone & AudioWorklet
// ---------------------------------------------------------------------------

async function ensureAudioContext() {
  if (audioCtx && audioCtx.state !== "closed") return;

  audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE_IN });

  // Analyser for the visualiser
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  analyser.smoothingTimeConstant = 0.75;

  // Load worklet from Blob URL
  const blob = new Blob([WORKLET_SRC], { type: "application/javascript" });
  const url = URL.createObjectURL(blob);
  await audioCtx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);
}

async function startRecording() {
  if (state !== STATE.IDLE) return;
  transition(STATE.RECORDING);

  try {
    await ensureAudioContext();
    if (audioCtx.state === "suspended") await audioCtx.resume();

    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: SAMPLE_RATE_IN,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    const source = audioCtx.createMediaStreamSource(micStream);
    source.connect(analyser);

    workletNode = new AudioWorkletNode(audioCtx, "pcm-capture");
    workletNode.port.onmessage = (e) => {
      if (state !== STATE.RECORDING || !ws || ws.readyState !== WebSocket.OPEN)
        return;
      ws.send(e.data);
    };

    source.connect(workletNode);
    workletNode.port.postMessage("start");

    startVisualiser();
  } catch (err) {
    console.error("microphone error", err);
    appendSystem(`Microphone access denied: ${err.message}`);
    transition(STATE.IDLE);
  }
}

function flushVad() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  for (let i = 0; i < VAD_FLUSH_CHUNKS; i++) {
    ws.send(new Float32Array(CHUNK_SAMPLES).buffer);
  }
}

function stopRecording() {
  if (workletNode) {
    workletNode.port.postMessage("stop");
    workletNode.disconnect();
    workletNode = null;
  }
  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
  }
  flushVad();
  stopVisualiser();
}

function handleMicButton() {
  if (state === STATE.RECORDING) {
    stopRecording();
    transition(STATE.PROCESSING);
  } else if (state === STATE.IDLE) {
    startRecording();
  }
}

// ---------------------------------------------------------------------------
// Audio playback (f32-LE PCM from server at 24kHz)
// ---------------------------------------------------------------------------

function handleAudio(buffer) {
  if (state === STATE.PROCESSING || state === STATE.SPEAKING) {
    transition(STATE.SPEAKING);
  }

  const samples = new Float32Array(buffer);
  const duration = samples.length / SAMPLE_RATE_OUT;

  if (!audioCtx || audioCtx.state === "closed") return;

  // Reuse existing context or resume
  if (audioCtx.state === "suspended") audioCtx.resume();

  const ab = audioCtx.createBuffer(1, samples.length, SAMPLE_RATE_OUT);
  ab.copyToChannel(samples, 0);

  const src = audioCtx.createBufferSource();
  src.buffer = ab;
  src.connect(audioCtx.destination);

  const now = audioCtx.currentTime;
  const at = Math.max(now, nextPlayAt);
  src.start(at);
  nextPlayAt = at + duration;

  // When the last queued chunk finishes, go back to IDLE
  src.onended = () => {
    if (audioCtx && audioCtx.currentTime >= nextPlayAt - 0.05) {
      if (state === STATE.SPEAKING) transition(STATE.IDLE);
    }
  };

  // Append assistant turn on first chunk of a response
  if (state === STATE.SPEAKING && nextPlayAt - duration <= now + 0.05) {
    appendAssistantTurn();
  }
}

// ---------------------------------------------------------------------------
// Visualiser (circular frequency bars)
// ---------------------------------------------------------------------------

function startVisualiser() {
  cancelAnimationFrame(animFrame);
  drawVisualiser();
}

function stopVisualiser() {
  cancelAnimationFrame(animFrame);
  animFrame = null;
  clearCanvas();
}

function clearCanvas() {
  const c = els.canvas;
  canvasCtx.clearRect(0, 0, c.width, c.height);
}

function drawVisualiser() {
  animFrame = requestAnimationFrame(drawVisualiser);

  if (!analyser) {
    clearCanvas();
    return;
  }

  const c = els.canvas;
  const w = (c.width = c.offsetWidth);
  const h = (c.height = c.offsetHeight);
  const cx = w / 2;
  const cy = h / 2;
  const ctx = canvasCtx;

  ctx.clearRect(0, 0, w, h);

  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(data);

  const bars = 60;
  const radius = Math.min(w, h) * 0.26;
  const barMax = Math.min(w, h) * 0.14;
  const step = Math.floor(data.length / bars);
  const slice = (Math.PI * 2) / bars;

  for (let i = 0; i < bars; i++) {
    const v = data[i * step] / 255;
    const barH = 3 + v * barMax;
    const ang = slice * i - Math.PI / 2;
    const x1 = cx + Math.cos(ang) * radius;
    const y1 = cy + Math.sin(ang) * radius;
    const x2 = cx + Math.cos(ang) * (radius + barH);
    const y2 = cy + Math.sin(ang) * (radius + barH);

    // Gradient per bar: accent → cyan based on frequency intensity
    const r = Math.round(123 + (34 - 123) * v);
    const g = Math.round(110 + (211 - 110) * v);
    const b = Math.round(246 + (238 - 246) * v);
    ctx.strokeStyle = `rgba(${r},${g},${b},${0.55 + v * 0.45})`;
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
}

// ---------------------------------------------------------------------------
// Transcript
// ---------------------------------------------------------------------------

let _assistantRow = null;

function appendUserTurn(text) {
  _assistantRow = null;
  removePlaceholder();

  const turn = document.createElement("div");
  turn.className = "turn";
  turn.innerHTML = `
    <span class="turn-role user">You</span>
    <div class="turn-body">
      <span class="turn-text">${escHtml(text)}</span>
      <span class="turn-meta">${timestamp()}</span>
    </div>`;
  els.transcript.appendChild(turn);
  scrollTranscript();
}

function appendAssistantTurn() {
  if (_assistantRow) return;
  removePlaceholder();

  const turn = document.createElement("div");
  turn.className = "turn";
  turn.innerHTML = `
    <span class="turn-role assistant">Assistant</span>
    <div class="turn-body">
      <div class="audio-bar">
        <div class="audio-bar-icon">
          <span></span><span></span><span></span><span></span><span></span>
        </div>
      </div>
      <span class="turn-meta">${timestamp()}</span>
    </div>`;
  els.transcript.appendChild(turn);
  _assistantRow = turn;
  scrollTranscript();
}

function appendSystem(msg) {
  removePlaceholder();
  const el = document.createElement("div");
  el.className = "turn";
  el.innerHTML = `
    <span class="turn-role" style="color:var(--text-dim)">System</span>
    <div class="turn-body">
      <span class="turn-text" style="color:var(--text-dim);font-style:italic">${escHtml(msg)}</span>
    </div>`;
  els.transcript.appendChild(el);
  scrollTranscript();
}

function removePlaceholder() {
  const ph = els.transcript.querySelector(".transcript-empty");
  if (ph) ph.remove();
}

function scrollTranscript() {
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function clearTranscript() {
  _assistantRow = null;
  els.transcript.innerHTML =
    '<p class="transcript-empty">Transcript will appear here</p>';
}

function timestamp() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function escHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ---------------------------------------------------------------------------
// Connect / disconnect button
// ---------------------------------------------------------------------------

function handleConnectButton() {
  if (state === STATE.DISCONNECTED) {
    connect();
  } else {
    disconnect();
  }
}

// ---------------------------------------------------------------------------
// User turn detection
// When recording stops we approximate a "user turn" with a timing placeholder.
// Real STT transcripts would arrive via a text WebSocket message; for now we
// add a marker so the transcript reflects activity.
// ---------------------------------------------------------------------------

function onRecordingStop() {
  appendUserTurn("(audio sent)");
}

// ---------------------------------------------------------------------------
// Wire up events
// ---------------------------------------------------------------------------

els.micBtn.addEventListener("click", () => {
  if (state === STATE.RECORDING) {
    onRecordingStop();
  }
  handleMicButton();
});

els.connectBtn.addEventListener("click", handleConnectButton);

els.clearBtn.addEventListener("click", clearTranscript);

els.serverInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && state === STATE.DISCONNECTED) connect();
});

// Keyboard shortcut: Space to toggle recording when focused outside input
document.addEventListener("keydown", (e) => {
  if (e.target === els.serverInput) return;
  if (e.code === "Space" && !e.repeat) {
    e.preventDefault();
    if (state === STATE.RECORDING) {
      onRecordingStop();
      handleMicButton();
    } else if (state === STATE.IDLE) {
      handleMicButton();
    }
  }
  if (e.code === "Escape" && state === STATE.RECORDING) {
    onRecordingStop();
    handleMicButton();
  }
});

// Initial render
render();
