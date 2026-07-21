// voicekit — voice test UI
// Protocol: ws://localhost:8000/session
// t0 = performance.now() when end_of_speech is sent; all client latency measured from t0

// constants
const GATEWAY_URL     = 'ws://localhost:8000/session';
const SAMPLE_RATE_IN  = 16000;
const SAMPLE_RATE_OUT = 24000;
const CHUNK_SAMPLES   = 1600; // 100ms at 16kHz

// DOM
const statusBadge  = document.getElementById('status-badge');
const recordBtn    = document.getElementById('record-btn');
const recordLabel  = document.getElementById('record-label');
const hintText     = document.getElementById('hint-text');
const oscCanvas    = document.getElementById('osc-canvas');
const userText     = document.getElementById('user-text');
const agentText    = document.getElementById('agent-text');
const errorBar     = document.getElementById('error-bar');
const totalLabel   = document.getElementById('total-label');
const sessionIdEl  = document.getElementById('session-id');
const turnCountEl  = document.getElementById('turn-count');
const chunkCountEl = document.getElementById('chunk-count');

const metricEls = {
  stt:        [document.getElementById('bar-stt'),         document.getElementById('val-stt')],
  llm:        [document.getElementById('bar-llm'),         document.getElementById('val-llm')],
  tts:        [document.getElementById('bar-tts'),         document.getElementById('val-tts')],
  total:      [document.getElementById('bar-total'),       document.getElementById('val-total')],
  firstAudio: [document.getElementById('bar-first-audio'), document.getElementById('val-first-audio')],
  transcript: [document.getElementById('bar-transcript'),  document.getElementById('val-transcript-latency')],
  roundTrip:  [document.getElementById('bar-roundtrip'),   document.getElementById('val-roundtrip')],
  network:    [document.getElementById('bar-network'),     document.getElementById('val-network')],
};

// state
let ws          = null;
let mediaStream = null;
let audioCtx    = null;
let sourceNode  = null;
let workletNode = null;
let workletBlobUrl = null;
let listening   = false;
let processing  = false;
let sessionId   = null;
let turnCount   = 0;
let chunkCount  = 0;

// timing — all relative to t0
let t0           = 0;
let firstAudioMs = null;
let transcriptMs = null;
let roundTripMs  = null;

// playback
let isFirstChunk = true;
let playbackCtx  = null;
let nextPlayTime = 0;

// oscilloscope
const oscCtx = oscCanvas.getContext('2d');
let oscAnimId = null;
let oscData   = new Float32Array(128);


// status

function setStatus(state) {
  statusBadge.className = 'status-badge status-' + state;
  statusBadge.textContent = state;
}

function setHint(text) {
  hintText.textContent = text;
}

function showError(msg) {
  errorBar.textContent = msg;
  errorBar.classList.add('visible');
  setTimeout(() => errorBar.classList.remove('visible'), 6000);
}

function hideError() {
  errorBar.classList.remove('visible');
}


// oscilloscope

function drawOsc() {
  const w = oscCanvas.width;
  const h = oscCanvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const baseR = w / 2 - 2;

  oscCtx.clearRect(0, 0, w, h);

  const ringColor = listening
    ? 'rgba(255,107,53,'
    : processing
      ? 'rgba(0,212,255,'
      : 'rgba(42,42,56,';

  const segments = oscData.length;
  const angleStep = (Math.PI * 2) / segments;

  oscCtx.beginPath();
  for (let i = 0; i <= segments; i++) {
    const idx = i % segments;
    const amp = listening
      ? Math.abs(oscData[idx]) * 30
      : processing
        ? 6 + Math.sin(Date.now() / 200 + idx * 0.3) * 6
        : 0;
    const r     = baseR + amp;
    const angle = angleStep * i - Math.PI / 2;
    const x     = cx + r * Math.cos(angle);
    const y     = cy + r * Math.sin(angle);
    if (i === 0) oscCtx.moveTo(x, y);
    else oscCtx.lineTo(x, y);
  }
  oscCtx.closePath();

  const opacity = listening ? 0.9 : (processing ? 0.7 : 0.4);
  oscCtx.strokeStyle = ringColor + opacity + ')';
  oscCtx.lineWidth   = listening ? 2.5 : 1.5;
  oscCtx.stroke();

  if (listening || processing) {
    oscCtx.beginPath();
    oscCtx.arc(cx, cy, baseR - 14, 0, Math.PI * 2);
    oscCtx.strokeStyle = ringColor + '0.1)';
    oscCtx.lineWidth   = 12;
    oscCtx.stroke();
  }

  oscAnimId = requestAnimationFrame(drawOsc);
}

function startOsc() {
  if (!oscAnimId) drawOsc();
}

function stopOsc() {
  if (oscAnimId) { cancelAnimationFrame(oscAnimId); oscAnimId = null; }
  oscCtx.clearRect(0, 0, oscCanvas.width, oscCanvas.height);
}


// WebSocket

function connect() {
  ws = new WebSocket(GATEWAY_URL);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {};

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      handleAudioChunk(event.data);
      return;
    }

    let msg;
    try { msg = JSON.parse(event.data); } catch { return; }

    switch (msg.type) {
      case 'ready':
        sessionId = msg.session_id;
        sessionIdEl.textContent = sessionId || '—';
        setStatus('connected');
        enableRecord();
        setHint('hold to speak — release to send');
        break;

      case 'ping':
        ws.send(JSON.stringify({ type: 'pong' }));
        break;

      case 'transcript':
        transcriptMs = performance.now() - t0;
        userText.textContent = msg.text || '—';
        break;

      case 'response':
        agentText.textContent = msg.text || '—';
        break;

      case 'metrics':
        roundTripMs = performance.now() - t0;
        handleMetrics(msg);
        break;

      case 'error':
        showError(msg.message || 'unknown error');
        resetToReady();
        break;
    }
  };

  ws.onclose = () => {
    setStatus('disconnected');
    disableRecord('disconnected');
    setHint('connection closed — reload to reconnect');
    stopOsc();
    stopMic();
  };

  ws.onerror = () => {
    showError('WebSocket error — is the gateway running at ' + GATEWAY_URL + '?');
    setStatus('disconnected');
    disableRecord('error');
  };
}


// metrics

function handleMetrics(msg) {
  const sttMs   = msg.stt_ms             || 0;
  const llmMs   = msg.llm_first_token_ms || 0;
  const ttsMs   = msg.tts_first_chunk_ms || 0;
  const totalMs = msg.total_ms           || 0;

  // ceiling scales all bars; round-trip can exceed server total due to network
  const ceiling = Math.max(totalMs, roundTripMs || 0, firstAudioMs || 0) * 1.05;

  function pct(ms) {
    return ceiling > 0 ? Math.min((ms / ceiling) * 100, 100).toFixed(1) + '%' : '0%';
  }

  function fmt(ms) {
    if (ms == null) return '—';
    return ms >= 1000 ? (ms / 1000).toFixed(2) + 's' : Math.round(ms) + 'ms';
  }

  function set(key, ms) {
    const [bar, val] = metricEls[key];
    bar.style.width  = pct(ms);
    val.textContent  = fmt(ms);
  }

  set('stt',        sttMs);
  set('llm',        llmMs);
  set('tts',        ttsMs);
  set('total',      totalMs);
  set('firstAudio', firstAudioMs);
  set('transcript', transcriptMs);
  set('roundTrip',  roundTripMs);
  set('network',    roundTripMs != null ? Math.max(0, roundTripMs - totalMs) : null);

  totalLabel.textContent  = fmt(totalMs);
  turnCountEl.textContent = ++turnCount;
  chunkCountEl.textContent = chunkCount;

  resetToReady();
}


// AudioWorklet mic capture
// Runs on the audio rendering thread (not main thread) — no UI jank.
// Inlined as a Blob URL so it works from file:// without CORS issues.
// ${CHUNK_SAMPLES} is interpolated at blob creation time — the worklet
// scope is isolated and cannot reference main-thread variables directly.

const WORKLET_SOURCE = `
class VoiceCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(${CHUNK_SAMPLES});
    this._filled = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    // oscilloscope snapshot — 64 samples per quantum
    const snap = new Float32Array(64);
    const step = Math.max(1, Math.floor(channel.length / 64));
    for (let i = 0; i < 64; i++) snap[i] = channel[i * step] || 0;
    this.port.postMessage({ type: 'osc', snap }, [snap.buffer]);

    // accumulate into chunk buffer; transfer when full (zero-copy)
    let offset = 0;
    while (offset < channel.length) {
      const space = ${CHUNK_SAMPLES} - this._filled;
      const take  = Math.min(space, channel.length - offset);
      this._buffer.set(channel.subarray(offset, offset + take), this._filled);
      this._filled += take;
      offset       += take;

      if (this._filled === ${CHUNK_SAMPLES}) {
        this.port.postMessage({ type: 'chunk', buffer: this._buffer.buffer }, [this._buffer.buffer]);
        this._buffer = new Float32Array(${CHUNK_SAMPLES});
        this._filled = 0;
      }
    }

    return true;
  }
}

registerProcessor('voice-capture', VoiceCaptureProcessor);
`;

async function startMic() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: SAMPLE_RATE_IN, channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });
  } catch (err) {
    showError('Microphone access denied: ' + err.message);
    return false;
  }

  audioCtx   = new AudioContext({ sampleRate: SAMPLE_RATE_IN });
  sourceNode = audioCtx.createMediaStreamSource(mediaStream);

  workletBlobUrl = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'application/javascript' }));

  try {
    await audioCtx.audioWorklet.addModule(workletBlobUrl);
  } catch (err) {
    showError('AudioWorklet failed to load: ' + err.message);
    return false;
  }

  workletNode = new AudioWorkletNode(audioCtx, 'voice-capture');

  workletNode.port.onmessage = (event) => {
    if (!listening) return;
    if (event.data.type === 'osc') {
      oscData = new Float32Array(event.data.snap);
    } else if (event.data.type === 'chunk') {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(event.data.buffer);
    }
  };

  sourceNode.connect(workletNode); // capture only — not connected to destination

  return true;
}

function stopMic() {
  if (workletNode)    { workletNode.port.onmessage = null; workletNode.disconnect(); workletNode = null; }
  if (sourceNode)     { sourceNode.disconnect(); sourceNode = null; }
  if (audioCtx)       { audioCtx.close(); audioCtx = null; }
  if (mediaStream)    { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  if (workletBlobUrl) { URL.revokeObjectURL(workletBlobUrl); workletBlobUrl = null; }
  oscData = new Float32Array(128);
}


// audio playback

function initPlaybackCtx() {
  if (!playbackCtx || playbackCtx.state === 'closed') {
    playbackCtx  = new AudioContext({ sampleRate: SAMPLE_RATE_OUT });
    nextPlayTime = 0;
  }
}

async function handleAudioChunk(buffer) {
  if (!firstAudioMs && t0 > 0) firstAudioMs = performance.now() - t0;
  chunkCount++;
  initPlaybackCtx();

  let samples;
  if (isFirstChunk) {
    // chunk 1 = complete WAV (header + float32 PCM); decode normally
    isFirstChunk = false;
    try {
      const decoded = await playbackCtx.decodeAudioData(buffer.slice(0));
      samples = decoded.getChannelData(0);
    } catch {
      samples = new Float32Array(buffer); // fallback: treat as raw PCM
    }
  } else {
    // chunks 2..N = raw float32 PCM, no header
    samples = new Float32Array(buffer);
  }

  schedulePlayback(samples);
}

function schedulePlayback(samples) {
  if (!playbackCtx) return;
  const buf    = playbackCtx.createBuffer(1, samples.length, SAMPLE_RATE_OUT);
  buf.getChannelData(0).set(samples);
  const source = playbackCtx.createBufferSource();
  source.buffer = buf;
  source.connect(playbackCtx.destination);
  const startAt = Math.max(playbackCtx.currentTime, nextPlayTime);
  source.start(startAt);
  nextPlayTime = startAt + samples.length / SAMPLE_RATE_OUT;
}

function resetPlayback() {
  isFirstChunk = true;
  firstAudioMs = null;
  chunkCount   = 0;
  nextPlayTime = 0;
}


// record button

function enableRecord() {
  recordBtn.disabled      = false;
  recordBtn.className     = 'record-btn';
  recordLabel.textContent = 'hold to speak';
}

function disableRecord(label) {
  recordBtn.disabled      = true;
  recordBtn.className     = 'record-btn';
  recordLabel.textContent = label || 'unavailable';
}

function setListeningState() {
  listening = true;
  processing = false;
  recordBtn.className     = 'record-btn listening';
  recordLabel.textContent = 'listening';
  setStatus('listening');
  setHint('release to send');
  startOsc();
}

function setProcessingState() {
  listening  = false;
  processing = true;
  recordBtn.className     = 'record-btn processing';
  recordBtn.disabled      = true;
  recordLabel.textContent = 'processing';
  setStatus('processing');
  setHint('waiting for response…');
  oscData = new Float32Array(128);
}

function resetToReady() {
  listening  = false;
  processing = false;
  recordBtn.className     = 'record-btn';
  recordBtn.disabled      = false;
  recordLabel.textContent = 'hold to speak';
  setStatus('connected');
  setHint('hold to speak — release to send');
  resetPlayback();
  stopMic();
}


// event handlers

async function onRecordStart(event) {
  event.preventDefault();
  if (listening || processing || !ws || ws.readyState !== WebSocket.OPEN) return;

  hideError();
  firstAudioMs = null;
  transcriptMs = null;
  roundTripMs  = null;
  t0           = 0;
  userText.textContent   = '—';
  agentText.textContent  = '—';
  totalLabel.textContent = '—';

  const ok = await startMic();
  if (!ok) return;
  setListeningState();
}

function onRecordEnd(event) {
  event.preventDefault();
  if (!listening) return;
  if (ws && ws.readyState === WebSocket.OPEN) {
    t0 = performance.now();
    ws.send(JSON.stringify({ type: 'end_of_speech' }));
  }
  stopMic();
  setProcessingState();
}

recordBtn.addEventListener('pointerdown',  onRecordStart);
recordBtn.addEventListener('pointerup',    onRecordEnd);
recordBtn.addEventListener('pointerleave', onRecordEnd);

// Space bar to record
document.addEventListener('keydown', async (e) => {
  if (e.code === 'Space' && !e.repeat && !recordBtn.disabled) await onRecordStart(e);
});
document.addEventListener('keyup', (e) => {
  if (e.code === 'Space') onRecordEnd(e);
});


// init
setStatus('disconnected');
disableRecord('connecting');
startOsc();
connect();