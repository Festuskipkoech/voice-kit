use std::path::Path;

use anyhow::{Context, Result};
use ort::session::Session;

pub const CHUNK_SAMPLES: usize = 512;
pub const SAMPLE_RATE: i64 = 16000;

/// Pluggable VAD interface. Each session owns one `Box<dyn VadEngine>` -
/// `Sync` is not required, only `Send` for use inside `tokio::spawn`.
pub trait VadEngine: Send {
    /// Feed one fixed-size chunk. Returns `Some(utterance)` at end-of-speech.
    fn process_chunk(&mut self, samples: &[f32]) -> Result<Option<Vec<f32>>>;
    /// Flush buffered speech immediately - call when the session ends.
    fn flush(&mut self) -> Option<Vec<f32>>;
}

/// Silero VAD backed by ONNX Runtime. Owns the session and RNN state.
///
/// v5/v6 ONNX model contract (3 inputs, 2 outputs):
///   inputs:  input [1, 512] f32 | state [2, 1, 128] f32 | sr scalar i64
///   outputs: output [1, 1] f32  | stateN [2, 1, 128] f32
///
/// The state tensor packs h and c (128 values each) into a single [2, 1, 128]
/// tensor. Reset to zeros at the start of every independent stream.
pub struct Vad {
    session: Session,
    threshold: f32,
    silence_chunks: usize,
    state: Vec<f32>, // LSTM state [2, 1, 128] - h and c combined
    speech_buffer: Vec<f32>,
    silence_count: usize,
    in_speech: bool,
}

// ort::Session holds a raw pointer; the runtime itself is thread-safe.
unsafe impl Send for Vad {}
unsafe impl Sync for Vad {}

impl Vad {
    pub fn new(model_path: &Path, threshold: f32, silence_ms: u64) -> Result<Self> {
        let env = crate::engines::ort_env().context("ort environment init")?;
        let session = Session::builder(env)
            .context("ort builder")?
            .commit_from_file(model_path)
            .with_context(|| format!("loading {}", model_path.display()))?;

        let silence_chunks = ((silence_ms as f64 / 1000.0) * SAMPLE_RATE as f64
            / CHUNK_SAMPLES as f64)
            .ceil() as usize;

        Ok(Self {
            session,
            threshold,
            silence_chunks,
            state: vec![0.0_f32; 2 * 1 * 128],
            speech_buffer: Vec::new(),
            silence_count: 0,
            in_speech: false,
        })
    }

    pub fn process_chunk(&mut self, samples: &[f32]) -> Result<Option<Vec<f32>>> {
        debug_assert_eq!(samples.len(), CHUNK_SAMPLES);
        let prob = self.run_silero(samples)?;

        if prob >= self.threshold {
            self.in_speech = true;
            self.silence_count = 0;
            self.speech_buffer.extend_from_slice(samples);
        } else if self.in_speech {
            self.speech_buffer.extend_from_slice(samples);
            self.silence_count += 1;
            if self.silence_count >= self.silence_chunks {
                let trim = self.silence_chunks * CHUNK_SAMPLES;
                let len = self.speech_buffer.len().saturating_sub(trim);
                let utt = self.speech_buffer[..len].to_vec();
                self.reset();
                if !utt.is_empty() {
                    return Ok(Some(utt));
                }
            }
        }

        Ok(None)
    }

    pub fn flush(&mut self) -> Option<Vec<f32>> {
        if self.in_speech && !self.speech_buffer.is_empty() {
            let utt = std::mem::take(&mut self.speech_buffer);
            self.reset();
            Some(utt)
        } else {
            None
        }
    }

    fn reset(&mut self) {
        self.speech_buffer.clear();
        self.silence_count = 0;
        self.in_speech = false;
        self.state.iter_mut().for_each(|x| *x = 0.0);
    }

    fn run_silero(&mut self, samples: &[f32]) -> Result<f32> {
        let audio = ort::value::Value::from_array(([1_usize, CHUNK_SAMPLES], samples.to_vec()))
            .context("audio tensor")?;
        let state_in = ort::value::Value::from_array(([2_usize, 1, 128], self.state.clone()))
            .context("state tensor")?;
        let sr = ort::value::Value::from_array(([] as [usize; 0], vec![SAMPLE_RATE]))
            .context("sr tensor")?;

        let outputs = self
            .session
            .run(ort::inputs![audio, state_in, sr])
            .context("silero inference")?;

        let prob = *outputs[0]
            .try_extract_tensor::<f32>()
            .context("extract prob")?
            .1
            .first()
            .context("empty prob")?;

        self.state.copy_from_slice(
            outputs[1]
                .try_extract_tensor::<f32>()
                .context("extract stateN")?
                .1,
        );

        Ok(prob)
    }
}

impl VadEngine for Vad {
    fn process_chunk(&mut self, samples: &[f32]) -> Result<Option<Vec<f32>>> {
        // Explicit UFCS avoids ambiguity with the trait method of the same name.
        Vad::process_chunk(self, samples)
    }

    fn flush(&mut self) -> Option<Vec<f32>> {
        Vad::flush(self)
    }
}
