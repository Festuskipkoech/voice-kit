use voicekit_core::vad::CHUNK_SAMPLES;
use std::collections::VecDeque;

// MockVad
// Drives the state machine with a fixed probability sequence.
// Keeps the same logic as Vad without touching ONNX.

struct MockVad {
    threshold: f32,
    silence_chunks: usize,
    probs: VecDeque<f32>,
    buffer: Vec<f32>,
    silence_count: usize,
    in_speech: bool,
}

impl MockVad {
    fn new(threshold: f32, silence_ms: u64, probs: Vec<f32>) -> Self {
        let silence_chunks =
            ((silence_ms as f64 / 1000.0) * 16000.0 / CHUNK_SAMPLES as f64).ceil() as usize;
        Self {
            threshold,
            silence_chunks,
            probs: probs.into(),
            buffer: Vec::new(),
            silence_count: 0,
            in_speech: false,
        }
    }

    fn process(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        let prob = self.probs.pop_front().unwrap_or(0.0);

        if prob >= self.threshold {
            self.in_speech = true;
            self.silence_count = 0;
            self.buffer.extend_from_slice(samples);
        } else if self.in_speech {
            self.buffer.extend_from_slice(samples);
            self.silence_count += 1;

            if self.silence_count >= self.silence_chunks {
                let trim = self.silence_chunks * CHUNK_SAMPLES;
                let len  = self.buffer.len().saturating_sub(trim);
                let utt  = self.buffer[..len].to_vec();
                self.buffer.clear();
                self.silence_count = 0;
                self.in_speech = false;
                if !utt.is_empty() {
                    return Some(utt);
                }
            }
        }
        None
    }

    fn flush(&mut self) -> Option<Vec<f32>> {
        if self.in_speech && !self.buffer.is_empty() {
            let utt = std::mem::take(&mut self.buffer);
            self.silence_count = 0;
            self.in_speech = false;
            Some(utt)
        } else {
            None
        }
    }
}

// helpers

fn silence() -> Vec<f32> { vec![0.0; CHUNK_SAMPLES] }

fn speech() -> Vec<f32> {
    (0..CHUNK_SAMPLES)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
        .collect()
}

fn repeat(p: f32, n: usize) -> Vec<f32> { vec![p; n] }

// tests

#[test]
fn silence_never_dispatches() {
    let mut vad = MockVad::new(0.5, 800, repeat(0.0, 50));
    for _ in 0..50 {
        assert!(vad.process(&silence()).is_none());
    }
    assert!(vad.flush().is_none());
}

#[test]
fn speech_then_silence_dispatches_once() {
    let probs: Vec<f32> = repeat(0.9, 10).into_iter().chain(repeat(0.0, 25)).collect();
    let mut vad = MockVad::new(0.5, 800, probs);
    let mut count = 0;

    for _ in 0..10 { vad.process(&speech()); }
    for i in 0..25usize {
        let r = vad.process(&silence());
        if i == 24 { assert!(r.is_some(), "must dispatch on chunk 24"); }
        else        { assert!(r.is_none(), "must not dispatch before chunk 24"); }
        if r.is_some() { count += 1; }
    }

    assert_eq!(count, 1);
}

#[test]
fn silence_counter_resets_on_resumed_speech() {
    let probs: Vec<f32> = repeat(0.9, 10)
        .into_iter()
        .chain(repeat(0.0, 10)) // partial silence - no dispatch
        .chain(repeat(0.9, 10)) // speech resumes - counter resets
        .chain(repeat(0.0, 25)) // full silence - dispatch
        .collect();

    let mut vad = MockVad::new(0.5, 800, probs);
    let mut dispatched = false;

    for _ in 0..10 { assert!(vad.process(&speech()).is_none()); }
    for _ in 0..10 { assert!(vad.process(&silence()).is_none()); }
    for _ in 0..10 { assert!(vad.process(&speech()).is_none()); }
    for _ in 0..25 { if vad.process(&silence()).is_some() { dispatched = true; } }

    assert!(dispatched);
}

#[test]
fn flush_on_empty_returns_none() {
    let mut vad = MockVad::new(0.5, 800, vec![]);
    assert!(vad.flush().is_none());
}

#[test]
fn flush_returns_trailing_speech() {
    let mut vad = MockVad::new(0.5, 800, repeat(0.9, 5));
    for _ in 0..5 { assert!(vad.process(&speech()).is_none()); }
    let utt = vad.flush().expect("flush must return buffered speech");
    assert_eq!(utt.len(), 5 * CHUNK_SAMPLES);
}

#[test]
fn two_utterances_dispatch_twice() {
    let probs: Vec<f32> = repeat(0.9, 5)
        .into_iter()
        .chain(repeat(0.0, 25))
        .chain(repeat(0.9, 5))
        .chain(repeat(0.0, 25))
        .collect();

    let mut vad = MockVad::new(0.5, 800, probs);
    let mut count = 0;
    for _ in 0..60 { if vad.process(&speech()).is_some() { count += 1; } }
    assert_eq!(count, 2);
}

#[test]
fn trailing_silence_trimmed_from_utterance() {
    let probs: Vec<f32> = repeat(0.9, 10).into_iter().chain(repeat(0.0, 25)).collect();
    let mut vad = MockVad::new(0.5, 800, probs);
    let mut utt = None;

    for _ in 0..10 { vad.process(&speech()); }
    for _ in 0..25 { if let Some(u) = vad.process(&silence()) { utt = Some(u); } }

    assert_eq!(utt.unwrap().len(), 10 * CHUNK_SAMPLES);
}