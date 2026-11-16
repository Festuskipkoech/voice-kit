use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use misaki_rs::{G2P, Language};
use ort::session::Session;
use ort::value::{DynValue, Value};

use crate::config::TtsConfig;
use crate::engines::TtsEngine;

// 0 = BOS/EOS pad token; model accepts at most 512 tokens.
const PAD_ID: i64 = 0;
const MAX_LEN: usize = 512;

pub struct ParadeeEngine {
    session: Mutex<Session>,
    g2p: Mutex<G2P>,
    // phoneme char to integer ID, loaded from config.json at construction.
    vocab: HashMap<char, i64>,
}

unsafe impl Send for ParadeeEngine {}
unsafe impl Sync for ParadeeEngine {}

impl ParadeeEngine {
    pub fn new(model_path: &Path, cfg: &TtsConfig) -> Result<Self> {
        let onnx_path = model_path.join("onnx/paradee_int8.onnx");
        let cfg_path = model_path.join("config.json");

        anyhow::ensure!(
            onnx_path.exists(),
            "Paradee ONNX not found: {}",
            onnx_path.display()
        );
        anyhow::ensure!(
            cfg_path.exists(),
            "config.json not found: {}",
            cfg_path.display()
        );

        // cfg is retained for future per-instance tuning (sample_rate etc.);
        // currently only voice is used at synthesis time, passed per call.
        let _ = cfg;

        let env = crate::engines::ort_env().context("ort init")?;
        let session = Session::builder(env)
            .context("session builder")?
            .commit_from_file(&onnx_path)
            .with_context(|| format!("loading {}", onnx_path.display()))?;

        let vocab = load_vocab(&cfg_path)?;

        Ok(Self {
            session: Mutex::new(session),
            g2p: Mutex::new(G2P::new(Language::EnglishUS)),
            vocab,
        })
    }
}

impl TtsEngine for ParadeeEngine {
    fn synthesize(&self, text: &str, voice: &str) -> Result<Vec<f32>> {
        let voice_id: i64 = match voice {
            "af_heart" => 0,
            "bf_isabella" => 1,
            "am_puck" => 2,
            "bm_fable" => 3,
            other => anyhow::bail!("unknown Paradee voice: {other}"),
        };

        let phonemes = {
            let g2p = self.g2p.lock().unwrap();
            let (ph, _) = g2p.g2p(text).context("g2p")?;
            ph
        };

        let ids = text_to_ids(&phonemes, &self.vocab);
        anyhow::ensure!(!ids.is_empty(), "phoneme sequence empty for: {text:?}");

        let t = ids.len();
        let inputs: Vec<(String, DynValue)> = vec![
            (
                "input_ids".to_string(),
                Value::from_array(([1_usize, t], ids))
                    .context("input_ids")?
                    .into(),
            ),
            (
                "speed".to_string(),
                Value::from_array(([1_usize], vec![1.0_f32]))
                    .context("speed")?
                    .into(),
            ),
            (
                "voice".to_string(),
                Value::from_array(([1_usize], vec![voice_id]))
                    .context("voice")?
                    .into(),
            ),
        ];

        let mut sess = self.session.lock().unwrap();
        let outputs = sess.run(inputs).context("paradee run")?;

        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("waveform output")?;

        Ok(data.to_vec())
    }
}

// Helpers

/// Parse config.json and extract the "vocab" map: phoneme char to token ID.
fn load_vocab(path: &Path) -> Result<HashMap<char, i64>> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let val: serde_json::Value = serde_json::from_str(&raw).context("parsing config.json")?;

    let obj = val["vocab"]
        .as_object()
        .context("config.json: 'vocab' is not an object")?;

    let vocab = obj
        .iter()
        .filter_map(|(k, v)| {
            // Each key is a single phoneme symbol (one Unicode char).
            let ch = k.chars().next()?;
            let id = v.as_i64()?;
            Some((ch, id))
        })
        .collect();

    Ok(vocab)
}

/// Convert a phoneme string to padded token IDs.
/// Unknown chars are silently skipped. Result is capped at MAX_LEN.
pub fn text_to_ids(phonemes: &str, vocab: &HashMap<char, i64>) -> Vec<i64> {
    let mut ids: Vec<i64> = std::iter::once(PAD_ID)
        .chain(phonemes.chars().filter_map(|c| vocab.get(&c).copied()))
        .chain(std::iter::once(PAD_ID))
        .collect();

    if ids.len() > MAX_LEN {
        ids.truncate(MAX_LEN - 1);
        ids.push(PAD_ID);
    }
    ids
}
