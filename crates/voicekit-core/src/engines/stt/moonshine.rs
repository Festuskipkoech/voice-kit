use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;

use crate::engines::SttEngine;

// Moonshine-tiny constants (from config.json on HuggingFace)
const DECODER_START_TOKEN: i64 = 1;
const EOS_TOKEN: i64 = 2;
const NUM_DECODER_LAYERS: usize = 6;
const NUM_KV_HEADS: usize = 8;
const DIM_KV: usize = 36; // hidden_size(288) / num_attn_heads(8)
const MAX_STEPS_PER_SEC: usize = 6;
const MAX_POSITION_EMBED: usize = 448;

pub struct MoonshineEngine {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    tokenizer: Tokenizer,
}

// Safety: Session holds a raw pointer; the runtime is thread-safe.
// Callers guard mutation with Mutex.
unsafe impl Send for MoonshineEngine {}
unsafe impl Sync for MoonshineEngine {}

impl MoonshineEngine {
    pub fn new(model_path: &Path) -> Result<Self> {
        let enc_path = model_path.join("onnx/encoder_model_quantized.onnx");
        let dec_path = model_path.join("onnx/decoder_model_merged_quantized.onnx");
        let tok_path = model_path.join("tokenizer.json");

        let env = crate::engines::ort_env().context("ort init")?;

        let encoder = Session::builder(env)
            .context("encoder builder")?
            .commit_from_file(&enc_path)
            .with_context(|| format!("loading {}", enc_path.display()))?;

        let decoder = Session::builder(env)
            .context("decoder builder")?
            .commit_from_file(&dec_path)
            .with_context(|| format!("loading {}", dec_path.display()))?;

        let tokenizer = Tokenizer::from_file(&tok_path)
            .with_context(|| format!("loading {}", tok_path.display()))?;

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            tokenizer,
        })
    }
}

impl SttEngine for MoonshineEngine {
    fn transcribe(&self, audio: &[f32]) -> Result<String> {
        let (enc_seq, enc_hidden, enc_data) = {
            let mut enc = self.encoder.lock().unwrap();
            run_encoder(&mut enc, audio)?
        };

        let tokens = {
            let mut dec = self.decoder.lock().unwrap();
            decode_loop(&mut dec, enc_seq, enc_hidden, enc_data, audio.len())?
        };

        Ok(self.tokenizer.decode(&tokens[1..])) // skip start token
    }
}

// Encoder

fn run_encoder(session: &mut Session, audio: &[f32]) -> Result<(usize, usize, Vec<f32>)> {
    let input =
        Value::from_array(([1_usize, audio.len()], audio.to_vec())).context("encoder input")?;

    let out = session.run(ort::inputs![input]).context("encoder run")?;

    // last_hidden_state: [batch=1, seq, hidden=288]
    let (shape, data) = out[0]
        .try_extract_tensor::<f32>()
        .context("encoder output")?;

    Ok((shape[1] as usize, shape[2] as usize, data.to_vec()))
}

// Decoder loop

fn decode_loop(
    session: &mut Session,
    enc_seq: usize,
    enc_hidden: usize,
    enc_data: Vec<f32>,
    audio_len: usize,
) -> Result<Vec<i64>> {
    // KV cache keyed by past_key_values input names; seq starts at 0
    let mut kv: HashMap<String, (Vec<usize>, Vec<f32>)> = past_kv_names()
        .into_iter()
        .map(|name| (name, (vec![1, NUM_KV_HEADS, 0, DIM_KV], vec![])))
        .collect();

    let max_steps = {
        let by_audio = (audio_len as f64 / 16000.0 * MAX_STEPS_PER_SEC as f64) as usize;
        by_audio.min(MAX_POSITION_EMBED)
    };

    let mut tokens: Vec<i64> = vec![DECODER_START_TOKEN];

    for step in 0..max_steps {
        let use_cache = step > 0;
        let next = decoder_step(
            session,
            *tokens.last().unwrap(),
            enc_seq,
            enc_hidden,
            &enc_data,
            use_cache,
            &mut kv,
        )?;

        if next == EOS_TOKEN {
            break;
        }
        tokens.push(next);
    }

    Ok(tokens)
}

fn decoder_step(
    session: &mut Session,
    last_token: i64,
    enc_seq: usize,
    enc_hidden: usize,
    enc_data: &[f32],
    use_cache: bool,
    kv: &mut HashMap<String, (Vec<usize>, Vec<f32>)>,
) -> Result<i64> {
    let mut inputs: Vec<(String, Value)> = Vec::new();

    inputs.push((
        "input_ids".into(),
        Value::from_array(([1_usize, 1], vec![last_token]))
            .context("input_ids")?
            .into(),
    ));
    inputs.push((
        "encoder_hidden_states".into(),
        Value::from_array(([1_usize, enc_seq, enc_hidden], enc_data.to_vec()))
            .context("encoder_hidden_states")?
            .into(),
    ));

    // past_key_values.{layer}.{decoder|encoder}.{key|value}
    for name in past_kv_names() {
        let (shape, data) = kv.get(&name).unwrap();
        inputs.push((
            name.clone(),
            Value::from_array((shape.clone(), data.clone()))
                .with_context(|| format!("kv {name}"))?
                .into(),
        ));
    }

    // use_cache_branch is last per the ONNX model's declared input order
    inputs.push((
        "use_cache_branch".into(),
        Value::from_array(([1_usize], vec![use_cache]))
            .context("use_cache_branch")?
            .into(),
    ));

    let outputs = session.run(inputs).context("decoder run")?;

    // logits: [1, 1, vocab_size] - argmax over vocab dim
    let (_, logits) = outputs[0].try_extract_tensor::<f32>().context("logits")?;
    let next_token = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as i64)
        .context("empty logits")?;

    // Step 0: populate all KV entries (decoder + encoder cross-attn).
    // Step 1+: update decoder entries only; encoder cross-attn is static.
    for (j, past_name) in past_kv_names().iter().enumerate() {
        if use_cache && past_name.contains(".encoder.") {
            continue;
        }
        let out_idx = j + 1;
        if out_idx >= outputs.len() {
            break;
        }
        let (shape, data) = outputs[out_idx]
            .try_extract_tensor::<f32>()
            .context("present kv")?;
        kv.insert(
            past_name.clone(),
            (shape.iter().map(|&x| x as usize).collect(), data.to_vec()),
        );
    }

    Ok(next_token)
}

// Input names: past_key_values.{layer}.{decoder|encoder}.{key|value}
fn past_kv_names() -> Vec<String> {
    let mut v = Vec::with_capacity(NUM_DECODER_LAYERS * 4);
    for layer in 0..NUM_DECODER_LAYERS {
        for module in ["decoder", "encoder"] {
            for kv in ["key", "value"] {
                v.push(format!("past_key_values.{layer}.{module}.{kv}"));
            }
        }
    }
    v
}

// Tokenizer

/// BPE vocab decode - id to token string.
pub struct Tokenizer {
    id_to_token: HashMap<i64, String>,
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let val: serde_json::Value =
            serde_json::from_str(&raw).context("parsing tokenizer.json")?;

        let vocab = val["model"]["vocab"]
            .as_object()
            .context("tokenizer.json: model.vocab not an object")?;

        let id_to_token = vocab
            .iter()
            .map(|(tok, id)| {
                let id = id.as_i64().unwrap_or(0);
                // U+0120 = BPE space prefix; U+010A = newline
                let s = tok.replace('\u{0120}', " ").replace('\u{010A}', "\n");
                (id, s)
            })
            .collect();

        Ok(Self { id_to_token })
    }

    /// Decode token IDs to text, skipping unknowns.
    pub fn decode(&self, ids: &[i64]) -> String {
        ids.iter()
            .filter_map(|id| self.id_to_token.get(id))
            .cloned()
            .collect::<String>()
            .trim()
            .to_string()
    }
}
