use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::sampling::LlamaSampler;
use tokio::sync::mpsc::Sender;

use crate::config::LlmConfig;
use crate::engines::LlmEngine;

// Max tokens generated per turn - keeps voice responses concise.
const MAX_NEW_TOKENS: usize = 256;

pub struct LlamaCppEngine {
    // Backend must outlive the model and context; keep it here.
    _backend: LlamaBackend,
    model: LlamaModel,
    system_prompt: String,
    // The context is created once at startup and reused across turns.
    // A Mutex is required because LlamaContext is !Sync and generate()
    // is called from spawn_blocking threads (which are Send).
    ctx: Mutex<LlamaContext<'static>>,
}

// SAFETY: LlamaBackend and LlamaModel hold raw pointers; the crate guarantees
// thread-safety for both. LlamaContext is wrapped in a Mutex so exclusive access
// is enforced at runtime.
unsafe impl Send for LlamaCppEngine {}
unsafe impl Sync for LlamaCppEngine {}

impl LlamaCppEngine {
    /// `model_path` is the absolute path to the GGUF file, resolved from the
    /// model registry before this constructor is called.
    pub fn new(model_path: &Path, cfg: &LlmConfig) -> Result<Self> {
        anyhow::ensure!(
            model_path.exists(),
            "GGUF not found: {}",
            model_path.display()
        );

        let backend = LlamaBackend::init().context("llama backend init")?;
        let model = LlamaModel::load_from_file(&backend, model_path, &LlamaModelParams::default())
            .with_context(|| format!("loading {}", model_path.display()))?;

        let n_ctx = NonZeroU32::new(cfg.context_size.max(512)).unwrap();
        let cparams = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx))
            .with_n_batch(512);

        // SAFETY: We extend the lifetime to 'static because _backend and model
        // are stored in the same struct and will outlive the context.
        // This is safe as long as we never drop _backend or model while the
        // context is still in use, which the struct layout guarantees.
        let ctx = model
            .new_context(&backend, cparams)
            .context("creating llama context")?;
        let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };

        Ok(Self {
            _backend: backend,
            model,
            system_prompt: cfg.system_prompt.clone(),
            ctx: Mutex::new(ctx),
        })
    }
}

impl LlmEngine for LlamaCppEngine {
    fn generate(&self, user_text: &str, tx: Sender<String>) -> Result<()> {
        let prompt = format_prompt(&self.system_prompt, user_text);
        let vocab = self.model.vocab();

        // add_special=true (BOS), parse_special=true (handle <start_of_turn> etc.)
        let tokens = vocab.tokenize(prompt.as_bytes(), true, true);
        anyhow::ensure!(!tokens.is_empty(), "tokenize returned empty");

        // Acquire the context for the duration of this turn.
        let mut ctx = self
            .ctx
            .lock()
            .map_err(|_| anyhow::anyhow!("context mutex poisoned"))?;

        // Clear KV-cache from the previous turn so we start fresh.
        ctx.clear_kv_cache();

        // Prefill: request logits only on the last token.
        let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
        let last = tokens.len() - 1;
        for (i, &tok) in tokens.iter().enumerate() {
            batch
                .add(tok, i as i32, &[0], i == last)
                .context("batch add prefill")?;
        }
        ctx.decode(&mut batch).context("prefill decode")?;

        // Greedy-ish sampler chain.
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::temp(0.8),
            LlamaSampler::dist(42),
        ]);

        let mut n_past = tokens.len() as i32;
        let mut logits_idx = last as i32;

        for _ in 0..MAX_NEW_TOKENS {
            let mut cands = ctx.token_data_array_ith(logits_idx);
            sampler.apply(&mut cands);
            let tok = cands
                .selected_token()
                .unwrap_or_else(|| sampler.sample(&ctx, logits_idx));
            sampler.accept(tok);

            if vocab.is_eog(tok) {
                break;
            }

            // token_to_piece returns Vec<u8>; lossy-convert for partial UTF-8.
            let piece = vocab.token_to_piece(tok, true, None);
            let text = String::from_utf8_lossy(&piece).into_owned();

            // Best-effort send - if the receiver dropped, stop generating.
            if tx.blocking_send(text).is_err() {
                break;
            }

            batch.clear();
            batch
                .add(tok, n_past, &[0], true)
                .context("batch add decode")?;
            n_past += 1;
            ctx.decode(&mut batch).context("decode step")?;
            logits_idx = 0;
        }

        Ok(())
    }
}

/// Build a Gemma-3 chat-formatted prompt from system + user text.
/// Exported so the unit test can call it directly.
pub fn format_prompt(system: &str, user: &str) -> String {
    format!(
        "<start_of_turn>system\n{system}<end_of_turn>\n\
         <start_of_turn>user\n{user}<end_of_turn>\n\
         <start_of_turn>model\n"
    )
}
