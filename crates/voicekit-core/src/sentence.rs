use sentencex::segment;

const HARD_CAP: usize = 200;

/// Accumulates LLM tokens and flushes complete sentences to TTS.
pub struct SentenceBuffer {
    buf: String,
}

impl SentenceBuffer {
    pub fn new() -> Self {
        Self { buf: String::new() }
    }

    /// Append a token. Returns any sentences that are ready for TTS.
    pub fn push(&mut self, token: &str) -> Vec<String> {
        self.buf.push_str(token);
        self.extract()
    }

    /// Drain remaining text at end of generation.
    pub fn flush(&mut self) -> Option<String> {
        let text = self.buf.trim().to_string();
        self.buf.clear();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }

    fn extract(&mut self) -> Vec<String> {
        // Hard cap: flush the whole buffer immediately.
        if self.buf.len() >= HARD_CAP {
            let out = self.buf.trim().to_string();
            self.buf.clear();
            return if out.is_empty() { vec![] } else { vec![out] };
        }

        let segs = segment("en", &self.buf);

        // Need at least two segments: one complete + one in-progress tail.
        if segs.len() < 2 {
            return vec![];
        }

        let mut ready: Vec<String> = Vec::new();
        // Everything except the last segment is a complete sentence.
        let tail = segs.last().unwrap();
        let tail_start = self.buf.len() - tail.len();
        for seg in &segs[..segs.len() - 1] {
            let s = seg.trim().to_string();
            if !s.is_empty() {
                ready.push(s);
            }
        }
        self.buf = self.buf[tail_start..].to_string();
        ready
    }
}

impl Default for SentenceBuffer {
    fn default() -> Self {
        Self::new()
    }
}
