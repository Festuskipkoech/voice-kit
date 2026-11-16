use voicekit_core::sentence::SentenceBuffer;

// sentencex only produces 2+ segments when a second sentence begins after a terminator.
// Push "Sentence one. Next" - sentencex splits into ["Sentence one. ", "Next"],
// so the first segment is flushed and "Next" stays in the buffer.

#[test]
fn boundary_on_period() {
    let mut buf = SentenceBuffer::new();
    let ready = buf.push("Hello world. Next");
    assert_eq!(ready, vec!["Hello world."]);
}

#[test]
fn boundary_on_question_mark() {
    let mut buf = SentenceBuffer::new();
    let ready = buf.push("How are you? Fine");
    assert_eq!(ready, vec!["How are you?"]);
}

#[test]
fn boundary_on_exclamation() {
    let mut buf = SentenceBuffer::new();
    let ready = buf.push("Watch out! Then");
    assert_eq!(ready, vec!["Watch out!"]);
}

#[test]
fn no_boundary_on_dr_period() {
    let mut buf = SentenceBuffer::new();
    // "Dr." must NOT trigger a flush - it is an abbreviation.
    let ready = buf.push("Dr. Smith said hello");
    assert!(ready.is_empty(), "Dr. must not trigger a sentence boundary");
}

#[test]
fn hard_cap_at_200_chars() {
    let mut buf = SentenceBuffer::new();
    // Push a 200-char string with no sentence boundary.
    let long = "a".repeat(200);
    let ready = buf.push(&long);
    assert!(!ready.is_empty(), "hard cap must flush at 200 chars");
    assert!(!ready[0].is_empty());
    // Buffer must be drained.
    assert!(buf.flush().is_none());
}

#[test]
fn flush_returns_remainder() {
    let mut buf = SentenceBuffer::new();
    // Push text with no sentence terminator - nothing flushed yet.
    let ready = buf.push("Still going");
    assert!(ready.is_empty());
    // flush() drains remainder.
    let tail = buf.flush();
    assert_eq!(tail, Some("Still going".to_string()));
    // Second flush is empty.
    assert!(buf.flush().is_none());
}

#[test]
fn multiple_sentences_in_one_push() {
    let mut buf = SentenceBuffer::new();
    // Two complete sentences plus a trailing fragment forces sentencex to yield 3 segments.
    let ready = buf.push("First sentence. Second sentence. Third");
    assert_eq!(ready.len(), 2);
    assert!(ready[0].contains("First"));
    assert!(ready[1].contains("Second"));
}
