use voicekit_core::engines::stt::moonshine::Tokenizer;

// helpers
fn make_tokenizer(pairs: &[(&str, i64)]) -> Tokenizer {
    let vocab: serde_json::Map<String, serde_json::Value> = pairs
        .iter()
        .map(|(tok, id)| (tok.to_string(), serde_json::json!(id)))
        .collect();

    let json = serde_json::json!({
        "model": { "type": "BPE", "vocab": vocab }
    });

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, json.to_string()).unwrap();
    Tokenizer::from_file(&path).expect("tokenizer from_file")
}

// tests
#[test]
fn decode_known_tokens() {
    let tok = make_tokenizer(&[("\u{0120}hello", 10), ("\u{0120}world", 11)]);
    assert_eq!(tok.decode(&[10, 11]), "hello world");
}

#[test]
fn decode_skips_unknown_ids() {
    let tok = make_tokenizer(&[("\u{0120}hi", 5)]);
    assert_eq!(tok.decode(&[999, 5, 999]), "hi");
}

#[test]
fn decode_empty_gives_empty_string() {
    let tok = make_tokenizer(&[("\u{0120}word", 1)]);
    assert_eq!(tok.decode(&[]), "");
}

#[test]
fn decode_trims_whitespace() {
    let tok = make_tokenizer(&[("\u{0120}word", 1)]);
    assert_eq!(tok.decode(&[1]), "word");
}

#[test]
fn decode_newline_token() {
    let tok = make_tokenizer(&[("\u{010A}", 3), ("\u{0120}line", 4)]);
    let result = tok.decode(&[4, 3, 4]);
    assert!(result.contains('\n'), "expected newline in {result:?}");
}
