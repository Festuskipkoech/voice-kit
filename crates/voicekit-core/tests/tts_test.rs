use std::collections::HashMap;
use voicekit_core::engines::tts::paradee::text_to_ids;

// Minimal inline vocab - only the symbols needed by these tests.
// IDs match the real Kokoro config.json entries so the tests remain meaningful.
fn test_vocab() -> HashMap<char, i64> {
    [('ɹ', 41_i64), ('h', 10), ('ɛ', 20), ('l', 30), ('o', 40)]
        .into_iter()
        .collect()
}

#[test]
fn phoneme_ids_pad_both_ends() {
    let vocab = test_vocab();
    let ids = text_to_ids("ɹ", &vocab);
    assert_eq!(ids.first(), Some(&0), "must start with pad token 0");
    assert_eq!(ids.last(), Some(&0), "must end with pad token 0");
    assert_eq!(ids.len(), 3); // [0, 41, 0]
}

#[test]
fn phoneme_ids_known_symbol() {
    let vocab = test_vocab();
    let ids = text_to_ids("ɹ", &vocab);
    assert_eq!(ids[1], 41, "ɹ must map to ID 41");
}

#[test]
fn phoneme_ids_skips_unknown() {
    let vocab = test_vocab();
    // '@' is not in the vocab - must be silently dropped.
    let ids = text_to_ids("@", &vocab);
    assert_eq!(ids, vec![0, 0]);
}

#[test]
fn phoneme_ids_truncates_at_512() {
    let vocab = test_vocab();
    // 600 reps of a known symbol → must be capped at 512.
    let long = "ɹ".repeat(600);
    let ids = text_to_ids(&long, &vocab);
    assert_eq!(ids.len(), 512, "output must be capped at 512 tokens");
    assert_eq!(
        ids.last(),
        Some(&0),
        "last token must be pad after truncation"
    );
}
