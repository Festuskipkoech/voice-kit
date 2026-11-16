//! Tests for text_clean::clean - markdown stripping and emoji removal.

use voicekit_core::text_clean::clean;

#[test]
fn strips_bold_and_italic() {
    assert_eq!(clean("**bold** and *italic*"), "bold and italic");
}

#[test]
fn strips_atx_header() {
    assert_eq!(clean("## Hello World"), "Hello World");
}

#[test]
fn strips_bullet_list_prefix() {
    assert_eq!(clean("* Machine Learning: a key idea"), "Machine Learning: a key idea");
}

#[test]
fn keeps_link_text_drops_url() {
    assert_eq!(clean("[Rust](https://rust-lang.org)"), "Rust");
}

#[test]
fn strips_inline_code() {
    assert_eq!(clean("call `foo()` to start"), "call foo() to start");
}

#[test]
fn removes_emoji() {
    assert_eq!(clean("Great work! 🎉"), "Great work! ");
}

#[test]
fn removes_emoji_and_markdown_together() {
    assert_eq!(
        clean("**Machine Learning:** A key part of AI. 🙂"),
        "Machine Learning: A key part of AI. "
    );
}

#[test]
fn plain_text_passes_through_unchanged() {
    assert_eq!(clean("Hello, how can I help you today?"), "Hello, how can I help you today?");
}

#[test]
fn empty_input_returns_empty() {
    assert_eq!(clean(""), "");
}

#[test]
fn purely_formatting_input_has_no_alphabetic_content() {
    // markdown_strip may leave a bare "*" - nothing alphabetic remains
    let result = clean("* ");
    assert!(!result.chars().any(|c| c.is_alphabetic()), "expected no letters, got: {result:?}");
}