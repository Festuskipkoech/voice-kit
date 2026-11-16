use voicekit_core::engines::llm::llamacpp::format_prompt;

#[test]
fn format_prompt_includes_system() {
    let out = format_prompt("Be helpful.", "hello");
    assert!(out.contains("Be helpful."), "system prompt missing");
}

#[test]
fn format_prompt_includes_user_text() {
    let out = format_prompt("Be helpful.", "What is two plus two?");
    assert!(out.contains("What is two plus two?"), "user text missing");
}

#[test]
fn format_prompt_has_turn_markers() {
    let out = format_prompt("sys", "usr");
    assert!(
        out.contains("<start_of_turn>system\n"),
        "system turn tag missing"
    );
    assert!(
        out.contains("<start_of_turn>user\n"),
        "user turn tag missing"
    );
    assert!(
        out.contains("<start_of_turn>model\n"),
        "model turn tag missing"
    );
    assert!(out.contains("<end_of_turn>"), "end_of_turn missing");
}

#[test]
fn format_prompt_model_tag_is_last() {
    let out = format_prompt("sys", "usr");
    let model_pos = out.rfind("<start_of_turn>model").unwrap();
    let user_pos = out.find("<start_of_turn>user").unwrap();
    assert!(model_pos > user_pos, "model turn must follow user turn");
    // Prompt should end with the open model tag, ready for generation.
    assert!(
        out.ends_with("<start_of_turn>model\n"),
        "prompt must end with open model tag"
    );
}
