//! LLM output normalisation for TTS - strips markdown and emoji.

use markdown_strip::strip_markdown;

/// Strip markdown formatting and emoji from a raw LLM sentence.
pub fn clean(text: &str) -> String {
    // emoji first - markdown_strip corrupts multi-byte emoji sequences
    let no_emoji: String = text.chars().filter(|&c| !is_emoji(c)).collect();
    strip_markdown(&no_emoji)
}

/// Returns true for characters in the standard emoji Unicode ranges.
fn is_emoji(c: char) -> bool {
    matches!(c as u32,
        0x1F300..=0x1F9FF |  // misc symbols, pictographs, emoticons, transport
        0x1FA00..=0x1FAFF |  // chess, symbols extended-A
        0x2600..=0x27BF  |  // misc symbols, dingbats
        0xFE00..=0xFE0F  |  // variation selectors
        0x1F1E0..=0x1F1FF   // regional indicator letters (flags)
    )
}
