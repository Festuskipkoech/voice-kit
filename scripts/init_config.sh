#!/usr/bin/env sh
# Writes /data/config.toml from the template if missing, validates if present.
set -eu

CONFIG_PATH="/data/config.toml"
TEMPLATE_PATH="/etc/voicekit/config.template.toml"

if [ -f "$CONFIG_PATH" ]; then
    echo "[config-init] config.toml found - validating..."
    if command -v python3 > /dev/null 2>&1; then
        python3 -c "
import sys
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print('[config-init] WARNING: no TOML parser; skipping validation', file=sys.stderr)
        sys.exit(0)
with open('$CONFIG_PATH', 'rb') as f:
    tomllib.load(f)
print('[config-init] config.toml is valid')
"
    fi
    exit 0
fi

echo ""
echo "VOICEKIT - first run. No config.toml found."
echo "Writing defaults to: $CONFIG_PATH"
echo ""
echo "Review these settings before starting the server:"
echo "  [llm] system_prompt - assistant personality"
echo "  [tts] voice - TTS voice (default: af_heart)"
echo "  [server] host / port / max_connections"
echo "  [session_policy] allow_*_override - what clients may change"
echo ""
echo "Edit ./voicekit-data/config.toml then: docker compose up voicekit-server"
echo ""

mkdir -p "$(dirname "$CONFIG_PATH")"
cp "$TEMPLATE_PATH" "$CONFIG_PATH"

echo "[config-init] done."
exit 0