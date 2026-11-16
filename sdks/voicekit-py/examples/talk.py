"""
Minimal end-to-end example: captures microphone audio, sends it to a local
Voicekit server, and plays back the TTS response in real time.

Usage:
    python examples/talk.py
    python examples/talk.py --url ws://myserver:8080 --voice bf_isabella
"""

import argparse
import asyncio

from voicekit import VoicekitClient, SessionConfig
from voicekit.audio import AudioCapture, AudioPlayer


async def run(url: str, voice: str) -> None:
    async with VoicekitClient(url) as client:
        await client.start_session(SessionConfig(voice=voice))
        print(f"connected to {url} — speak now (Ctrl-C to stop)")

        async with AudioCapture() as cap, AudioPlayer() as player:

            async def send_loop() -> None:
                async for frame in cap:
                    await client.send_audio(frame)

            async def recv_loop() -> None:
                async for frame in client.receive_audio():
                    await player.play(frame)

            try:
                await asyncio.gather(send_loop(), recv_loop())
            except asyncio.CancelledError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Voicekit demo")
    parser.add_argument("--url", default="ws://localhost:8080", help="server URL")
    parser.add_argument("--voice", default="af_heart", help="TTS voice ID")
    args = parser.parse_args()

    try:
        asyncio.run(run(args.url, args.voice))
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
