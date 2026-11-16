"""
Minimal end-to-end example: captures microphone audio, sends it to a local
Voicekit server, and plays back the TTS response in real time.

Usage:
    python examples/talk.py
    python examples/talk.py --url ws://myserver:8080 --voice bf_isabella
"""

import argparse
import asyncio

import numpy as np

from voicekit import VoicekitClient, SessionConfig
from voicekit.audio import AudioCapture, AudioPlayer

# RMS energy threshold to distinguish speech from background noise.
# Raise this if your mic picks up too much ambient sound as "speech".
_SPEECH_THRESHOLD = 0.01

# How many consecutive silent frames before we consider speech finished.
# At 512 samples / 16 kHz = 32 ms per frame; 25 frames ~ 800 ms.
_SILENCE_FRAMES = 25


def _status(msg: str) -> None:
    """Overwrite the current terminal line with a status message."""
    print(f"\r  {msg:<55}", end="", flush=True)


async def run(url: str, voice: str | None) -> None:
    async with VoicekitClient(url) as client:
        await client.start_session(SessionConfig(voice=voice))
        print(f"connected to {url}\n")
        _status("listening  (speak now, Ctrl-C to stop)")

        async with AudioCapture() as cap, AudioPlayer() as player:

            async def send_loop() -> None:
                """
                Forward mic frames to the server and update the status line
                based on outgoing audio energy so the user can see when the
                VAD is likely to have picked up their voice.
                """
                was_speaking = False
                silence_count = 0

                async for frame in cap:
                    await client.send_audio(frame)

                    energy = float(np.sqrt(np.mean(frame**2)))

                    if energy > _SPEECH_THRESHOLD:
                        was_speaking = True
                        silence_count = 0
                        _status("listening  [speech detected]")
                    else:
                        if was_speaking:
                            silence_count += 1
                            if silence_count >= _SILENCE_FRAMES:
                                was_speaking = False
                                silence_count = 0
                                _status("processing ...")

            async def recv_loop() -> None:
                """
                Play each incoming audio frame and keep the status line updated
                so the user knows when the server is responding.
                """
                first_frame = True

                async for frame in client.receive_audio():
                    if first_frame:
                        first_frame = False
                        # Move to a new line so the response text has its own row.
                        print()
                        _status("speaking   ...")

                    await player.play(frame)

                # Back to idle after the server finishes this response.
                print()
                _status("listening  (speak now, Ctrl-C to stop)")

            try:
                await asyncio.gather(send_loop(), recv_loop())
            except asyncio.CancelledError:
                pass

        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Voicekit demo")
    parser.add_argument("--url", default="ws://localhost:8080", help="server URL")
    parser.add_argument(
        "--voice",
        default=None,
        help="TTS voice ID (uses server default if omitted)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run(args.url, args.voice))
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
