"""
Voicekit Python SDK.

Public API::

    from voicekit import VoicekitClient, SessionConfig
    from voicekit.audio import AudioCapture, AudioPlayer
    from voicekit.exceptions import (
        VoicekitError,
        HandshakeError,
        PolicyError,
        ProtocolError,
        NotConnectedError,
    )
"""

from .client import VoicekitClient
from .protocol import SessionConfigFrame as SessionConfig
from .exceptions import (
    VoicekitError,
    HandshakeError,
    PolicyError,
    ProtocolError,
    NotConnectedError,
)

__all__ = [
    "VoicekitClient",
    "SessionConfig",
    "VoicekitError",
    "HandshakeError",
    "PolicyError",
    "ProtocolError",
    "NotConnectedError",
]

__version__ = "0.1.0"
