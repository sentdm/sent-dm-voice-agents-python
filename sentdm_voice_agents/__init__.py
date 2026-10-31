"""Sent Voice Agents for Python: an AI agent that takes and places phone calls through Sent.

    from sentdm_voice_agents import Agent, Session

    class Support(Agent):
        async def on_session_start(self, session: Session) -> None:
            await session.say("Hi, this is Acme support. How can I help?")

        async def on_user_said(self, session: Session, said) -> None:
            if "person" in said.text:
                await session.say("Transferring you now.")
                await session.transfer("+14155550100")
            else:
                await session.say(f"You said {said.text}")

    Support(api_key=..., agent_id=...).serve()

The agent connects to Sent with the account API key and serves calls until stopped. `frames` holds the
protocol's frames as types, generated from sent-dm-voice-protocol.
"""

from . import frames
from .agent import DEFAULT_URL, Agent, Refused
from .audio import audio_frame, bytes_per_frame, samples_of, stream_of
from .frames import PROTOCOL_VERSION, FrameError, Party, SessionEnd, ToolResult, UserSaid
from .session import Session, SessionEnded
from .version import __version__

__all__ = [
    "Agent",
    "DEFAULT_URL",
    "FrameError",
    "PROTOCOL_VERSION",
    "Party",
    "Refused",
    "Session",
    "SessionEnd",
    "SessionEnded",
    "ToolResult",
    "UserSaid",
    "__version__",
    "audio_frame",
    "bytes_per_frame",
    "frames",
    "samples_of",
    "stream_of",
]
