"""The conformance agent of sent-dm-voice-protocol's conformance/README.md, on this SDK's API. The protocol's
harness runs it in CI before every release: `python -m sentdm_voice_agents.conformance` connects to
SENT_CONFORMANCE_URL with any API key and does exactly what the README's table says."""

from __future__ import annotations

import logging
import os

from .. import frames
from ..agent import Agent
from ..session import Session, SessionEnded

GREETING = "Hello from the conformance agent."


def decide(text: str) -> list[tuple[str, dict]]:
    """What a user.said asks for: a list of (action, details) the agent performs in order. A command is the
    text up to the first space; `stream` takes the next word as the first piece."""
    command, _, rest = text.partition(" ")
    if command == "echo":
        return [("say", {"text": rest})]
    if command == "stream":
        word, _, remainder = rest.partition(" ")
        return [("say", {"text": word, "last": False}), ("say", {"text": remainder})]
    if command == "transfer":
        return [("tool", {"name": "transfer", "arguments": {"to": {"kind": "user", "value": "conformance-desk"}}})]
    if command == "hangup":
        return [("tool", {"name": "hangup"})]
    if command == "dtmf":
        return [("tool", {"name": "send_dtmf", "arguments": {"digits": rest}})]
    if command == "hold":
        return [("tool", {"name": "hold", "arguments": {"held": True}})]
    return []


class ConformanceAgent(Agent):
    async def on_session_start(self, session: Session) -> None:
        if session.session_type == "text":
            await session.say(GREETING)

    async def on_user_said(self, session: Session, said: frames.UserSaid) -> None:
        if said.correction:
            await session.clear()
        for action, details in decide(said.text):
            try:
                if action == "say":
                    await session.say(**details)
                else:
                    # The harness answers every tool call; the result also reaches on_tool_result.
                    await session.tool(details["name"], details.get("arguments"))
            except SessionEnded:
                return

    async def on_tool_result(self, session: Session, result: frames.ToolResult) -> None:
        if not result.ok and result.error is not None:
            try:
                await session.say(f"Tool failed: {result.error.code}")
            except SessionEnded:
                pass

    async def on_audio(self, session: Session, samples: bytes) -> None:
        await session.send_audio(samples)


def main() -> None:
    logging.basicConfig(level=os.environ.get("SENT_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    url = os.environ.get("SENT_CONFORMANCE_URL") or "ws://localhost:8765/v1/agents/connect"
    ConformanceAgent(api_key="conformance", agent_id="agent_conformance", url=url).serve()


if __name__ == "__main__":
    main()
