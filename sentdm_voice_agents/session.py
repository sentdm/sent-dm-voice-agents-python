"""One session: a call while this agent is on it.

A Session is what the handlers of an Agent receive. It carries what session.start said about the call, sends the
agent's frames on whichever connection the session is on, keeps the latest counted frames so a resume can send
again the ones Sent did not read, and turns a tool call into the tool.result that answers it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from . import frames
from .audio import audio_frame

if TYPE_CHECKING:
    from .agent import _Connection

log = logging.getLogger("sentdm_voice_agents")


class SessionEnded(Exception):
    """The session ended before the tool call was answered."""


class Session:
    """The call the agent is on, and the ways to act on it. Created by the agent; one task runs its handlers."""

    def __init__(self, start: frames.SessionStart, connection: _Connection) -> None:
        self.start = start
        self.id = start.session_id
        self.call_id = start.call_id
        self.agent_id = start.agent_id
        self.direction = start.direction
        self.from_ = start.from_
        self.to = start.to
        self.session_type = start.session_type
        self.language = start.language
        self.context: dict[str, Any] = dict(start.context or {})
        self.started_at = start.started_at
        # Hz of the session's 16-bit mono PCM, both ways, in an audio session.
        self.sample_rate = start.audio.sample_rate if start.audio else 16000
        # The number in the header of the session's binary audio frames on its current connection.
        self.stream: int | None = start.audio.stream if start.audio else None
        # The latest resume token, or None when the session cannot be resumed. Keep it out of logs.
        self.token: str | None = start.resume.token if start.resume else None
        self.window_ms: int | None = start.resume.window_ms if start.resume else None
        # How many of Sent's counted frames for the session the agent has read: the `received` of a resume.
        self.received = 0
        # session.end, once it arrived.
        self.end: frames.SessionEnd | None = None
        self._connection = connection
        # A session.resume was sent on the current connection and Sent has not answered it yet.
        self._resuming = False
        # The latest counted frames the agent sent, and how many came before them.
        self._kept: list[frames.Frame] = []
        self._before = 0
        self._calls = 0
        self._pending: dict[str, asyncio.Future[frames.ToolResult]] = {}
        self._inbox: asyncio.Queue[frames.Frame | bytes | None] = asyncio.Queue()

    # ----- what the agent does on the call -----

    @property
    def ended(self) -> bool:
        return self.end is not None

    @property
    def resumable(self) -> bool:
        """Whether the session survives a lost connection: it has a resume token."""
        return self.token is not None

    @property
    def sent(self) -> int:
        """How many counted frames the agent has sent for the session."""
        return self._before + len(self._kept)

    async def say(self, text: str, *, last: bool = True, interruptible: bool = True) -> None:
        """Text sessions: speak `text` to the caller. Send a reply in pieces with last=False and finish with
        last=True; the pieces are spoken as one utterance."""
        await self._send(frames.AgentSay(session_id=self.id, text=text, interruptible=interruptible, last=last))

    async def clear(self) -> None:
        """Drop the speech or audio the agent sent that Sent has not played yet."""
        await self._send(frames.AgentClear(session_id=self.id))

    async def send_audio(self, samples: bytes) -> None:
        """Audio sessions: play `samples`, 16-bit little-endian mono PCM at the session's sample rate. Send
        20 ms per call, and no more than 200 ms."""
        if self.stream is None:
            raise RuntimeError(f"session {self.id} is a text session")
        await self._connection.send_bytes(audio_frame(self.stream, samples))

    async def transfer(self, to: frames.Party | str) -> frames.ToolResult:
        """Move the call to `to`: a Party, or an E.164 number. Returns Sent's answer; on success the session then
        ends with reason transferred."""
        party = frames.Party(kind="number", value=to) if isinstance(to, str) else to
        return await self.tool("transfer", {"to": {"kind": party.kind, **({"value": party.value} if party.value else {})}})

    async def hangup(self) -> frames.ToolResult:
        """End the call, after the speech the agent queued before it."""
        return await self.tool("hangup")

    async def send_dtmf(self, digits: str) -> frames.ToolResult:
        """Play `digits` on the call: 0 to 9, *, #, A to D, and w to wait half a second."""
        return await self.tool("send_dtmf", {"digits": digits})

    async def hold(self, held: bool = True) -> frames.ToolResult:
        """Put the caller on hold, or take them off it."""
        return await self.tool("hold", {"held": held})

    async def tool(self, name: str, arguments: dict[str, Any] | None = None) -> frames.ToolResult:
        """Call a tool by name and wait for its tool.result. Raises SessionEnded if the session ends first."""
        if self.ended:
            raise SessionEnded(self.id)
        self._calls += 1
        call_ref = f"t{self._calls}"
        future: asyncio.Future[frames.ToolResult] = asyncio.get_running_loop().create_future()
        self._pending[call_ref] = future
        await self._send(frames.ToolCall(session_id=self.id, call_ref=call_ref, name=name, arguments=arguments))
        return await future

    # ----- what the agent's connection does with the session -----

    async def _send(self, frame: frames.Frame) -> None:
        """Sends a frame for the session on its connection, keeping it if it counts. A frame that could not be
        sent still counts: Sent's `received` on the next resume says whether to send it again."""
        if self.ended:
            raise SessionEnded(self.id)
        self._keep(frame)
        await self._connection.send(frame)

    def _keep(self, frame: frames.Frame) -> None:
        if frame.TYPE in frames.UNCOUNTED:
            return
        self._kept.append(frame)
        if len(self._kept) > frames.RETAINED_FRAMES:
            self._kept.pop(0)
            self._before += 1

    def _read(self, frame: frames.Frame) -> None:
        """Counts one of Sent's frames for the session, as Counting frames in docs/protocol.md says: every frame
        with the session's id except the ones marked uncounted, also one of a type the SDK does not know."""
        if frame.TYPE not in frames.UNCOUNTED:
            self.received += 1

    def _resume_frame(self) -> frames.SessionResume:
        if not self.token:
            raise RuntimeError(f"session {self.id} has no resume token")
        return frames.SessionResume(session_id=self.id, token=self.token, received=self.received)

    def _resumed(self, frame: frames.SessionResumed) -> list[frames.Frame]:
        """Takes the answer to a session.resume. Returns the frames to send again, or a hangup when Sent needs one
        the session no longer keeps."""
        self._resuming = False
        self.token = frame.resume.token if frame.resume else None
        if frame.audio is not None:
            self.stream = frame.audio.stream
        if frame.resume is None:
            # The session ended while it was detached; its replay ends with session.end.
            return []
        received = frame.received
        unread = None if received < self._before else self._kept[received - self._before :]
        self._before = received
        self._kept = []
        if unread is None:
            self._calls += 1
            again: list[frames.Frame] = [frames.ToolCall(session_id=self.id, call_ref=f"t{self._calls}", name="hangup")]
        else:
            again = list(unread)
        for frame_again in again:
            self._keep(frame_again)
        return again

    def _answer_tool(self, result: frames.ToolResult) -> None:
        future = self._pending.pop(result.call_ref, None)
        if future is not None and not future.done():
            future.set_result(result)

    def _finish(self, end: frames.SessionEnd) -> None:
        self.end = end
        for future in self._pending.values():
            if not future.done():
                future.set_exception(SessionEnded(self.id))
        self._pending.clear()

    def __repr__(self) -> str:
        return f"Session({self.id}, {self.session_type}, {self.direction})"
