"""The agent: one process that serves calls on the Sent Voice Agents protocol.

Subclass Agent and override the handlers you need; run() connects to Sent with the account API key and serves
sessions until stop(). The runtime answers heartbeats, opens a new connection when one drains or is lost, and
resumes the sessions that can move, so the handlers only see calls. One task runs each session's handlers in
the order Sent's frames arrived; different sessions run at the same time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from . import frames
from .audio import samples_of, stream_of
from .session import Session

log = logging.getLogger("sentdm_voice_agents")

DEFAULT_URL = f"wss://{frames.HOST}{frames.CONNECT_PATH}"
# The longest wait before another attempt to connect.
MAX_BACKOFF_S = 30.0
# How long a handshake may take before the attempt counts as failed.
HANDSHAKE_TIMEOUT_S = 10.0
# How many heartbeat intervals without a frame make a connection dead, as Heartbeats in docs/protocol.md says.
SILENT_INTERVALS = 3
# Why Sent refuses a handshake with each status, from Connecting in docs/protocol.md. Each needs a person.
REFUSALS = {
    400: "agent_id or the subprotocol is missing",
    401: "the API key is not valid",
    403: "voice is not enabled for the account, or the key does not own the agent",
    404: "the agent does not exist",
}


class Refused(Exception):
    """Sent refused the connection for a reason that needs a person: the status and why."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(f"{status}: {reason}")
        self.status = status
        self.reason = reason


@dataclass(eq=False)
class _Connection:
    """One WebSocket to Sent, and the sessions it reads and writes."""

    pending: list[Session]  # sessions to resume here once Sent is ready
    socket: ClientConnection | None = None
    ready: bool = False
    draining: bool = False
    sessions: dict[str, Session] = field(default_factory=dict)
    heartbeat_s: float = 0.0
    task: asyncio.Task[None] | None = None

    async def send(self, frame: frames.Frame) -> None:
        data = frames.encode(frame)
        try:
            if self.socket is None:
                raise ConnectionClosed(None, None)
            await self.socket.send(json.dumps(data))
        except ConnectionClosed:
            log.info("not sent: %s, the connection is not open", data["type"])

    async def send_bytes(self, frame: bytes) -> None:
        try:
            if self.socket is None:
                raise ConnectionClosed(None, None)
            await self.socket.send(frame)
        except ConnectionClosed:
            pass


class Agent:
    """Serves calls. Override the handlers; they run on the session's own task, in order."""

    def __init__(self, *, api_key: str, agent_id: str, url: str = DEFAULT_URL) -> None:
        self.api_key = api_key
        self.agent_id = agent_id
        self.url = url
        self._connections: set[_Connection] = set()
        self._sessions: dict[str, Session] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._attempt = 0
        self._stopped = False
        self._done: asyncio.Future[None] | None = None

    # ----- handlers to override -----

    async def on_connected(self, ready: frames.ConnectionReady) -> None:
        """A connection to Sent is ready for sessions."""

    async def on_session_start(self, session: Session) -> None:
        """A call reached the agent. In a text session, say() to speak; an audio session waits for audio."""

    async def on_user_said(self, session: Session, said: frames.UserSaid) -> None:
        """Text sessions: the caller said something. With correction=True it replaces the previous one."""

    async def on_audio(self, session: Session, samples: bytes) -> None:
        """Audio sessions: the caller's audio, 16-bit little-endian mono PCM at session.sample_rate."""

    async def on_tool_result(self, session: Session, result: frames.ToolResult) -> None:
        """Sent answered a tool call. The call's awaiting coroutine gets the same result."""

    async def on_frame(self, session: Session, frame: frames.UnknownFrame) -> None:
        """A session frame this SDK version does not know. It was counted; nothing else is needed."""

    async def on_session_end(self, session: Session, end: frames.SessionEnd) -> None:
        """The call left the agent. session.end says why."""

    async def on_session_lost(self, session: Session) -> None:
        """The session's connection was lost and it could not be resumed: Sent ends the call with
        agent_unreachable. No session.end follows."""

    async def on_error(self, error: frames.ErrorFrame) -> None:
        """Sent refused a frame; the connection stays open."""

    # ----- running -----

    def serve(self) -> None:
        """Runs the agent until SIGINT or SIGTERM. For a program whose main is the agent."""

        async def main() -> None:
            loop = asyncio.get_running_loop()
            for signum in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(signum, self.stop)
            await self.run()

        asyncio.run(main())

    async def run(self) -> None:
        """Serves calls until stop(). Raises Refused when Sent refuses the connection for a reason that needs a
        person."""
        self._done = asyncio.get_running_loop().create_future()
        self._open([], 0.0)
        try:
            await self._done
        finally:
            await self._close_all()

    def stop(self) -> None:
        """Stops serving: every connection closes with 1000, so the calls on them follow the agent's fallback
        at once."""
        self._finish()

    @property
    def sessions(self) -> list[Session]:
        """Every live session."""
        return list(self._sessions.values())

    def _finish(self, error: Exception | None = None) -> None:
        if self._done is None or self._done.done():
            return
        if error is None:
            self._done.set_result(None)
        else:
            self._done.set_exception(error)

    async def _close_all(self) -> None:
        self._stopped = True
        tasks = []
        for connection in list(self._connections):
            if connection.socket is not None:
                await connection.socket.close(1000)
            elif connection.task is not None:
                connection.task.cancel()
            if connection.task is not None:
                tasks.append(connection.task)
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*tasks, *self._tasks.values(), return_exceptions=True)

    # ----- connections -----

    def _open(self, pending: list[Session], delay: float) -> None:
        connection = _Connection(pending)
        self._connections.add(connection)
        connection.task = asyncio.create_task(self._serve(connection, delay))

    async def _serve(self, connection: _Connection, delay: float) -> None:
        await asyncio.sleep(delay)
        parts = urlsplit(self.url)
        query = urlencode([*parse_qsl(parts.query), ("agent_id", self.agent_id)])
        url = urlunsplit(parts._replace(query=query))
        try:
            async with connect(
                url,
                subprotocols=[frames.SUBPROTOCOL],  # type: ignore[list-item]
                additional_headers={"Authorization": f"Bearer {self.api_key}"},
                open_timeout=HANDSHAKE_TIMEOUT_S,
                close_timeout=1.0,
            ) as socket:
                connection.socket = socket
                log.info("connected to %s", parts.netloc)
                await self._read(connection, socket)
        except InvalidStatus as refusal:
            status = refusal.response.status_code
            log.error("refused with %s: %s", status, REFUSALS.get(status, "trying again"))
            if status in REFUSALS:
                self._finish(Refused(status, REFUSALS[status]))
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - a failed attempt is logged and tried again
            log.warning("connection failed: %s: %s", type(error).__name__, error)
        finally:
            await self._closed(connection)

    async def _read(self, connection: _Connection, socket: ClientConnection) -> None:
        while True:
            timeout = SILENT_INTERVALS * connection.heartbeat_s if connection.ready else None
            try:
                message = await asyncio.wait_for(socket.recv(), timeout)
            except TimeoutError:
                log.warning("no frame for %.0f s, closing", timeout or 0)
                return
            except ConnectionClosed as closed:
                close = closed.rcvd or closed.sent
                log.info("closed with %s %s", close.code if close else 1006, close.reason if close else "")
                return
            if isinstance(message, bytes):
                self._audio(connection, message)
                continue
            try:
                frame = frames.decode(json.loads(message))
            except (ValueError, frames.FrameError) as error:
                log.warning("ignored a frame: %s", error)
                continue
            await self._handle(connection, frame)

    async def _handle(self, connection: _Connection, frame: frames.Frame) -> None:
        if isinstance(frame, frames.ConnectionReady):
            await self._ready(connection, frame)
        elif isinstance(frame, frames.Ping):
            await connection.send(frames.Pong(ts=frame.ts))
        elif isinstance(frame, frames.ConnectionDraining):
            self._drain(connection)
        elif isinstance(frame, frames.SessionStart):
            self._start(connection, frame)
        elif isinstance(frame, frames.SessionResumed):
            await self._resumed(connection, frame)
        elif isinstance(frame, frames.ErrorFrame):
            self._error(connection, frame)
            await self._safely(self.on_error(frame))
        elif isinstance(frame, frames.UnknownFrame) and frame.session_id is None:
            log.info("ignored a %s frame", frame.type)
        else:
            self._session_frame(connection, frame)

    async def _ready(self, connection: _Connection, frame: frames.ConnectionReady) -> None:
        connection.ready = True
        connection.heartbeat_s = frame.heartbeat_interval_ms / 1000
        self._attempt = 0
        pending = [s for s in connection.pending if self._sessions.get(s.id) is s]
        connection.pending = []
        log.info("ready: protocol %s, up to %s sessions, resuming %s", frame.protocol_version, frame.max_sessions, len(pending))
        for session in pending:
            await self._resume(connection, session)
        await self._safely(self.on_connected(frame))

    async def _resume(self, connection: _Connection, session: Session) -> None:
        for other in self._connections:
            other.sessions.pop(session.id, None)
        connection.sessions[session.id] = session
        session._connection = connection
        session._resuming = True
        await connection.send(session._resume_frame())

    def _drain(self, connection: _Connection) -> None:
        connection.draining = True
        moving = [s for s in connection.sessions.values() if s.token and not s._resuming]
        log.info("draining, moving %s session(s)", len(moving))
        self._open(moving, 0.0)

    def _start(self, connection: _Connection, frame: frames.SessionStart) -> None:
        session = Session(frame, connection)
        self._sessions[session.id] = session
        connection.sessions[session.id] = session
        self._tasks[session.id] = asyncio.create_task(self._run_session(session))
        log.info("session %s started: %s, %s", session.id, frame.session_type, frame.direction)

    def _session_frame(self, connection: _Connection, frame: frames.Frame) -> None:
        session_id = getattr(frame, "session_id", None)
        session = connection.sessions.get(session_id) if session_id else None
        # A session that moved to another connection is read there: Sent replays what this one did not deliver.
        if session is None or session._resuming:
            return
        session._read(frame)
        if isinstance(frame, frames.ToolResult):
            session._answer_tool(frame)
        session._inbox.put_nowait(frame)
        if isinstance(frame, frames.SessionEnd):
            session._finish(frame)
            self._forget(session)

    def _audio(self, connection: _Connection, message: bytes) -> None:
        if len(message) < 2:
            return
        stream = stream_of(message)
        for session in connection.sessions.values():
            if session.stream == stream and not session._resuming:
                session._inbox.put_nowait(message)
                return

    async def _resumed(self, connection: _Connection, frame: frames.SessionResumed) -> None:
        session = connection.sessions.get(frame.session_id)
        if session is None or not session._resuming:
            return
        again = session._resumed(frame)
        log.info("session %s resumed, sending %s frame(s) again", session.id, len(again))
        for frame_again in again:
            await connection.send(frame_again)

    def _error(self, connection: _Connection, frame: frames.ErrorFrame) -> None:
        log.warning("error %s: %s", frame.error.code, frame.error.message)
        session = connection.sessions.get(frame.session_id or "")
        if session is None or not session._resuming:
            return
        session._resuming = False
        connection.sessions.pop(session.id, None)
        if frame.error.code == "connection_full":
            # This connection already carries max_sessions. The token is still good on another one.
            self._open([session], 0.0)
            return
        self._lost(session)

    def _forget(self, session: Session) -> None:
        self._sessions.pop(session.id, None)
        for connection in self._connections:
            connection.sessions.pop(session.id, None)

    def _lost(self, session: Session) -> None:
        self._forget(session)
        session._inbox.put_nowait(None)

    async def _closed(self, connection: _Connection) -> None:
        if connection not in self._connections:
            return
        self._connections.discard(connection)
        if self._stopped:
            return

        def queued(session: Session) -> bool:
            return any(session in other.pending for other in self._connections)

        lost = list(connection.sessions.values())
        movable = [s for s in lost if s.token and not s._resuming and not queued(s)]
        for session in lost:
            if session in movable or queued(session):
                continue
            log.warning("session %s lost: %s", session.id, "its resume was not answered" if session.token else "it has no resume token")
            self._lost(session)
        pending = [s for s in connection.pending if self._sessions.get(s.id) is s] + movable
        if connection.draining and not pending:
            return
        delay = self._backoff()
        log.info("reconnecting in %g s to resume %s session(s)", delay, len(pending))
        self._open(pending, delay)

    def _backoff(self) -> float:
        delay = 0.0 if self._attempt == 0 else min(MAX_BACKOFF_S, float(2 ** (self._attempt - 1)))
        self._attempt += 1
        return delay

    # ----- the session's task -----

    async def _run_session(self, session: Session) -> None:
        await self._safely(self.on_session_start(session))
        while True:
            item = await session._inbox.get()
            if item is None:
                await self._safely(self.on_session_lost(session))
                break
            if isinstance(item, bytes):
                await self._safely(self.on_audio(session, samples_of(item)))
            elif isinstance(item, frames.UserSaid):
                await self._safely(self.on_user_said(session, item))
            elif isinstance(item, frames.ToolResult):
                await self._safely(self.on_tool_result(session, item))
            elif isinstance(item, frames.SessionEnd):
                await self._safely(self.on_session_end(session, item))
                break
            elif isinstance(item, frames.UnknownFrame):
                await self._safely(self.on_frame(session, item))
        self._tasks.pop(session.id, None)

    async def _safely(self, handler: object) -> None:
        """Runs a handler; an exception in it is logged and does not take the session or the agent down."""
        try:
            await handler  # type: ignore[misc]
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("a handler failed")
