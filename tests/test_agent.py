"""The agent's bookkeeping around a connection, without a socket: which errors refuse a resume, when the
backoff starts over, and what a lost session's task sees."""

from __future__ import annotations

import asyncio

from sentdm_voice_agents import frames
from sentdm_voice_agents.agent import Agent, _Connection
from sentdm_voice_agents.session import Session, SessionEnded

from test_session import Recorder, start


def error(**fields) -> frames.ErrorFrame:
    return frames.decode({"type": "error", "session_id": "vs_1", "error": {"code": "frame_not_allowed", "message": "no"}, **fields})


def resuming_session(agent: Agent, connection: _Connection) -> Session:
    session = Session(start(), Recorder())
    session._resuming = True
    agent._sessions[session.id] = session
    connection.sessions[session.id] = session
    agent._connections.add(connection)
    return session


async def test_only_an_error_about_the_resume_refuses_it():
    agent = Agent(api_key="k", agent_id="agent_1")
    connection = _Connection([])
    session = resuming_session(agent, connection)
    agent._error(connection, error(frame_type="agent.say"))
    assert session._resuming and session.id in connection.sessions and not session.lost
    agent._error(connection, error(frame_type="session.resume", error={"code": "resume_failed", "message": "gone"}))
    assert session.lost and session.id not in connection.sessions and session.id not in agent._sessions
    assert await session._inbox.get() is None


async def test_a_full_connection_moves_the_resume_elsewhere():
    agent = Agent(api_key="k", agent_id="agent_1")
    connection = _Connection([])
    session = resuming_session(agent, connection)
    agent._error(connection, error(error={"code": "connection_full", "message": "full"}))
    assert not session.lost and session.id not in connection.sessions
    opened = [c for c in agent._connections if c is not connection]
    assert len(opened) == 1 and opened[0].pending == [session]
    for c in opened:
        c.task.cancel()
    await asyncio.gather(*(c.task for c in opened), return_exceptions=True)


async def test_the_backoff_starts_over_after_a_heartbeat_not_on_ready():
    agent = Agent(api_key="k", agent_id="agent_1")
    connection = _Connection([])
    agent._attempt = 3
    await agent._ready(connection, frames.decode({"type": "connection.ready", "protocol_version": "0.2.0", "heartbeat_interval_ms": 15000, "max_sessions": 10}))
    assert agent._attempt == 3
    await agent._handle(connection, frames.decode({"type": "ping", "ts": 1}))
    assert agent._attempt == 0
    await asyncio.gather(*agent._background, return_exceptions=True)


async def test_a_lost_session_reaches_its_handler_after_its_tool_call_fails():
    seen: list[str] = []

    class Noticing(Agent):
        async def on_session_start(self, session: Session) -> None:
            try:
                await session.hangup()
            except SessionEnded:
                seen.append("tool failed")

        async def on_session_lost(self, session: Session) -> None:
            seen.append("lost")

    agent = Noticing(api_key="k", agent_id="agent_1")
    connection = _Connection([])
    agent._connections.add(connection)
    agent._start(connection, start())
    await asyncio.sleep(0)
    session = agent.sessions[0]
    agent._lost(session)
    await asyncio.wait_for(agent._tasks.get(session.id) or asyncio.sleep(0), 1)
    assert seen == ["tool failed", "lost"] and agent.sessions == []
