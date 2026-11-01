"""The session's bookkeeping, with a connection that records what it sends: counting both ways, the retained
frames, what a resume sends again, and tool calls answered by their result."""

from __future__ import annotations

import asyncio

import pytest

from sentdm_voice_agents import frames
from sentdm_voice_agents.session import Session, SessionEnded


class Recorder:
    def __init__(self) -> None:
        self.frames: list[frames.Frame] = []
        self.audio: list[bytes] = []

    async def send(self, frame):
        self.frames.append(frame)

    async def send_bytes(self, data):
        self.audio.append(data)


def start(**extra) -> frames.SessionStart:
    return frames.decode({
        "type": "session.start", "session_id": "vs_1", "call_id": "call_1", "agent_id": "agent_1",
        "direction": "inbound", "from": {"kind": "number", "value": "+14155550100"},
        "to": {"kind": "agent", "value": "agent_1"}, "session_type": "text", "started_at": "2026-10-31T00:00:00Z",
        "resume": {"token": "vsr_" + "a" * 30, "window_ms": 15000}, **extra,
    })


async def test_counts_both_ways_and_keeps_the_latest_frames():
    recorder = Recorder()
    session = Session(start(), recorder)
    assert session.resumable and session.received == 0 and session.sent == 0
    session._read(frames.decode({"type": "user.said", "session_id": "vs_1", "text": "hi"}))
    session._read(frames.decode({"type": "session.resumed", "session_id": "vs_1", "received": 0}))  # uncounted
    assert session.received == 1
    for i in range(frames.RETAINED_FRAMES + 5):
        await session.say(f"line {i}")
    assert session.sent == frames.RETAINED_FRAMES + 5
    assert len(session._kept) == frames.RETAINED_FRAMES and session._before == 5
    assert len(recorder.frames) == frames.RETAINED_FRAMES + 5


async def test_resume_sends_again_what_sent_did_not_read():
    session = Session(start(), Recorder())
    await session.say("one")
    await session.say("two")
    await session.say("three")
    resume = session._resume_frame()
    assert resume.token.startswith("vsr_") and resume.received == 0
    again = session._resumed(frames.decode({"type": "session.resumed", "session_id": "vs_1", "received": 1, "resume": {"token": "vsr_" + "b" * 30, "window_ms": 15000}}))
    assert [f.text for f in again] == ["two", "three"]
    assert session.token.startswith("vsr_b") and session.sent == 3


async def test_resume_past_the_retained_frames_hangs_up():
    session = Session(start(), Recorder())
    for i in range(frames.RETAINED_FRAMES + 3):
        await session.say(str(i))
    again = session._resumed(frames.decode({"type": "session.resumed", "session_id": "vs_1", "received": 1, "resume": {"token": "vsr_" + "c" * 30, "window_ms": 15000}}))
    assert len(again) == 1 and isinstance(again[0], frames.ToolCall) and again[0].name == "hangup"


async def test_tool_calls_are_answered_by_their_result():
    recorder = Recorder()
    session = Session(start(), recorder)
    task = asyncio.create_task(session.transfer("+14155550199"))
    await asyncio.sleep(0)
    call = recorder.frames[-1]
    assert isinstance(call, frames.ToolCall) and call.name == "transfer" and call.arguments == {"to": {"kind": "number", "value": "+14155550199"}}
    session._answer_tool(frames.decode({"type": "tool.result", "session_id": "vs_1", "call_ref": call.call_ref, "ok": True}))
    result = await task
    assert result.ok


async def test_a_session_that_ended_refuses_more():
    session = Session(start(), Recorder())
    task = asyncio.create_task(session.hangup())
    await asyncio.sleep(0)
    session._finish(frames.decode({"type": "session.end", "session_id": "vs_1", "reason": "caller_hung_up", "duration_ms": 10}))
    with pytest.raises(SessionEnded):
        await task
    with pytest.raises(SessionEnded):
        await session.say("too late")


async def test_audio_goes_out_on_the_session_stream():
    recorder = Recorder()
    session = Session(start(session_type="audio", audio={"stream": 7}), recorder)
    await session.send_audio(b"\x01\x00" * 320)
    assert recorder.audio[0][:2] == b"\x00\x07" and len(recorder.audio[0]) == 642


async def test_a_frame_that_breaks_the_protocol_is_refused_before_it_counts():
    recorder = Recorder()
    session = Session(start(), recorder)
    with pytest.raises(frames.FrameError):
        await session.say("")
    with pytest.raises(frames.FrameError):
        await session.send_dtmf("xyz")
    assert session.sent == 0 and recorder.frames == []


async def test_frames_sent_during_a_resume_wait_for_the_replay():
    recorder = Recorder()
    session = Session(start(), recorder)
    await session.say("before")
    session._resuming = True
    await session.say("during")
    task = asyncio.create_task(session.hangup())
    await asyncio.sleep(0)
    assert [f.text for f in recorder.frames] == ["before"] and session.sent == 3
    again = session._resumed(frames.decode({"type": "session.resumed", "session_id": "vs_1", "received": 1, "resume": {"token": "vsr_" + "d" * 30, "window_ms": 15000}}))
    assert [getattr(f, "text", getattr(f, "name", None)) for f in again] == ["during", "hangup"]
    session._answer_tool(frames.decode({"type": "tool.result", "session_id": "vs_1", "call_ref": again[1].call_ref, "ok": True}))
    assert (await task).ok


async def test_audio_during_a_resume_is_dropped_and_after_the_end_refused():
    recorder = Recorder()
    session = Session(start(session_type="audio", audio={"stream": 7}), recorder)
    session._resuming = True
    await session.send_audio(b"\x01\x00" * 320)
    assert recorder.audio == []
    session._resuming = False
    session._finish(frames.decode({"type": "session.end", "session_id": "vs_1", "reason": "caller_hung_up", "duration_ms": 10}))
    with pytest.raises(SessionEnded):
        await session.send_audio(b"\x01\x00" * 320)


async def test_a_lost_session_fails_its_tool_calls_and_refuses_more():
    session = Session(start(), Recorder())
    task = asyncio.create_task(session.hangup())
    await asyncio.sleep(0)
    session._abandon()
    with pytest.raises(SessionEnded):
        await task
    assert session.ended and session.lost
    with pytest.raises(SessionEnded):
        await session.say("too late")


async def test_unknown_session_frames_count_and_errors_do_not():
    session = Session(start(), Recorder())
    session._read(frames.decode({"type": "session.note", "session_id": "vs_1", "x": 1}))
    assert session.received == 1
    session._read(frames.decode({"type": "error", "session_id": "vs_1", "error": {"code": "frame_not_allowed", "message": "no"}}))
    assert session.received == 1
