"""The generated frames against the protocol's own fixtures, copied here by the lane at render time."""

import pytest

from sentdm_voice_agents import frames
from conftest import fixture_files


@pytest.mark.parametrize("path", fixture_files("valid"), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_valid_frames_round_trip(path, load):
    data = load(path)
    frame = frames.decode(data)
    assert not isinstance(frame, frames.UnknownFrame)
    again = frames.encode(frame)
    for key, value in data.items():
        assert again.get(key) == value, f"{key} changed on the way back"
    assert frames.decode(again) == frame


@pytest.mark.parametrize("path", fixture_files("invalid"), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_invalid_frames_are_refused(path, load):
    with pytest.raises(frames.FrameError):
        frames.decode(load(path))


def test_unknown_frames_and_fields_pass():
    frame = frames.decode({"type": "session.note", "session_id": "vs_1"})
    assert isinstance(frame, frames.UnknownFrame)
    said = frames.decode({"type": "user.said", "session_id": "vs_1", "text": "hi", "mood": "good"})
    assert said.correction is False
