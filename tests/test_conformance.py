from sentdm_voice_agents.conformance.__main__ import decide


def test_commands():
    assert decide("echo hello there") == [("say", {"text": "hello there"})]
    assert decide("stream Let me think") == [("say", {"text": "Let", "last": False}), ("say", {"text": "me think"})]
    assert decide("transfer") == [("tool", {"name": "transfer", "arguments": {"to": {"kind": "user", "value": "conformance-desk"}}})]
    assert decide("hangup") == [("tool", {"name": "hangup"})]
    assert decide("dtmf 12#") == [("tool", {"name": "send_dtmf", "arguments": {"digits": "12#"}})]
    assert decide("hold") == [("tool", {"name": "hold", "arguments": {"held": True}})]
    assert decide("something else") == []
