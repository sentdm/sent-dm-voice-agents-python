# Sent Voice Agents for Python

An AI agent that takes and places phone calls through [Sent](https://sent.dm), on the
[Sent Voice Agents protocol](https://github.com/sentdm/sent-dm-voice-protocol) 0.2.0.

Status: Preview, with the protocol. Generated and rendered by `sent-dm-sdk-pipeline` from the protocol at
`v0.2.0`; every release passes the protocol's conformance scenarios in CI.

Read the [Voice Agents docs](https://docs.sent.dm) to get started.

## Install

```sh
pip install sentdm-voice-agents
```

Python 3.11 or later. The one dependency is `websockets`.

## Quickstart

```python
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

Support(api_key="sk_live_...", agent_id="agent_...").serve()
```

`serve()` connects to Sent with the account API key and serves calls until SIGINT or SIGTERM; `await
agent.run()` does the same inside your own event loop. Each session's handlers run in order on their own task,
so sessions never wait for one another. The runtime answers heartbeats, opens a new connection when one drains
or is lost, and resumes the sessions that can move; the handlers only see calls.

| Handler | When |
| --- | --- |
| `on_session_start(session)` | A call reached the agent. In a text session, `say()` to speak |
| `on_user_said(session, said)` | Text sessions: the caller said something; `said.correction` means it replaces the previous one |
| `on_audio(session, samples)` | Audio sessions: the caller's audio, 16-bit mono PCM at `session.sample_rate` |
| `on_tool_result(session, result)` | Sent answered a tool call; the awaiting call gets the same result |
| `on_session_end(session, end)` | The call left the agent; `end.reason` says why |
| `on_session_lost(session)` | The connection was lost and the session could not be resumed |

On a `Session`: `say(text, last=True, interruptible=True)`, `clear()`, `send_audio(samples)`, and the tools
`transfer(to)`, `hangup()`, `send_dtmf(digits)`, `hold(held)`, each returning Sent's `ToolResult`. `session.from_`,
`to`, `direction`, `session_type`, `language`, `context`, `sample_rate` carry what `session.start` said.

`sentdm_voice_agents.frames` holds every frame as a type, with `decode()` and `encode()` that check the
protocol's rules; it is generated from the protocol and not edited by hand.

## Develop

```sh
uv sync                     # the package and its test dependencies
uv run pytest               # the runtime and the frames against the protocol's fixtures
```

The conformance scenarios run in CI through the protocol's action, against `python -m sentdm_voice_agents.conformance`.
Releases are cut by release-please from `main` and published to PyPI from GitHub Actions.
