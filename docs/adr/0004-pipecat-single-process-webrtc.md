# Use Pipecat with SmallWebRTC in one process, with the prebuilt browser UI

The voice channel is built with Pipecat. Its development runner serves the prebuilt browser client (`/client`) and the WebRTC signaling endpoint from the same Python process that runs the bot. After the handshake, audio flows peer-to-peer between the browser and Python. There is no separate frontend service and no media server: a reviewer runs one command and opens `localhost`.

## Considered Options

- **LiveKit Agents.** A comparable framework (`AgentTask`, `TaskGroup`), but it needs a LiveKit server or cloud account. Rejected: more infrastructure for no gain in this case.
- **A hosted transport (Daily, LiveKit Cloud).** Rejected for the same reason. It becomes relevant for production scale or telephony.
- **A custom React frontend.** Rejected: it would add a Node toolchain and a second service, and the prebuilt client already covers connect, microphone and playback.
- **A local microphone and speaker in the terminal.** Rejected: the browser gives a better demo.

## Consequences

- The Pipecat runner finds the bot through the `__main__` module. So `support-agent voice` starts the voice adapter as its own program (`python -m support_agent.adapters.inbound.voice`).
- Scaling to many concurrent calls means splitting signaling from bot workers. This is listed as a potential improvement.
