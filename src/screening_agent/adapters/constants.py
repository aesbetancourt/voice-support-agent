"""Protocol strings that the adapters share: fixed by Pipecat, RTVI or Gemini, not tunable.

Tunable values (models, voices, thresholds) are in ``config/settings.toml`` instead.
"""

from __future__ import annotations

# LLM context role for instructions that are not the candidate's words.
DEVELOPER_ROLE = "developer"

# Key in the browser client's connection body that chooses the greeting language.
CLIENT_LANGUAGE_KEY = "language"

# Types of the RTVI server messages that the browser client shows in its Events panel.
SERVER_MESSAGE_LANGUAGE = "language"
SERVER_MESSAGE_FRUSTRATION = "frustration"

# The Pipecat development runner's transport for the browser demo.
WEBRTC_TRANSPORT = "webrtc"

# Attribute of Google's raw STT result that holds the detected language of an utterance.
GOOGLE_LANGUAGE_CODE_ATTR = "language_code"

# Why the voice pipeline ends after a closing message.
END_REASON_CLOSED = "conversation closed"

# Tags that Gemini sometimes writes as plain reply text; they must never reach the candidate.
THOUGHT_OPEN_TAG = "<thought>"
THOUGHT_CLOSE_TAG = "</thought>"
