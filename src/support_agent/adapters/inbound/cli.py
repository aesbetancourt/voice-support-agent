"""Inbound adapter: the command line, ``support-agent chat | voice | simulate``."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from google.auth.exceptions import GoogleAuthError
from google.genai import errors as genai_errors
from loguru import logger

from support_agent.adapters.inbound.simulate import PERSONAS, run_persona
from support_agent.adapters.outbound.gemini import make_client
from support_agent.bootstrap import (
    make_sentiment_analyzer,
    make_summarizer,
    make_text_agent,
    new_conversation,
)
from support_agent.config import ConfigError, Settings
from support_agent.domain.models import ConversationRecord

MAX_INPUT_CHARS = 2000
VOICE_MODULE = "support_agent.adapters.inbound.voice"
_QUIT_COMMANDS = {"/quit", "/exit"}
_TECHNICAL_ERROR = "Sorry, I have a technical problem and must stop. Your conversation is saved."


async def run_chat(settings: Settings) -> None:
    client = make_client(settings)
    conversation = new_conversation(
        settings,
        "text",
        make_summarizer(settings, client),
        make_sentiment_analyzer(settings, client),
    )
    agent = make_text_agent(settings, client, conversation)
    print("Orbio Store support chat. Type /quit to leave.\n")
    # Config, auth and model errors show up here, before anything is saved.
    reply = await agent.greet()
    print(f"agent> {reply.text}\n")
    try:
        while not reply.ended:
            try:
                user_text = (await asyncio.to_thread(input, "you> ")).strip()
            except EOFError:
                break
            if not user_text:
                continue
            if user_text.lower() in _QUIT_COMMANDS:
                break
            reply = await agent.respond(user_text[:MAX_INPUT_CHARS])
            print(f"\nagent> {reply.text}\n")
    except genai_errors.APIError as exc:
        logger.error(f"Gemini API error: {exc}")
        print(f"\nagent> {_TECHNICAL_ERROR}\n")
    finally:
        record = await conversation.recorder.finish()
        _print_record(record, conversation.recorder.location)


async def run_simulations(settings: Settings, persona_name: str, max_turns: int) -> int:
    personas = [p for p in PERSONAS if persona_name in ("all", p.name)]
    if not personas:
        names = ", ".join(p.name for p in PERSONAS)
        print(f"Unknown persona '{persona_name}'. Choose one of: all, {names}.")
        return 2

    client = make_client(settings)
    failures = 0
    for persona in personas:
        result = await run_persona(settings, client, persona, max_turns)
        mark = "PASS" if result.passed else "FAIL"
        failures += not result.passed
        print(
            f"{mark}  {persona.name:<20} expected={persona.expected.value:<10} "
            f"got={result.record.status.value:<10} turns={result.turns:<3} id={result.record.id}"
        )
    print(f"\n{len(personas) - failures}/{len(personas)} personas passed.")
    return 1 if failures else 0


def _print_record(record: ConversationRecord, location: str) -> None:
    print("-" * 60)
    print(f"Status: {record.status.value}")
    print(f"Ticket: {record.ticket.model_dump_json(exclude_none=True)}")
    if record.summary:
        print(f"Summary: {record.summary.headline}")
    print(f"Saved:  {location}")


def _run_voice(extra_args: list[str]) -> None:
    # The Pipecat runner finds the bot through the __main__ module,
    # so start the bot module as the main program. Only the WebRTC transport is used.
    if not {"-t", "--transport"} & set(extra_args):
        extra_args = ["-t", "webrtc", *extra_args]
    os.execv(sys.executable, [sys.executable, "-m", VOICE_MODULE, *extra_args])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="support-agent", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("chat", help="Talk to the agent by text in the terminal.")
    commands.add_parser(
        "voice",
        help="Start the voice server (open http://localhost:7860). Extra args go to Pipecat.",
    )
    simulate = commands.add_parser("simulate", help="Run simulated customers against the agent.")
    simulate.add_argument("--persona", default="all", help="Persona name, or 'all'.")
    simulate.add_argument("--max-turns", type=int, default=16)

    args, extra = parser.parse_known_args(argv)
    if args.command == "voice":
        _run_voice(extra)
        return
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "WARNING"))
    try:
        settings = Settings.from_env()
        if args.command == "chat":
            asyncio.run(run_chat(settings))
        else:
            sys.exit(asyncio.run(run_simulations(settings, args.persona, args.max_turns)))
    except ConfigError as exc:
        sys.exit(f"Configuration error: {exc}")
    except GoogleAuthError as exc:
        sys.exit(
            f"Google Cloud authentication failed: {exc}\nRun: gcloud auth application-default login"
        )
    except genai_errors.APIError as exc:
        sys.exit(f"Gemini API error: {exc}")
    except KeyboardInterrupt:
        print("\nStopped.")
