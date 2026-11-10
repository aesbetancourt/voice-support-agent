"""Inbound adapter: the command line, ``screening-agent chat | voice | simulate``."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from google.auth.exceptions import GoogleAuthError
from google.genai import errors as genai_errors
from loguru import logger

from screening_agent.adapters.inbound.simulate import PERSONAS, run_persona
from screening_agent.adapters.outbound.gemini import make_client
from screening_agent.application.messages import technical_error_message
from screening_agent.bootstrap import (
    make_sentiment_analyzer,
    make_summarizer,
    make_text_agent,
    new_conversation,
)
from screening_agent.config import ConfigError, Settings
from screening_agent.domain.models import ConversationRecord, Language
from screening_agent.domain.validators import parse_language

# A long answer fits easily; a pasted document does not reach the model.
MAX_INPUT_CHARS = 2000
VOICE_MODULE = "screening_agent.adapters.inbound.voice"
_QUIT_COMMANDS = {"/quit", "/exit"}


async def run_chat(settings: Settings) -> None:
    client = make_client(settings)
    conversation = new_conversation(
        settings,
        "text",
        make_summarizer(settings, client),
        make_sentiment_analyzer(settings, client),
    )
    agent = make_text_agent(settings, client, conversation)
    print("Harbor Logistics screening chat. Type /quit to leave.\n")
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
        print(f"\nagent> {technical_error_message(conversation.recorder.language)}\n")
    finally:
        await agent.wait_for_reading()
        record = await conversation.recorder.finish()
        _print_record(record, conversation.recorder.location)


async def run_simulations(
    settings: Settings, persona_name: str, max_turns: int, with_judge: bool
) -> int:
    personas = [p for p in PERSONAS if persona_name in ("all", p.name)]
    if not personas:
        names = ", ".join(p.name for p in PERSONAS)
        print(f"Unknown persona '{persona_name}'. Choose one of: all, {names}.")
        return 2

    client = make_client(settings)
    failures = 0
    for persona in personas:
        result = await run_persona(settings, client, persona, max_turns, with_judge)
        mark = "PASS" if result.passed else "FAIL"
        failures += not result.passed
        print(
            f"{mark}  {persona.name:<20} expected={persona.expected.value:<12} "
            f"got={result.record.status.value:<12} turns={result.turns:<3} id={result.record.id}"
        )
        for criterion, reason in (result.verdict.failures() if result.verdict else {}).items():
            print(f"      judge: {criterion}: {reason}")
    print(f"\n{len(personas) - failures}/{len(personas)} personas passed.")
    return 1 if failures else 0


def _print_record(record: ConversationRecord, location: str) -> None:
    print("-" * 60)
    print(f"Status: {record.status.value} ({record.language.value})")
    card = record.scorecard
    print(f"Candidate: {card.candidate_name or '?'}  Recommendation: {card.recommendation.value}")
    if card.interview:
        print(f"Interview: {card.interview.starts_at:%A %Y-%m-%d %H:%M} ({card.booking_id})")
    if record.summary:
        print(f"Summary: {record.summary.headline}")
    print(f"Saved:  {location}")


def _language_arg(raw: str) -> Language:
    try:
        return parse_language(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _run_voice(extra_args: list[str]) -> None:
    # The Pipecat runner finds the bot through the __main__ module,
    # so start the bot module as the main program. Only the WebRTC transport is used.
    if not {"-t", "--transport"} & set(extra_args):
        extra_args = ["-t", "webrtc", *extra_args]
    os.execv(sys.executable, [sys.executable, "-m", VOICE_MODULE, *extra_args])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="screening-agent", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="Talk to the agent by text in the terminal.")
    voice = commands.add_parser(
        "voice",
        help="Start the voice server (open http://localhost:7860). Extra args go to Pipecat.",
    )
    simulate = commands.add_parser("simulate", help="Run simulated candidates against the agent.")
    simulate.add_argument("--persona", default="all", help="Persona name, or 'all'.")
    # A long enough limit for every persona; it stops a simulation that never ends.
    simulate.add_argument("--max-turns", type=int, default=24)
    simulate.add_argument(
        "--no-judge", action="store_true", help="Check the final status only, without the judge."
    )
    for command in (chat, voice, simulate):
        command.add_argument(
            "--language",
            type=_language_arg,
            help="Conversation language: en or es. Default: DEFAULT_LANGUAGE in .env, else en.",
        )

    args, extra = parser.parse_known_args(argv)
    if args.language is not None:
        # Settings read the language from the environment; the voice process inherits it.
        os.environ["DEFAULT_LANGUAGE"] = args.language.value
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
            exit_code = asyncio.run(
                run_simulations(settings, args.persona, args.max_turns, not args.no_judge)
            )
            sys.exit(exit_code)
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
