"""Voice wiring tests that need no network: tool schema and tool handler."""

from types import SimpleNamespace

from pipecat.frames.frames import EndWorkerFrame, TTSSpeakFrame

from support_agent.tools import TOOL_SPECS, ToolExecutor
from support_agent.voice.bot import build_tools_schema, make_tool_handler


class FakeLLM:
    def __init__(self):
        self.frames = []

    async def push_frame(self, frame, direction=None):
        self.frames.append(frame)


def make_params(llm, name, arguments, results):
    async def result_callback(result, properties=None):
        results.append((result, properties))

    return SimpleNamespace(
        function_name=name, arguments=arguments, llm=llm, result_callback=result_callback
    )


def test_tools_schema_lists_every_tool():
    schema = build_tools_schema()
    assert [f.name for f in schema.standard_tools] == [s.name for s in TOOL_SPECS]


async def test_handler_returns_result_to_llm(session):
    llm, results = FakeLLM(), []
    handler = make_tool_handler(ToolExecutor(session))
    await handler(make_params(llm, "lookup_order", {"order_number": "ORD-104233"}, results))
    assert results[0][0]["ok"]
    assert results[0][1] is None
    assert llm.frames == []


async def test_closing_tool_speaks_fixed_message_and_ends_call(session):
    llm, results = FakeLLM(), []
    handler = make_tool_handler(ToolExecutor(session))
    await handler(make_params(llm, "escalate_to_human", {"reason": "asked for a person"}, results))
    assert results[0][1].run_llm is False
    speak, end = llm.frames
    assert isinstance(speak, TTSSpeakFrame)
    assert "human agent" in speak.text
    assert isinstance(end, EndWorkerFrame)
