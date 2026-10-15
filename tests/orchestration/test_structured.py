# Tests for structured output: validation, repair and the events it produces.

from dataclasses import replace

import pytest

from sensai.core import events as ev
from sensai.core.agent_spec import AgentSpec
from sensai.llm.ollama import ChatStats
from sensai.orchestration import structured
from sensai.orchestration.session import Session
from sensai.orchestration.turn import run_turn
from tests.fakes import ScriptedClient, make_config

SCHEMA = {
    "type": "object",
    "properties": {
        "correction": {"type": "string"},
        "error_type": {"type": "string", "enum": ["particle", "conjugation", "none"]},
    },
    "required": ["correction", "error_type"],
    "additionalProperties": False,
}
GOOD = '{"correction": "私は学校に行きます", "error_type": "particle"}'
BAD = '{"correction": "私は学校に行きます", "error_type": "grammar"}'


def test_schema_instructions_carry_the_schema():
    text = structured.schema_instructions(SCHEMA)
    assert "only one JSON object" in text and '"error_type"' in text and "particle" in text


@pytest.mark.parametrize("schema, think, expected", [
    (None, False, None),
    (SCHEMA, False, "json"),
    (SCHEMA, True, SCHEMA),
])
def test_request_format(schema, think, expected):
    spec = AgentSpec(name="a", model="m", num_ctx=1, output_schema=schema, think=think)
    assert structured.request_format(spec) == expected


def test_validate_reply_valid():
    assert structured.validate_reply(GOOD, SCHEMA) == ({"correction": "私は学校に行きます", "error_type": "particle"}, [])


@pytest.mark.parametrize("fenced", [f"```json\n{GOOD}\n```", f"```\n{GOOD}\n```", f"  {GOOD}  "])
def test_validate_reply_strips_code_fences(fenced):
    data, problems = structured.validate_reply(fenced, SCHEMA)
    assert problems == [] and data["error_type"] == "particle"


def test_validate_reply_reports_schema_errors_with_their_path():
    data, problems = structured.validate_reply(BAD, SCHEMA)
    assert data is None and len(problems) == 1 and problems[0].startswith("error_type: 'grammar'")


def test_validate_reply_reports_missing_and_extra_fields():
    _, problems = structured.validate_reply('{"correction": "x", "extra": 1}', SCHEMA)
    assert any("'error_type' is a required property" in p for p in problems)
    assert any("Additional properties" in p for p in problems)
    assert all(p.startswith("reply:") for p in problems)


@pytest.mark.parametrize("text", ["私は学校へ行きます", "", "```"])
def test_validate_reply_reports_invalid_json(text):
    data, problems = structured.validate_reply(text, SCHEMA)
    assert data is None and problems[0].startswith("not valid JSON")


def test_repair_request_lists_the_problems():
    msg = structured.repair_request(["a: wrong", "b: missing"])
    assert "- a: wrong\n- b: missing" in msg and msg.endswith("only the corrected JSON object.")


def test_failure_message_quotes_and_truncates_the_reply():
    short = structured.failure_message(["x: bad"], "abc")
    assert "x: bad" in short and short.endswith("Reply was: abc")
    long = structured.failure_message(["x: bad"], "z" * 2000)
    assert long.endswith("z" * structured.MAX_SHOWN + "…")


def structured_session(*replies, think=False, stats=None):
    session = Session(make_config(), client=ScriptedClient(*replies, stats=stats))
    session.spec = replace(session.spec, output_schema=SCHEMA, think=think)
    return session


async def collect(session, text="私は学校を行きます"):
    return [e async for e in run_turn(session, text)]


def kinds(events):
    return [type(e).__name__ for e in events]


async def test_valid_reply_becomes_structured_output_without_tokens():
    session = structured_session(GOOD)
    events = await collect(session)
    assert kinds(events) == ["TurnStarted", "AgentRunStarted", "StructuredOutput",
                             "AgentRunFinished", "TurnFinished"]
    out = events[2]
    assert out.data["error_type"] == "particle" and out.attempts == 1
    assert out.run_id == events[1].run_id and events[3].ok is True
    assert session.history[-1] == {"role": "assistant", "content": GOOD}


async def test_schema_is_in_the_prompt_and_json_mode_is_requested():
    session = structured_session(GOOD)
    await collect(session)
    _, messages, options = session.client.calls[0]
    assert options["format"] == "json"
    assert messages[0]["role"] == "system" and '"error_type"' in messages[0]["content"]
    assert messages[0]["content"].startswith(session.spec.system_prompt)


async def test_schema_is_sent_as_format_when_thinking():
    session = structured_session(GOOD, think=True)
    await collect(session)
    assert session.client.calls[0][2]["format"] == SCHEMA


async def test_one_repair_attempt_fixes_the_reply():
    session = structured_session(BAD, GOOD)
    events = await collect(session)
    out = next(e for e in events if isinstance(e, ev.StructuredOutput))
    assert out.attempts == 2 and out.data["error_type"] == "particle"
    _, retry, _ = session.client.calls[1]
    assert retry[-2] == {"role": "assistant", "content": BAD}
    assert retry[-1]["role"] == "user" and "error_type: 'grammar'" in retry[-1]["content"]
    assert session.history[-1]["content"] == GOOD


async def test_second_failure_is_reported_loudly_and_rolled_back():
    session = structured_session(BAD, "not json at all")
    events = await collect(session)
    assert kinds(events) == ["TurnStarted", "AgentRunStarted", "Error", "AgentRunFinished",
                             "TurnFinished"]
    error = events[2]
    assert error.kind == "invalid_output"
    assert "not valid JSON" in error.message and "not json at all" in error.message
    assert events[3].ok is False
    assert session.history == [] and len(session.client.calls) == 2


async def test_stats_are_summed_over_the_repair():
    stats = ChatStats(prompt_eval_count=100, eval_count=20, total_duration_ns=1_000)
    session = structured_session(BAD, GOOD, stats=stats)
    finished = (await collect(session))[-1]
    assert (finished.prompt_eval_count, finished.eval_count) == (200, 40)
    assert finished.total_duration_ns == 2_000


async def test_agents_without_schema_still_stream_text():
    session = Session(make_config(), client=ScriptedClient("Konnichiwa"))
    events = await collect(session, "hello")
    assert "Token" in kinds(events) and "StructuredOutput" not in kinds(events)
    assert "format" not in session.client.calls[0][2]
