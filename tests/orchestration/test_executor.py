# Tests for the agent executor: one AgentSpec run on its own, its events, failures and stats.

from sensai.core import events as ev
from sensai.core.agent_spec import AgentSpec
from sensai.llm.ollama import ChatStats, OllamaError, Thought
from sensai.orchestration.executor import AgentRun, sum_stats, system_prompt
from tests.fakes import FakeClient

SPEC = AgentSpec(name="corrector", model="qwen3.5:9b", num_ctx=4096, system_prompt="Correct it.")
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
MESSAGES = [{"role": "user", "content": "私は学校を行きます"}]


class ThinkingClient(FakeClient):

    def __init__(self, *replies):
        super().__init__()
        self.replies = list(replies)

    def chat_stream(self, model, messages, **options):
        self.calls.append((model, messages, options))
        yield Thought("checking")
        yield self.replies.pop(0)


async def run_events(run, messages=MESSAGES):
    return [event async for event in run.events(messages)]


def kinds(events):
    return [type(event).__name__ for event in events]


async def test_run_emits_its_events_between_start_and_finish():
    run = AgentRun(FakeClient(stats=ChatStats(eval_count=2)), SPEC, "turn-1", parent_run_id="parent")
    events = await run_events(run)
    assert kinds(events) == ["AgentRunStarted", "Token", "Token", "AgentRunFinished"]
    assert events[0].agent == "corrector" and events[0].parent_run_id == "parent"
    assert {event.run_id for event in events} == {run.run_id}
    assert {event.turn_id for event in events} == {"turn-1"}
    assert run.ok and run.text == "Konnichiwa" and run.stats == ChatStats(eval_count=2)
    assert run.first_output_ns is not None


async def test_run_sends_the_messages_with_the_spec_settings():
    client = FakeClient()
    await run_events(AgentRun(client, SPEC, "turn-1"))
    model, messages, options = client.calls[0]
    assert model == "qwen3.5:9b" and messages == MESSAGES
    assert options == {"num_ctx": 4096, "temperature": 0.3, "think": False}


async def test_model_failure_ends_the_run_not_ok():
    run = AgentRun(FakeClient(error=OllamaError("timeout", "too slow")), SPEC, "turn-1")
    events = await run_events(run)
    assert kinds(events) == ["AgentRunStarted", "Error", "AgentRunFinished"]
    assert events[1].kind == "timeout" and events[2].ok is False
    assert not run.ok and run.text == "" and run.first_output_ns is None


async def test_structured_run_keeps_the_data_and_thinks_during_the_repair():
    spec = AgentSpec(name="checker", model="m", num_ctx=1, output_schema=SCHEMA)
    run = AgentRun(ThinkingClient("not json", '{"ok": true}'), spec, "turn-1")
    events = await run_events(run)
    assert kinds(events) == ["AgentRunStarted", "Thinking", "Thinking", "StructuredOutput",
                             "AgentRunFinished"]
    assert run.data == {"ok": True} and events[3].attempts == 2 and run.ok


def test_system_prompt_adds_the_schema_only_when_there_is_one():
    assert system_prompt(SPEC) == "Correct it."
    structured_spec = AgentSpec(name="checker", model="m", num_ctx=1, system_prompt="Check.",
                                output_schema=SCHEMA)
    assert system_prompt(structured_spec).startswith("Check.\n\n") and '"ok"' in system_prompt(structured_spec)


def test_sum_stats_skips_missing_stats():
    stats = ChatStats(eval_count=3, total_duration_ns=10)
    assert sum_stats(None, None) is None
    assert sum_stats(None, stats) == stats and sum_stats(stats, None) == stats
    assert sum_stats(stats, stats) == ChatStats(eval_count=6, total_duration_ns=20)
