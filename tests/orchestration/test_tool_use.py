# Tests for the executor's tool loop (native and prompted tool calling) and its ReAct strategy.

import json

from sensai.core import events as ev
from sensai.core.agent_spec import AgentSpec
from sensai.core.handoff import Handoff
from sensai.knowledge.tools import registry
from sensai.llm.ollama import ChatStats, OllamaError, ToolRequest
from sensai.orchestration.executor import AgentRun
from tests.fakes import FakeClient

SENSEI = AgentSpec(name="sensei", model="qwen3.5:9b", num_ctx=4096, system_prompt="Teach.")
CORRECTOR = AgentSpec(name="corrector", model="qwen3.5:9b", num_ctx=1024, strategy="react", max_steps=3,
                      handoffs=("sensei",), output_schema={"type": "object"})
QUESTION = [{"role": "system", "content": "Teach."}, {"role": "user", "content": "What is the te-form of 行く?"}]


class ToolClient(FakeClient):
    """Replays one scripted reply per call; a reply is a list of pieces (text or ToolRequest)."""

    def __init__(self, *replies, native=True):
        super().__init__(stats=ChatStats(prompt_eval_count=5, eval_count=2))
        self.replies = list(replies)
        self.native = native

    def capabilities(self, model):
        return {"completion", "tools"} if self.native else {"completion"}

    def chat_stream(self, model, messages, **options):
        self.calls.append((model, [dict(message) for message in messages], options))
        yield from self.replies.pop(0)
        return self.stats


async def run_events(run, messages=QUESTION):
    return [event async for event in run.events(messages)]


def kinds(events):
    return [type(event).__name__ for event in events]


def tools():
    return registry.subset(["dictionary", "conjugate", "kanji"])


async def test_a_native_tool_call_runs_the_tool_and_feeds_its_result_back():
    client = ToolClient([ToolRequest("conjugate", {"verb": "行く", "form": "te"})], ["It is ", "行って."])
    run = AgentRun(client, SENSEI, "t", tools=tools())
    events = await run_events(run)
    assert kinds(events) == ["AgentRunStarted", "ToolCall", "ToolResult", "Token", "Token", "AgentRunFinished"]
    assert json.loads(events[2].result)["result"] == "行って" and events[2].ok
    assert run.text == "It is 行って." and run.ok and run.stats.eval_count == 4
    first, second = client.calls
    assert [tool["function"]["name"] for tool in first[2]["tools"]] == ["dictionary", "conjugate", "kanji"]
    assert second[1][-2]["tool_calls"][0]["function"]["name"] == "conjugate"
    assert second[1][-1] == {"role": "tool", "content": events[2].result, "tool_name": "conjugate"}


async def test_the_last_round_offers_no_tools_and_asks_for_the_answer():
    call = [ToolRequest("kanji", {"character": "行"})]
    client = ToolClient(call, call, ["Done."])
    run = AgentRun(client, SENSEI, "t", tools=tools(), max_tool_rounds=2)
    events = await run_events(run)
    assert kinds(events).count("ToolCall") == 2 and run.text == "Done."
    assert "tools" not in client.calls[-1][2] and "without calling" in client.calls[-1][1][-1]["content"]


async def test_a_failing_tool_is_reported_and_the_run_goes_on():
    client = ToolClient([ToolRequest("conjugate", {"verb": "猫", "form": "te"})], ["猫 is not a verb."])
    events = await run_events(AgentRun(client, SENSEI, "t", tools=tools()))
    result = next(event for event in events if isinstance(event, ev.ToolResult))
    assert result.ok is False and events[-1].ok is True


async def test_a_prompted_tool_call_is_held_back_from_the_learner():
    client = ToolClient(['  {"tool": "dictionary",', ' "args": {"word": "猫"}}'], ["Neko ", "means cat."],
                        native=False)
    run = AgentRun(client, SENSEI, "t", tools=tools())
    events = await run_events(run)
    assert kinds(events) == ["AgentRunStarted", "ToolCall", "ToolResult", "Token", "Token", "AgentRunFinished"]
    assert run.text == "Neko means cat."
    first, second = client.calls
    assert "tools" not in first[2] and "You have access to the following tools" in first[1][0]["content"]
    assert second[1][-1]["content"].startswith("Tool result (dictionary):")


async def test_a_prompted_reply_that_is_not_a_call_reaches_the_learner():
    client = ToolClient(['{"just": "data"}'], native=False)
    run = AgentRun(client, SENSEI, "t", tools=tools())
    events = await run_events(run)
    assert kinds(events) == ["AgentRunStarted", "Token", "AgentRunFinished"] and run.text == '{"just": "data"}'


async def test_prompted_mode_without_a_system_message_adds_one():
    client = ToolClient(["Plain answer."], native=False)
    await run_events(AgentRun(client, SENSEI, "t", tools=tools()), [{"role": "user", "content": "hi"}])
    assert client.calls[0][1][0]["role"] == "system"


REACT_STEPS = [
    ['Thought: を looks wrong with 行く.\nAction: {"tool": "dictionary", "args": {"word": "行く"}}\nObservation: made up'],
    ["Verification: the note says に, never を.\nRevised: no\nRule: 行く takes に for the destination.\n"
     "Correction: 私は学校に行きます\nFinal Answer: 私は学校に行きます (gakkou ni ikimasu) — use に."],
]


async def test_the_react_strategy_streams_its_steps_and_hands_off_verified_data():
    client = ToolClient(*REACT_STEPS)
    run = AgentRun(client, CORRECTOR, "t", tools=tools())
    run.handoff = Handoff("checker", "corrector", "particle", {"error_type": "particle"}, 1, "r0")
    events = await run_events(run, [{"role": "user", "content": "私は学校を行きます"}])
    assert kinds(events) == ["AgentRunStarted", "Thinking", "ToolCall", "ToolResult", "Thinking",
                             "StructuredOutput", "AgentRunFinished"]
    data = run.data
    assert (data["correction"], data["verified"], data["verified_with"]) == ("私は学校に行きます", True, ["dictionary"])
    assert data["handoff"]["to"] == "sensei" and data["first_check"] == {"error_type": "particle"}
    assert "Observation: made up" not in client.calls[1][1][-2]["content"]
    assert "first check suggested" in client.calls[0][1][1]["content"]
    assert all(call[2]["num_ctx"] == 1024 for call in client.calls)
    assert run.stats.eval_count == 4 and run.ok


async def test_a_final_answer_before_any_tool_is_sent_back_for_verification():
    client = ToolClient(["Final Answer: 私は学校に行きます"], *REACT_STEPS)
    run = AgentRun(client, CORRECTOR, "t", tools=tools())
    await run_events(run, [{"role": "user", "content": "私は学校を行きます"}])
    assert "verify with a tool" in client.calls[1][1][-1]["content"] and run.data["verified"]


async def test_a_corrector_that_never_verifies_is_marked_unverified():
    client = ToolClient(["Final Answer: x"], ["Final Answer: x"], ["Final Answer: x"])
    run = AgentRun(client, CORRECTOR, "t", tools=tools())
    await run_events(run, [{"role": "user", "content": "私は学校を行きます"}])
    assert run.data["verified"] is False and "could not be verified" in run.data["handoff"]["reason"]


async def test_a_model_error_in_the_react_loop_fails_the_run():
    run = AgentRun(FakeClient(error=OllamaError("connection", "down")), CORRECTOR, "t", tools=tools())
    events = await run_events(run, [{"role": "user", "content": "x"}])
    assert kinds(events) == ["AgentRunStarted", "Error", "AgentRunFinished"] and run.error_kind == "connection"


async def test_a_react_context_too_big_is_an_invalid_output():
    spec = AgentSpec(name="corrector", model="m", num_ctx=8192, strategy="react")
    events = await run_events(AgentRun(FakeClient(), spec, "t"), [{"role": "user", "content": "x"}])
    assert events[1].kind == "invalid_output" and "800 tokens" in events[1].message


async def test_text_before_a_native_tool_call_is_kept_apart_from_the_answer():
    client = ToolClient(["Let me check.", ToolRequest("kanji", {"character": "行"})], ["It is 行."])
    run = AgentRun(client, SENSEI, "t", tools=tools())
    events = await run_events(run)
    assert run.text == "Let me check.\n\nIt is 行."
    assert "".join(event.text for event in events if isinstance(event, ev.Token)) == run.text


async def test_a_tool_call_on_the_last_prompted_round_is_never_shown():
    call = ['{"tool": "kanji", "args": {"character": "行"}}']
    client = ToolClient(call, call, native=False)
    run = AgentRun(client, SENSEI, "t", tools=tools(), max_tool_rounds=1)
    await run_events(run)
    assert run.text == "Sorry, I could not finish looking this up."


async def test_a_reply_starting_with_inline_code_streams_in_prompted_mode():
    client = ToolClient(["`", "行って", "` is the te-form."], native=False)
    run = AgentRun(client, SENSEI, "t", tools=tools())
    events = await run_events(run)
    assert kinds(events).count("Token") == 2 and run.text == "`行って` is the te-form."


async def test_a_capabilities_failure_falls_back_to_prompted_tools():
    class NoShow(ToolClient):
        def capabilities(self, model):
            raise OllamaError("bad_request", "no /api/show")

    client = NoShow(["Plain."])
    run = AgentRun(client, SENSEI, "t", tools=tools())
    await run_events(run)
    assert run.ok and "You have access" in client.calls[0][1][0]["content"]


async def test_the_repair_of_a_tool_agent_keeps_the_tool_exchange():
    spec = AgentSpec(name="x", model="m", num_ctx=4096, system_prompt="s",
                     output_schema={"type": "object", "required": ["a"]})
    client = ToolClient([ToolRequest("kanji", {"character": "行"})], ['{"b": 1}'], ['{"a": 1}'])
    run = AgentRun(client, spec, "t", tools=tools())
    await run_events(run)
    assert run.data == {"a": 1}
    assert any(message["role"] == "tool" for message in client.calls[-1][1])


async def test_an_unexpected_error_in_the_react_loop_fails_the_run_cleanly(monkeypatch):
    from sensai.orchestration import executor

    def explode(*args, **kwargs):
        raise IndexError("list index out of range")

    monkeypatch.setattr(executor, "run_react_loop", explode)
    events = await run_events(AgentRun(ToolClient(), CORRECTOR, "t", tools=tools()), [{"role": "user", "content": "x"}])
    assert events[1].kind == "invalid_output" and "IndexError" in events[1].message and events[-1].ok is False


async def test_an_unverified_correction_falls_back_to_the_first_check():
    client = ToolClient(["Final Answer: x"], ["Final Answer: x"], ["Final Answer: x"])
    run = AgentRun(client, CORRECTOR, "t", tools=tools())
    run.handoff = Handoff("checker", "corrector", "particle", {"correction": "私は学校に行きます"}, 1, "r0")
    await run_events(run, [{"role": "user", "content": "私は学校を行きます"}])
    assert run.data["correction"] == "私は学校に行きます" and run.data["verified"] is False
