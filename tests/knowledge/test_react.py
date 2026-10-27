"""Tests for the ReAct parser and reasoning loop (Layer 4 - Feature A1)."""
from sensai.knowledge.react import (
    REACT_DEFAULT_NUM_CTX,
    parse_react_output,
    run_react_loop,
)
from sensai.knowledge.tools import registry


class ScriptedOllamaClient:
    """Fake client returning scripted model outputs while recording num_ctx."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.recorded_num_ctx: list[int] = []
        self.calls_count = 0

    def chat_stream(self, model: str, messages: list[dict], num_ctx: int = 800, temperature: float = 0.1):
        self.recorded_num_ctx.append(num_ctx)
        self.calls_count += 1
        reply = self._responses.pop(0) if self._responses else "Final Answer: Fallback."

        def _gen():
            yield reply
            return None

        return _gen()


def test_parse_valid_action():
    output = (
        "Thought: I need to check the reading of this kanji before answering.\n"
        'Action: {"tool": "kanji", "args": {"character": "鬱"}}'
    )
    result = parse_react_output(output)
    assert result["type"] == "action"
    assert result["thought"] == "I need to check the reading of this kanji before answering."
    assert result["action"]["tool"] == "kanji"
    assert result["action"]["args"]["character"] == "鬱"


def test_parse_final_answer_with_verification_and_rule():
    output = (
        "Verification: The dictionary confirms 行く takes に or へ.\n"
        "Rule: Destination particle rule (に/へ with motion verbs)\n"
        "Final Answer: 私は学校に行きます (Watashi wa gakkou ni ikimasu) — Use に instead of を."
    )
    result = parse_react_output(output)
    assert result["type"] == "final_answer"
    assert "Destination particle rule" in result["rule"]
    assert "私は学校に行きます" in result["content"]


def test_parse_malformed_json():
    output = (
        "Thought: I am missing a quote.\n"
        'Action: {"tool": "dictionary", "args": {"word: 食べる}}'
    )
    result = parse_react_output(output)
    assert result["type"] == "error"
    assert "Malformed JSON" in result["message"]


def test_parse_no_format():
    output = "I think 食べる means to eat, but I'm not using the strict format."
    result = parse_react_output(output)
    assert result["type"] == "error"


def test_react_trace_shows_think_tool_verification_answer():
    """A1-1: The trace shows the steps: think, tool call, verification, answer."""
    scripted_outputs = [
        (
            "Thought: Let's check the verb 行く in the dictionary to verify its particle.\n"
            'Action: {"tool": "dictionary", "args": {"word": "行く"}}'
        ),
        (
            "Verification: The dictionary tool confirms 行く takes に or へ for destinations.\n"
            "Rule: Motion verbs take particle に or へ, never を.\n"
            "Final Answer: 私は学校に行きます (Watashi wa gakkou ni ikimasu) — Replace を with に."
        ),
    ]
    client = ScriptedOllamaClient(scripted_outputs)
    traced_steps = []

    result = run_react_loop(
        client=client,
        model="qwen3.5:9b",
        sentence="私は学校を行きます",
        tool_registry=registry,
        trace_callback=traced_steps.append,
    )

    step_names = [s.step for s in result.steps]
    assert step_names == ["think", "tool_call", "verification", "answer"]
    assert len(traced_steps) == 4
    assert result.steps[1].tool == "dictionary"
    assert "to go" in (result.steps[1].observation or "")


def test_react_revises_answer_when_verification_contradicts_first_idea():
    """A1-1: When the verification contradicts the first idea, the final answer is revised."""
    scripted_outputs = [
        # Step 1: The model initially guesses the wrong regular conjugation 'いいて' (iite)
        (
            "Thought: Since 行く ends in く (ku), my first idea is that the te-form is いいて (iite). "
            "Let me verify with the conjugate tool.\n"
            'Action: {"tool": "conjugate", "args": {"verb": "行く", "form": "te"}}'
        ),
        # Step 2: Tool returns '行って'. The model notices the contradiction and revises its answer.
        (
            "Verification: The tool returned 行って (itte), which contradicts my first idea of いいて! "
            "行く is an irregular exception verb in the te-form.\n"
            "Rule: Irregular Godan verb exception — 行く conjugates to 行って (itte), not いいて.\n"
            "Final Answer: 学校に行って (Gakkou ni itte) — 行く has an irregular te-form (行って)."
        ),
    ]
    client = ScriptedOllamaClient(scripted_outputs)

    result = run_react_loop(
        client=client,
        model="qwen3.5:9b",
        sentence="学校にいいて",
        tool_registry=registry,
    )

    assert result.revised is True
    assert "行って" in result.answer
    assert "いいて" not in result.correction
    assert "Irregular Godan verb exception" in result.rule


def test_react_stops_after_max_steps_and_uses_800_token_context():
    """A1-1: The loop stops after a fixed number of steps and works on a context of about 800 tokens."""
    infinite_actions = [
        'Thought: Checking again.\nAction: {"tool": "dictionary", "args": {"word": "食べる"}}'
        for _ in range(10)
    ]
    client = ScriptedOllamaClient(infinite_actions)

    result = run_react_loop(
        client=client,
        model="qwen3.5:9b",
        sentence="ご飯を食べる",
        tool_registry=registry,
        max_steps=3,
        num_ctx=REACT_DEFAULT_NUM_CTX,
    )

    assert result.stopped_by_max_steps is True
    assert client.calls_count == 3
    assert all(ctx == 800 for ctx in client.recorded_num_ctx)
    assert result.steps[-1].step == "answer"


def test_react_includes_grammar_rule_and_expandable_trace():
    """A1-2: The answer names the grammar rule and reasoning steps can be expanded in the trace."""
    scripted_outputs = [
        (
            "Thought: Verify the past negative polite form of 食べる.\n"
            'Action: {"tool": "conjugate", "args": {"verb": "食べる", "form": "past_negative_polite"}}'
        ),
        (
            "Verification: Confirmed that the polite past negative is 食べませんでした.\n"
            "Rule: Ichidan polite past negative uses stem + ませんでした (-masen deshita).\n"
            "Final Answer: 食べませんでした (Tabemasen deshita) — Drop る and add ませんでした."
        ),
    ]
    client = ScriptedOllamaClient(scripted_outputs)

    result = run_react_loop(
        client=client,
        model="qwen3.5:9b",
        sentence="食べましたない",
        tool_registry=registry,
    )

    trace_payload = result.to_trace_dict()
    assert "ませんでした" in result.rule
    assert trace_payload["rule"] == result.rule
    assert len(trace_payload["steps"]) == 4
    assert trace_payload["steps"][1]["tool"] == "conjugate"
    assert "食べませんでした" in trace_payload["steps"][1]["observation"]