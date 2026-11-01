# Tests for sliding-window compression: the summary provider, the trigger and the summarizer call.

from dataclasses import replace

from sensai.llm.ollama import OllamaError
from sensai.orchestration.context import ContextProvider, build_context
from sensai.orchestration.session import Session
from sensai.orchestration.summary import SummaryProvider, compress, summarize, summary_budget
from tests.conftest import memory_repository
from tests.fakes import FakeClient, ScriptedClient, make_config


def turns(count, size=400):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"{i}" * size} for i in range(count)]


def make_session(client, **summary):
    config = make_config()
    config["memory"]["summary"].update(summary)
    return Session(config, client=client, repo=memory_repository())


def tokens_of(messages):
    return sum(len(m["content"]) for m in messages) // 4


def test_provider_renders_the_summary_and_hides_an_empty_one():
    provider = SummaryProvider("studied particles")
    assert isinstance(provider, ContextProvider)
    assert provider.render(100) == "Summary of the earlier conversation: studied particles"
    assert SummaryProvider("").render(100) is None


def test_summarize_sends_previous_summary_and_new_turns():
    client = ScriptedClient("new summary ")
    session = make_session(client)
    text = summarize(session, session.spec, turns(2, 3), "old", 200)
    assert text == "new summary"
    model, messages, options = client.calls[0]
    assert model == session.spec.model
    assert "old" in messages[1]["content"] and "user: 000" in messages[1]["content"]
    assert options["num_ctx"] == session.spec.num_ctx
    assert options["num_predict"] == 200


def test_no_compression_below_the_trigger():
    client = ScriptedClient("S")
    session = make_session(client)
    session.history = turns(10, 10)
    compress(session, session.spec, 10_000, tokens_of)
    assert client.calls == [] and session.summary == "" and session.summarized == 0


def test_compression_folds_older_turns_and_keeps_the_recent_ones():
    client = ScriptedClient("S1", "S2")
    session = make_session(client, keep_recent=4)
    session.history = turns(10)
    compress(session, session.spec, 1000, tokens_of)
    assert (session.summary, session.summarized) == ("S1", 6)
    session.history += turns(6)
    compress(session, session.spec, 1000, tokens_of)
    assert (session.summary, session.summarized) == ("S2", 12)
    assert "S1" in client.calls[1][1][1]["content"]


def test_disabled_or_too_short_history_is_left_alone():
    client = ScriptedClient("S")
    off = make_session(client, enabled=False)
    off.history = turns(10)
    compress(off, off.spec, 100, tokens_of)
    short = make_session(client, keep_recent=6)
    short.history = turns(6)
    compress(short, short.spec, 100, tokens_of)
    assert client.calls == []


def test_a_failing_summarizer_keeps_the_plain_window():
    session = make_session(FakeClient(error=OllamaError("connection", "down")))
    session.history = turns(10)
    compress(session, session.spec, 1000, tokens_of)
    assert session.summary == "" and session.summarized == 0


def test_build_context_puts_the_summary_in_the_prompt_and_drops_folded_turns():
    session = make_session(ScriptedClient("learner knows は"), keep_recent=2)
    session.history = turns(8, 1500)
    messages = build_context(session, replace(session.spec, num_ctx=2048))
    assert "Summary of the earlier conversation: learner knows は" in messages[0]["content"]
    assert len(messages) - 1 <= 2 and session.summarized == 6


def test_reset_clears_the_summary():
    session = make_session(FakeClient())
    session.summary, session.summarized = "S", 4
    session.reset()
    assert (session.summary, session.summarized) == ("", 0)


def test_defaults_are_not_shared_between_configs():
    first, second = make_config(), make_config()
    first["memory"]["summary"]["enabled"] = False
    assert second["memory"]["summary"]["enabled"] is True


def test_compress_reports_what_was_folded():
    session = make_session(ScriptedClient("S1"), keep_recent=4)
    session.history = turns(10)
    result = compress(session, session.spec, 1000, tokens_of)
    assert (result.messages, result.tokens_before) == (6, tokens_of(session.history[:6]))
    assert result.tokens_after == tokens_of([{"role": "system", "content": "S1"}])


def test_compress_reports_nothing_below_the_trigger():
    session = make_session(ScriptedClient("S1"))
    session.history = turns(2, 10)
    assert compress(session, session.spec, 10_000, tokens_of) is None


def test_a_summary_longer_than_the_turns_it_replaces_is_dropped():
    session = make_session(ScriptedClient("word " * 2000), keep_recent=4)
    session.history = turns(10)
    assert compress(session, session.spec, 1000, tokens_of) is None
    assert (session.summary, session.summarized) == ("", 0)


def test_the_summary_budget_is_half_of_what_is_folded_within_its_share():
    assert summary_budget(262, 5783) == 131
    assert summary_budget(20_000, 5783) == int(5783 * 0.25)
    assert summary_budget(40, 5783) == 64


def test_the_turns_are_sent_as_a_transcript_to_describe_not_to_answer():
    client = ScriptedClient("notes")
    session = make_session(client)
    summarize(session, session.spec, turns(2, 3), "", 100)
    system, request = client.calls[0][1]
    assert "never continue the conversation" in system["content"]
    assert "<transcript>" in request["content"] and request["content"].rstrip().endswith("adds.")
