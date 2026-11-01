# Tests for knowledge in a full turn: tools for the sensei, RAG passages and citations, the checker ->
# corrector -> sensei chain, the corrector's errors in the ledger, and the session's Knowledge object.

import json
import sys

from sensai.core import events as ev
from sensai.core.agent_spec import spec_for
from sensai.knowledge.retrieval import Retrieval
from sensai.knowledge.rag import DocumentChunk
from sensai.knowledge.toolbox import Knowledge
from sensai.llm.ollama import ChatStats, OllamaError, ToolRequest
from sensai.memory.ledger import Ledger
from sensai.orchestration.session import Session
from sensai.orchestration.turn import run_turn
from tests.fakes import FakeClient, make_config

CHUNK = DocumentChunk("particles.md:1", "に marks the destination.", "particles.md", "Particle に", "N5", 1.0)
CITATION = "[particles.md § Particle に]"
CHECKED = json.dumps({"sentence": "私は学校を行きます", "is_correct": False, "correction": "私は学校に行きます",
                      "error_type": "particle", "rule": "に for destinations",
                      "handoff": {"to": "corrector", "reason": "verify"}}, ensure_ascii=False)
REACT = ['Thought: を is wrong.\nAction: {"tool": "dictionary", "args": {"word": "行く"}}',
         "Verification: confirmed.\nRevised: no\nRule: 行く takes に.\nCorrection: 私は学校に行きます\n"
         "Final Answer: 私は学校に行きます — use に."]


class Client(FakeClient):

    def __init__(self, *replies, native=True):
        super().__init__(stats=ChatStats(eval_count=1))
        self.replies = list(replies)
        self.native = native

    def capabilities(self, model):
        return {"completion", "tools"} if self.native else {"completion"}

    def chat_stream(self, model, messages, **options):
        self.calls.append((model, [dict(message) for message in messages], options))
        reply = self.replies.pop(0)
        yield from (reply if isinstance(reply, list) else [reply])
        return self.stats


class FixedRetriever:

    def __init__(self, kept, error=None):
        self.kept = kept
        self.error = error
        self.asked = []

    def retrieve(self, question, profile_text=None, model=None, num_ctx=4096):
        self.asked.append((question, profile_text, model))
        if self.error:
            raise self.error
        return Retrieval(question, "N5", "embeddings+model", list(self.kept))


def knowledge_session(client, retriever=None, **sections):
    config = make_config(tools={"enabled": True, "max_rounds": 3}, rag={**make_config()["rag"], "enabled": True},
                         **sections)
    session = Session(config, client=client)
    session.knowledge.retriever = retriever or FixedRetriever([CHUNK])
    return session


async def collect(session, text):
    return [event async for event in run_turn(session, text)]


async def test_the_sensei_gets_the_passages_and_the_answer_cites_them():
    session = knowledge_session(Client(["Use に ", CITATION]))
    session.repo.save_profile("default", "Level: N5")
    session.profile = session.repo.profile_system_message("default")
    events = await collect(session, "how do I say I go to school?")
    retrieved = next(event for event in events if isinstance(event, ev.Retrieved))
    assert retrieved.kept[0]["section"] == "Particle に" and retrieved.mode == "embeddings+model"
    system = session.client.calls[0][1][0]["content"]
    assert "Reference material" in system and "に marks the destination." in system
    assert session.knowledge.retriever.asked == [("how do I say I go to school?", "Learner profile:\nLevel: N5",
                                                  "qwen3.5:9b")]
    assert session.history[-1]["content"] == f"Use に {CITATION}"


async def test_an_answer_that_cites_nothing_gets_a_sources_line():
    session = knowledge_session(Client(["Use に."]))
    events = await collect(session, "how do I say I go to school?")
    tokens = [event.text for event in events if isinstance(event, ev.Token)]
    assert tokens[-1] == f"\n\nSources: {CITATION}"
    assert session.history[-1]["content"].endswith(CITATION)


async def test_no_passage_means_no_context_block_and_no_citation():
    session = knowledge_session(Client(["猫 (neko) — cat."]), FixedRetriever([]))
    await collect(session, "how do I say cat?")
    assert "Reference material" not in session.client.calls[0][1][0]["content"]
    assert session.history[-1]["content"] == "猫 (neko) — cat."


async def test_a_failing_retrieval_does_not_stop_the_turn():
    session = knowledge_session(Client(["Fine."]), FixedRetriever([], error=OllamaError("connection", "x")))
    events = await collect(session, "hello?")
    assert not any(isinstance(event, ev.Retrieved) for event in events)
    assert session.history[-1]["content"] == "Fine."


async def test_rag_off_retrieves_nothing():
    session = Session(make_config(tools={"enabled": True, "max_rounds": 3}), client=Client(["Hi."]))
    assert session.knowledge.retriever is None and session.knowledge.retrieve("q", None, "m", 8192) is None
    await collect(session, "hello")


async def test_the_sensei_calls_its_tools_during_a_turn():
    client = Client([ToolRequest("dictionary", {"word": "食べる"})], ["食べる (taberu) — to eat."])
    session = knowledge_session(client, FixedRetriever([]))
    events = await collect(session, "what does 食べる mean?")
    call = next(event for event in events if isinstance(event, ev.ToolCall))
    assert call.name == "dictionary" and session.history[-1]["content"] == "食べる (taberu) — to eat."
    names = [tool["function"]["name"] for tool in client.calls[0][2]["tools"]]
    assert names == ["dictionary", "conjugate", "kanji"]


async def test_a_wrong_sentence_goes_checker_corrector_sensei_and_is_recorded():
    client = Client(CHECKED, *REACT, ["Use に (ni) with 行く."])
    session = knowledge_session(client, FixedRetriever([]), agents={"default": "checker"})
    events = await collect(session, "私は学校を行きます")
    agents = [event.agent for event in events if isinstance(event, ev.AgentRunStarted)]
    assert agents == ["checker", "corrector", "sensei"]
    handoffs = [event.handoff for event in events if isinstance(event, ev.HandoffEvent)]
    assert handoffs[1].payload["verified"] is True and handoffs[1].payload["first_check"]["error_type"] == "particle"
    errors = Ledger(session.repo.conn).list("errors")
    assert [(row["sentence"], row["correction"], row["error_type"], row["session_id"]) for row in errors] == [
        ("私は学校を行きます", "私は学校に行きます", "particle", session.id)]


async def test_writing_the_correction_resolves_the_recorded_error():
    session = knowledge_session(Client(["Good."], ["Good."]), FixedRetriever([]))
    ledger = Ledger(session.repo.conn)
    ledger.create("corrector", "errors", sentence="私は学校を行きます", correction="私は学校に行きます",
                  error_type="particle")
    await collect(session, "私は学校を行きますか?")
    assert ledger.list("errors", resolved=1) == []
    await collect(session, "私は学校に行きます")
    assert len(ledger.list("errors", resolved=1)) == 1


async def test_an_unverified_correction_is_not_recorded():
    client = Client(CHECKED, "Final Answer: x", "Final Answer: x", "Final Answer: x", ["Explained."])
    session = knowledge_session(client, FixedRetriever([]), agents={"default": "checker"})
    await collect(session, "私は学校を行きます")
    assert Ledger(session.repo.conn).list("errors") == []


def test_tools_for_gives_each_agent_its_own_tools():
    knowledge = Knowledge(make_config(tools={"enabled": True, "max_rounds": 3}), FakeClient())
    assert knowledge.tools_for(spec_for("sensei", knowledge.config)).names() == ["dictionary", "conjugate", "kanji"]
    assert knowledge.tools_for(spec_for("checker", knowledge.config)) is None
    off = Knowledge(make_config(), FakeClient())
    assert off.tools_for(spec_for("sensei", off.config)) is None


def test_a_disabled_tool_is_left_out_for_every_agent():
    knowledge = Knowledge(make_config(tools={"enabled": True, "disabled": ["conjugate"]}), FakeClient())
    assert knowledge.tools_for(spec_for("sensei", knowledge.config)).names() == ["dictionary", "kanji"]
    assert knowledge.tools_for(spec_for("corrector", knowledge.config)).names() == ["dictionary", "kanji"]


def test_disabling_every_tool_of_an_agent_leaves_it_none():
    disabled = ["dictionary", "conjugate", "kanji"]
    knowledge = Knowledge(make_config(tools={"enabled": True, "disabled": disabled}), FakeClient())
    assert knowledge.tools_for(spec_for("corrector", knowledge.config)) is None


def own_server(enabled=True):
    return {"name": "sensai", "command": sys.executable, "args": ["-m", "sensai.knowledge.mcp_server"],
            "timeout_s": 10, "enabled": enabled}


def test_mcp_servers_start_once_and_their_tools_are_named_by_server():
    config = make_config(tools={"enabled": True, "max_rounds": 3},
                         mcp={"enabled": True, "servers": [own_server(), own_server(False),
                                                           {"name": "ghost", "command": "no-such-cmd-xyz"}]})
    knowledge = Knowledge(config, FakeClient())
    sensei = spec_for("sensei", config)
    try:
        chosen = knowledge.tools_for(type(sensei)(**{**sensei.__dict__, "tools": ("mcp:sensai", "mcp:ghost")}))
        assert chosen.names() == ["sensai_dictionary", "sensai_conjugate", "sensai_kanji"]
        assert "ghost" in knowledge.unavailable and len(knowledge.servers) == 1
        knowledge.start()
        assert len(knowledge.servers) == 1
    finally:
        knowledge.close()
    assert knowledge.servers == []


def test_mcp_off_starts_nothing():
    config = make_config(tools={"enabled": True, "max_rounds": 3}, mcp={"enabled": False, "servers": [own_server()]})
    knowledge = Knowledge(config, FakeClient())
    knowledge.start()
    assert knowledge.servers == [] and knowledge.unavailable == {}


async def test_a_mistake_is_recorded_once_and_with_the_checked_sentence():
    session = knowledge_session(Client(CHECKED, *REACT, ["Explained."], CHECKED, *REACT, ["Again."]),
                                FixedRetriever([]), agents={"default": "checker"})
    await collect(session, "私は学校を行きます")
    await collect(session, "私は学校を行きます")
    errors = Ledger(session.repo.conn).list("errors")
    assert len(errors) == 1 and errors[0]["sentence"] == "私は学校を行きます"


async def test_no_mistake_is_recorded_when_the_turn_fails_after_the_corrector():
    class FailingSensei(Client):
        def chat_stream(self, model, messages, **options):
            if len(self.calls) == 3:
                self.calls.append(None)
                raise OllamaError("stream", "broken")
            return super().chat_stream(model, messages, **options)

    session = knowledge_session(FailingSensei(CHECKED, *REACT), FixedRetriever([]), agents={"default": "checker"})
    await collect(session, "私は学校を行きます")
    assert Ledger(session.repo.conn).list("errors") == [] and session.history == []


async def test_retrieval_runs_once_per_turn_even_with_a_fallback():
    retriever = FixedRetriever([])
    session = knowledge_session(Client(CHECKED, "not react", "not react", "not react", ["Answer."]), retriever,
                                agents={"default": "checker"})
    await collect(session, "私は学校を行きます")
    assert len(retriever.asked) == 1


async def test_a_hybrid_retrieval_adds_no_sources_line():
    hybrid = FixedRetriever([CHUNK])
    hybrid.retrieve = lambda question, profile=None, model=None, num_ctx=4096: Retrieval(question, "N5", "embeddings+hybrid", [CHUNK])
    session = knowledge_session(Client(["Use に."]), hybrid)
    await collect(session, "how?")
    assert session.history[-1]["content"] == "Use に."


async def test_a_slow_retrieval_is_given_up(monkeypatch):
    import time
    from sensai.orchestration import orchestrator
    monkeypatch.setattr(orchestrator, "RETRIEVAL_TIMEOUT_S", 0.05)
    slow = FixedRetriever([CHUNK])
    slow.retrieve = lambda *args: time.sleep(0.3) or Retrieval("q", None, "x", [CHUNK])
    session = knowledge_session(Client(["Fine."]), slow)
    events = await collect(session, "how?")
    assert not any(isinstance(event, ev.Retrieved) for event in events) and session.history[-1]["content"] == "Fine."


async def test_the_corrector_never_triggers_the_history_summary():
    session = knowledge_session(Client(CHECKED, *REACT, ["Explained."]), FixedRetriever([]),
                                agents={"default": "checker"},
                                memory={**make_config()["memory"], "summary": {"enabled": True, "trigger": 0.0,
                                                                               "keep_recent": 0}})
    session.history = [{"role": "user", "content": "old " * 50}, {"role": "assistant", "content": "reply " * 50}]
    await collect(session, "私は学校を行きます")
    assert not any(call and call[2].get("num_ctx") == 1024 and "compress" in call[1][0]["content"]
                   for call in session.client.calls)


async def test_the_sensei_searches_with_the_correction_it_was_handed():
    retriever = FixedRetriever([])
    session = knowledge_session(Client(CHECKED, *REACT, ["Use に."]), retriever, agents={"default": "checker"})
    await collect(session, "私は学校を行きます")
    question = retriever.asked[0][0]
    assert question.startswith("私は学校を行きます") and "Correction: 私は学校に行きます" in question
    assert "Rule: 行く takes に." in question
