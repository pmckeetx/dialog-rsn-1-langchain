"""The agent against a fake realtime server: what it sends, and what it does with what comes back."""
import json

import pytest
from langchain_core.messages import AIMessage

import agent
import expert
from fake_realtime import FakeRealtime


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setenv("POLY_API_KEY", "test-key")


async def test_a_tool_call_runs_the_langchain_tool_and_asks_for_a_reply(monkeypatch):
    async with FakeRealtime() as server:
        monkeypatch.setattr(agent, "URL", server.url)
        ivy = await agent.ask("Do you have lavender?")

    assert server.headers["Authorization"] == "Bearer test-key"
    [update] = server.sent("session.update")
    assert len(update["session"]["tools"]) == 3

    [call_output] = [e["item"] for e in server.sent("conversation.item.create") if e["item"]["type"] == "function_call_output"]
    assert call_output["call_id"] == "call_1"
    assert json.loads(call_output["output"])["in_stock"] == 14

    # The tool's output is followed by a response.create, or the model never speaks the answer.
    sent = [e["item"]["type"] if e["type"] == "conversation.item.create" else e["type"] for e in server.received]
    assert sent[sent.index("function_call_output") + 1] == "response.create"
    assert ivy.transcript[-1][0] == "Ivy"
    assert '"in_stock": 14' in ivy.transcript[-1][1]


async def test_an_unknown_tool_reports_an_error_to_the_model(monkeypatch):
    async with FakeRealtime(tool_name="book_delivery", tool_args={}) as server:
        monkeypatch.setattr(agent, "URL", server.url)
        await agent.ask("Can you deliver?")

    [call_output] = [e["item"] for e in server.sent("conversation.item.create") if e["item"]["type"] == "function_call_output"]
    assert call_output["output"] == "Error: no tool named book_delivery"


async def test_the_gardener_gets_one_thread_per_voice_session(monkeypatch):
    seen = {}

    class FakeGardener:
        async def ainvoke(self, state, config):
            seen["question"] = state["messages"][0][1]
            seen["thread_id"] = config["configurable"]["thread_id"]
            return {"messages": [AIMessage("Water it less.")]}

    monkeypatch.setattr(expert, "gardener", FakeGardener())
    async with FakeRealtime(tool_name="ask_gardener", tool_args={"question": "Why is my lavender yellow?"}) as server:
        monkeypatch.setattr(agent, "URL", server.url)
        ivy = await agent.ask("Why is my lavender yellow?")

    assert seen == {"question": "Why is my lavender yellow?", "thread_id": "conversation_test"}

    # The tool call is answered at once with a hold line, and the gardener's answer follows as a system message.
    items = [e["item"] for e in server.sent("conversation.item.create")]
    [call_output] = [i for i in items if i["type"] == "function_call_output"]
    assert "checking" in call_output["output"]
    [system] = [i for i in items if i.get("role") == "system"]
    assert "Water it less." in system["content"][0]["text"]
    assert len(server.sent("response.create")) == 3  # the question, the hold line, the answer
    assert "Water it less." in ivy.transcript[-1][1]


async def test_a_refused_session_update_stops_the_agent(monkeypatch):
    async with FakeRealtime(refuse_update=True) as server:
        monkeypatch.setattr(agent, "URL", server.url)
        with pytest.raises(RuntimeError, match="nope"):
            await agent.ask("Hello?")


async def test_barge_in_clears_queued_audio():
    class FakeSpeaker:
        def __init__(self):
            self.played, self.cleared = b"", 0

        def play(self, pcm):
            self.played += pcm

        def clear(self):
            self.cleared += 1

    speaker = FakeSpeaker()
    ivy = agent.VoiceAgent(ws=None, tools=agent.TOOLS, speaker=speaker)
    await ivy.handle({"type": "response.output_audio.delta", "delta": "AAAA"})
    await ivy.handle({"type": "input_audio_buffer.speech_started"})
    assert speaker.played == b"\0\0\0"
    assert speaker.cleared == 1
