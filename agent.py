"""Ivy, the voice assistant at Fernbrook Garden Centre: Dialog-RSN-1 hears and speaks, LangChain runs the tools."""
import argparse
import asyncio
import base64
import json
import os

import websockets
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function

load_dotenv()

from expert import ask_gardener  # noqa: E402  (reads EXPERT_MODEL from .env)
from tools import check_stock, get_opening_hours  # noqa: E402

URL = os.environ.get("DIALOG_REALTIME_URL", "wss://api.studio.poly.ai/v1/realtime")
TOOLS = [check_stock, get_opening_hours, ask_gardener]
SLOW_TOOLS = {"ask_gardener"}  # Takes seconds, so Ivy says she's checking while it runs.

INSTRUCTIONS = """You are Ivy, the voice assistant at Fernbrook Garden Centre.
Everything you say is spoken aloud. Keep replies to one or two short sentences, with no lists or symbols.
Use check_stock for anything about stock, prices or where a plant is, and get_opening_hours for hours.
For advice about plant problems, plant care or what to plant where, use ask_gardener.
Never guess a price, a stock level or an opening time.
When the call starts, greet the caller in one short sentence."""


def realtime_tool(tool: BaseTool) -> dict:
    """Describe a LangChain tool in the flat shape the Realtime API expects."""
    return {"type": "function", **convert_to_openai_function(tool)}


def session_update(tools: list[BaseTool]) -> dict:
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": INSTRUCTIONS,
            "tools": [realtime_tool(t) for t in tools],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": {"type": "server_vad"},
                },
            },
        },
    }


class VoiceAgent:
    def __init__(self, ws, tools: list[BaseTool], speaker=None):
        self.ws = ws
        self.tools = {t.name: t for t in tools}
        self.speaker = speaker
        self.config = {"configurable": {"thread_id": "voice"}}  # LangChain config; the thread is set in start()
        self.tool_tasks: set[asyncio.Task] = set()
        self.transcript: list[tuple[str, str]] = []
        self.active: set[str] = set()  # ids of responses in progress
        self.requested = 0  # response.create sent, response.created not seen yet
        self.idle = asyncio.Event()  # set when both are empty
        self.idle.set()

    async def send(self, event: dict):
        if event["type"] == "response.create":
            self.requested += 1
            self.idle.clear()
        await self.ws.send(json.dumps(event))

    async def start(self):
        created = json.loads(await self.ws.recv())
        self.config["configurable"]["thread_id"] = created["session"]["poly_session_id"]
        await self.send(session_update(list(self.tools.values())))
        updated = json.loads(await self.ws.recv())
        if updated["type"] != "session.updated":
            # A refused update applies nothing, so the agent would run with no prompt and no tools.
            raise RuntimeError(updated.get("error", {}).get("message", f"unexpected {updated['type']}"))

    async def invoke(self, call: dict) -> str:
        tool = self.tools.get(call["name"])
        try:
            args = json.loads(call["arguments"] or "{}")
            result = await tool.ainvoke(args, self.config) if tool else f"Error: no tool named {call['name']}"
        except Exception as e:
            # Tell the model what went wrong so it can say so, instead of leaving the caller in silence.
            result = f"Error: {e}"
        return result if isinstance(result, str) else json.dumps(result)

    async def reply(self, call: dict, output: str):
        await self.send({
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": call["call_id"], "output": output},
        })
        await self.send({"type": "response.create"})

    async def run_tool(self, call: dict):
        if call["name"] not in SLOW_TOOLS:
            await self.reply(call, await self.invoke(call))
            return
        # Answer at once so Ivy can say she's checking, then pass the answer on when it lands.
        await self.reply(call, "The head gardener is looking into it. Tell the caller you're checking, in a few words.")
        answer = await self.invoke(call)
        await self.idle.wait()
        await self.send({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": f"The head gardener says: {answer} Pass this on to the caller now."}],
            },
        })
        await self.send({"type": "response.create"})

    async def handle(self, event: dict):
        match event["type"]:
            case "response.output_audio.delta" if self.speaker:
                self.speaker.play(base64.b64decode(event["delta"]))
            case "input_audio_buffer.speech_started" if self.speaker:
                self.speaker.clear()  # The caller is talking over Ivy: stop playback now.
            case "conversation.item.input_audio_transcription.completed":
                self.say("You", event["transcript"])
            case "response.output_audio_transcript.done":
                self.say("Ivy", event["transcript"])
            case "response.function_call_arguments.done":
                self.say("Tool", f"{event['name']}({event['arguments']})")
                task = asyncio.create_task(self.run_tool(event))
                self.tool_tasks.add(task)
                task.add_done_callback(self.tool_tasks.discard)
            case "response.created":
                self.requested = max(0, self.requested - 1)
                self.active.add(event["response"]["id"])
                self.idle.clear()
            case "response.done":
                self.active.discard(event["response"]["id"])
                self.update_idle()
            case "error":
                if event["error"].get("code") == "conversation_already_has_active_response":
                    self.requested = max(0, self.requested - 1)  # That response.create was refused.
                    self.update_idle()
                self.say("Error", event["error"]["message"])

    def update_idle(self):
        if not self.active and not self.requested:
            self.idle.set()

    def say(self, who: str, text: str):
        self.transcript.append((who, text))
        print(f"{who + ':':6} {text}")

    async def listen(self, until_reply=False):
        """Handle events until the connection closes, or with until_reply, until Ivy finishes a spoken reply."""
        async for raw in self.ws:
            event = json.loads(raw)
            await self.handle(event)
            if until_reply and event["type"] == "response.done":
                outputs = [item["type"] for item in event["response"].get("output", [])]
                if "message" in outputs and not self.tool_tasks:
                    return


def connect():
    return websockets.connect(URL, additional_headers={"Authorization": f"Bearer {os.environ['POLY_API_KEY']}"})


async def ask(text: str, speaker=None) -> VoiceAgent:
    """One typed turn: send the text, run any tools, and return once Ivy has replied."""
    async with connect() as ws:
        agent = VoiceAgent(ws, TOOLS, speaker)
        await agent.start()
        await agent.send({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
        })
        await agent.send({"type": "response.create"})
        await agent.listen(until_reply=True)
        return agent


async def call():
    """A live call on your microphone and speakers. Ctrl+C to hang up."""
    from audio import Microphone, Speaker

    async with connect() as ws:
        with Microphone() as mic, Speaker() as speaker:
            agent = VoiceAgent(ws, TOOLS, speaker)
            await agent.start()
            await agent.send({"type": "response.create"})  # Ivy greets the caller.
            print("Connected. Talk to Ivy. Ctrl+C to hang up.")

            async def stream_mic():
                while True:
                    pcm = await mic.queue.get()
                    await agent.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})

            await asyncio.gather(stream_mic(), agent.listen())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--say", help="send one typed turn instead of opening the microphone")
    parser.add_argument("--mute", action="store_true", help="with --say, print the reply without playing it")
    args = parser.parse_args()
    if not args.say:
        try:
            asyncio.run(call())
        except KeyboardInterrupt:
            print("\nHung up.")
        return

    async def typed_turn():
        if args.mute:
            await ask(args.say)
            return
        from audio import Speaker
        with Speaker() as speaker:
            await ask(args.say, speaker)
            await speaker.drain()

    asyncio.run(typed_turn())


if __name__ == "__main__":
    main()
