"""A tiny stand-in for the Dialog-RSN-1 realtime endpoint, so the tests run with no key and no network.

It speaks just enough of the protocol for one turn: the first response.create gets a scripted tool call, and
the response.create after the tool's output gets a short spoken reply that quotes the output. Timing follows
the real server where it matters: the tool call's response.done arrives a little late, a tool argument marked
poly_spoken is spoken in the same response as the call, and a response.create that arrives while a line is
still playing waits until it ends.
"""
import asyncio
import base64
import json

import websockets


class FakeRealtime:
    def __init__(self, tool_name="check_stock", tool_args=None, refuse_update=False):
        self.tool_name = tool_name
        self.tool_args = tool_args if tool_args is not None else {"plant": "lavender"}
        self.refuse_update = refuse_update
        self.received: list[dict] = []
        self.spoken: list[str] = []
        self.headers = None
        self.url = None

    def sent(self, event_type: str) -> list[dict]:
        return [e for e in self.received if e["type"] == event_type]

    async def __aenter__(self):
        self.server = await websockets.serve(self.handler, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/v1/realtime"
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()

    async def handler(self, ws):
        self.headers = ws.request.headers
        send = lambda event: ws.send(json.dumps(event))  # noqa: E731
        await send({"type": "session.created", "session": {"poly_session_id": "conversation_test"}})
        state = {"output": None, "spoken_args": {}, "count": 0}
        playing = asyncio.Lock()  # one spoken line at a time; a response.create waits its turn

        async def speak(text):
            self.spoken.append(text)
            await send({"type": "response.output_audio.delta", "delta": base64.b64encode(b"\0" * 960).decode()})
            await asyncio.sleep(0.1)
            await send({"type": "response.output_audio_transcript.done", "transcript": text})

        async def tool_call():
            async with playing:
                await send({"type": "response.created", "response": {"id": "resp_tool"}})
                await send({
                    "type": "response.function_call_arguments.done",
                    "name": self.tool_name,
                    "arguments": json.dumps(self.tool_args),
                    "call_id": "call_1",
                })
                outputs = [{"type": "function_call"}]
                line = next((self.tool_args[k] for k in state["spoken_args"].get(self.tool_name, []) if k in self.tool_args), None)
                if line:
                    await speak(line)
                    outputs.append({"type": "message"})
                else:
                    await asyncio.sleep(0.05)
                await send({"type": "response.done", "response": {"id": "resp_tool", "status": "completed", "output": outputs}})

        async def reply():
            async with playing:
                state["count"] += 1
                response_id = f"resp_reply_{state['count']}"
                await send({"type": "response.created", "response": {"id": response_id}})
                await speak(f"From the tool: {state['output']}")
                await send({"type": "response.done", "response": {"id": response_id, "status": "completed", "output": [{"type": "message"}]}})

        async for raw in ws:
            event = json.loads(raw)
            self.received.append(event)
            match event["type"]:
                case "session.update" if self.refuse_update:
                    await send({"type": "error", "error": {"code": "unsupported_parameter", "message": "nope"}})
                case "session.update":
                    for tool in event["session"].get("tools", []):
                        props = tool["parameters"].get("properties", {})
                        state["spoken_args"][tool["name"]] = [k for k, v in props.items() if v.get("poly_spoken")]
                    await send({"type": "session.updated", "session": event["session"]})
                case "conversation.item.create" if event["item"]["type"] == "function_call_output":
                    state["output"] = event["item"]["output"]
                case "response.create" if state["output"] is None:
                    asyncio.create_task(tool_call())
                case "response.create":
                    asyncio.create_task(reply())
