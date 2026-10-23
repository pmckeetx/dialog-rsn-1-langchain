"""A tiny stand-in for the Dialog-RSN-1 realtime endpoint, so the tests run with no key and no network.

It speaks just enough of the protocol for one turn: the first response.create gets a scripted tool call,
and later ones get a short spoken reply that quotes the tool's output or the latest system message. Timing
follows the real server closely enough to catch races: the tool call's response.done arrives a little late,
a spoken reply stays in progress for a moment, and a response.create during one is refused.
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
        self.errors: list[str] = []
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
        state = {"output": None, "speaking": False, "count": 0}

        async def finish_tool_call(response_id):
            await asyncio.sleep(0.05)
            await send({"type": "response.done", "response": {"id": response_id, "status": "completed", "output": [{"type": "function_call"}]}})

        async def finish_reply(response_id, text):
            await asyncio.sleep(0.1)
            await send({"type": "response.output_audio_transcript.done", "transcript": text})
            await send({"type": "response.done", "response": {"id": response_id, "status": "completed", "output": [{"type": "message"}]}})
            state["speaking"] = False

        async for raw in ws:
            event = json.loads(raw)
            self.received.append(event)
            match event["type"]:
                case "session.update" if self.refuse_update:
                    await send({"type": "error", "error": {"code": "unsupported_parameter", "message": "nope"}})
                case "session.update":
                    await send({"type": "session.updated", "session": event["session"]})
                case "conversation.item.create" if event["item"]["type"] == "function_call_output":
                    state["output"] = event["item"]["output"]
                case "conversation.item.create" if event["item"].get("role") == "system":
                    state["output"] = event["item"]["content"][0]["text"]
                case "response.create" if state["speaking"]:
                    self.errors.append("conversation_already_has_active_response")
                    await send({"type": "error", "error": {
                        "code": "conversation_already_has_active_response",
                        "message": "A response is already in progress.",
                    }})
                case "response.create" if state["output"] is None:
                    await send({"type": "response.created", "response": {"id": "resp_tool"}})
                    await send({
                        "type": "response.function_call_arguments.done",
                        "name": self.tool_name,
                        "arguments": json.dumps(self.tool_args),
                        "call_id": "call_1",
                    })
                    asyncio.create_task(finish_tool_call("resp_tool"))
                case "response.create":
                    state["speaking"] = True
                    state["count"] += 1
                    response_id = f"resp_reply_{state['count']}"
                    await send({"type": "response.created", "response": {"id": response_id}})
                    await send({"type": "response.output_audio.delta", "delta": base64.b64encode(b"\0" * 960).decode()})
                    asyncio.create_task(finish_reply(response_id, f"From the tool: {state['output']}"))
