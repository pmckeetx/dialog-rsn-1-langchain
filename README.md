# Voice agent with LangChain and Dialog-RSN-1

Ivy answers the phone at Fernbrook Garden Centre, a fictional garden shop. You talk to her through your
microphone. PolyAI's [Dialog-RSN-1](https://docs.poly.ai/dialog-rsn-1) hears you, decides when you've
finished, and speaks the reply. LangChain supplies the tools Ivy calls, plus a LangChain agent (the "head
gardener") she asks when a question needs real advice.

Write-up: [How to build a voice agent using LangChain and PolyAI's realtime voice model](https://www.petermckee.dev/blog/build-a-voice-agent-using-langchain-and-polyai)

## What's here

| File | What it does |
| --- | --- |
| `agent.py` | The voice agent: turns LangChain tools into realtime tools, opens the session, runs tool calls |
| `tools.py` | LangChain `@tool` functions: stock, opening hours, care notes |
| `expert.py` | The head gardener: a LangChain `create_agent` that Ivy calls as one tool |
| `audio.py` | Microphone and speaker at 24 kHz, with playback cleared on barge-in |
| `data.py` | The shop's fictional stock, hours and care notes |
| `tests/` | Offline tests against a fake realtime server, and live tests behind `POLYAI_LIVE=1` |

## Run it

You need Python 3.13 and [uv](https://docs.astral.sh/uv/), a Dialog-RSN-1 key from
[studio.poly.ai](https://studio.poly.ai) (API Keys tab), and an Anthropic key for the head gardener.
Wear headphones for the microphone mode, or Ivy hears herself and stops talking.

```bash
cp .env.example .env    # then fill in POLY_API_KEY and ANTHROPIC_API_KEY
uv sync
uv run agent.py         # talk to Ivy; Ctrl+C to hang up
uv run agent.py --say "Do you have lavender in stock?"   # one typed turn, spoken reply
```

A `POLY_API_KEY` already exported in your shell wins over `.env`. If you get HTTP 401, check `echo $POLY_API_KEY`.

## Test it

```bash
uv run pytest                          # offline: no keys, no network
POLYAI_LIVE=1 uv run pytest -m live    # against the real API
```

## License

Apache 2.0
