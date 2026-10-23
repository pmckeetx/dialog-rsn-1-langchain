"""Real calls to Dialog-RSN-1. Run with POLYAI_LIVE=1 uv run pytest -m live. They check shape, not wording."""
import os

import pytest

import agent

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("POLYAI_LIVE") != "1", reason="set POLYAI_LIVE=1 to call the real API"),
]


def tools_called(ivy):
    return [text.split("(")[0] for who, text in ivy.transcript if who == "Tool"]


async def test_stock_question_uses_check_stock():
    ivy = await agent.ask("Do you have any climbing roses in stock?")
    assert "check_stock" in tools_called(ivy)
    assert ivy.transcript[-1][0] == "Ivy" and ivy.transcript[-1][1]


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="the gardener needs ANTHROPIC_API_KEY")
async def test_advice_question_goes_to_the_gardener():
    ivy = await agent.ask("Something is eating holes in my hosta leaves. What can I do?")
    assert "ask_gardener" in tools_called(ivy)
    assert ivy.transcript[-1][0] == "Ivy" and ivy.transcript[-1][1]
