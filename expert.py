import os

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from tools import check_stock, plant_care

EXPERT_MODEL = os.environ.get("EXPERT_MODEL", "anthropic:claude-haiku-5-5")

gardener = create_agent(
    model=EXPERT_MODEL,
    tools=[plant_care, check_stock],
    system_prompt=(
        "You are the head gardener at Fernbrook Garden Centre. Answer the question using the care notes "
        "and stock tools, and suggest a product only if we stock it. Your answer is read aloud on a call, "
        "so use at most three short sentences of plain speech: no lists, markdown or symbols."
    ),
    checkpointer=InMemorySaver(),
)


@tool(parse_docstring=True)
async def ask_gardener(question: str, config: RunnableConfig) -> str:
    """Ask the head gardener for advice: plant problems, what to plant where, or how to care for something.

    Args:
        question: The caller's question, with any details they gave about their garden.
    """
    # One LangGraph thread per voice session, so follow-up questions have context.
    result = await gardener.ainvoke({"messages": [("user", question)]}, config)
    return result["messages"][-1].text
