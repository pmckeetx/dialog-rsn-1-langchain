from datetime import date

from langchain_core.tools import tool

from data import CARE, HOURS, INVENTORY, find


@tool(parse_docstring=True)
def check_stock(plant: str) -> dict:
    """Check whether the garden centre has a plant in stock, its price and where to find it.

    Args:
        plant: The plant's common name, such as "lavender" or "climbing rose".
    """
    name, item = find(INVENTORY, plant)
    if item is None:
        return {"plant": plant, "found": False}
    return {"plant": name, "found": True, **item}


@tool
def get_opening_hours() -> dict:
    """Get the garden centre's opening hours for each day of the week, and today's day."""
    return {"today": date.today().strftime("%A"), "hours": HOURS}


@tool(parse_docstring=True)
def plant_care(plant: str) -> str:
    """Look up the garden centre's care notes for a plant.

    Args:
        plant: The plant's common name.
    """
    name, notes = find(CARE, plant)
    return notes or f"No care notes for {plant}."
