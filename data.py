"""Fernbrook Garden Centre's stock, hours and care notes. Fictional, and kept in memory for the demo."""

INVENTORY = {
    "lavender": {"in_stock": 14, "price_usd": 8.50, "where": "the herb tables by the entrance"},
    "climbing rose": {"in_stock": 6, "price_usd": 24.00, "where": "the rose arbor"},
    "tomato plant": {"in_stock": 0, "price_usd": 4.50, "where": "the vegetable benches", "restock": "Saturday"},
    "japanese maple": {"in_stock": 3, "price_usd": 89.00, "where": "the tree yard"},
    "hosta": {"in_stock": 22, "price_usd": 12.00, "where": "the shade house"},
    "compost": {"in_stock": 40, "price_usd": 9.00, "where": "the covered yard, in 40 liter bags"},
    "copper tape": {"in_stock": 18, "price_usd": 7.00, "where": "the garden care aisle"},
    "wool slug pellets": {"in_stock": 9, "price_usd": 11.50, "where": "the garden care aisle"},
}

CARE = {
    "lavender": "Full sun and sharp drainage. Water only when the top inch of soil is dry; yellowing leaves "
                "usually mean too much water. Prune by a third after flowering, never into old wood.",
    "climbing rose": "Six hours of sun. Water deeply twice a week in summer. Tie canes sideways to get more "
                     "flowers. Feed in spring and again after the first flush.",
    "tomato plant": "Full sun, steady watering at the base, and a stake or cage. Pinch out side shoots on "
                    "vining types. Plant out after the last frost.",
    "japanese maple": "Dappled shade and shelter from wind. Keep the roots cool and moist; mulch in summer. "
                      "Leaf scorch at the edges means too much sun or wind.",
    "hosta": "Shade to part shade, moist rich soil. Slugs are the main pest: ring pots with copper tape, "
             "or scatter wool slug pellets around the crown.",
}

HOURS = {
    "Monday": "9 am to 6 pm",
    "Tuesday": "9 am to 6 pm",
    "Wednesday": "9 am to 6 pm",
    "Thursday": "9 am to 8 pm",
    "Friday": "9 am to 8 pm",
    "Saturday": "8 am to 6 pm",
    "Sunday": "10 am to 4 pm",
}


def find(table: dict, name: str):
    """Match a spoken name to a key by its words: "Lavenders", "lavender plant" and "rose" all find one."""
    words = {w.removesuffix("s") for w in name.lower().replace("-", " ").split()}
    for key in table:
        key_words = {w.removesuffix("s") for w in key.split()}
        if key_words <= words or words <= key_words:
            return key, table[key]
    return None, None
