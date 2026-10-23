from agent import TOOLS, realtime_tool, session_update
from expert import ask_gardener
from tools import check_stock, get_opening_hours, plant_care


def test_check_stock_matches_spoken_names():
    result = check_stock.invoke({"plant": "Lavenders"})
    assert result["found"] is True
    assert result["plant"] == "lavender"
    assert result["in_stock"] == 14


def test_check_stock_says_when_it_has_no_match():
    assert check_stock.invoke({"plant": "venus flytrap"}) == {"plant": "venus flytrap", "found": False}


def test_opening_hours_cover_the_week():
    result = get_opening_hours.invoke({})
    assert len(result["hours"]) == 7
    assert result["today"] in result["hours"]


def test_plant_care_falls_back_politely():
    assert "sharp drainage" in plant_care.invoke({"plant": "lavender"})
    assert plant_care.invoke({"plant": "cactus"}) == "No care notes for cactus."


def test_realtime_tool_is_flat_with_arg_descriptions():
    schema = realtime_tool(check_stock)
    assert schema["type"] == "function"
    assert schema["name"] == "check_stock"
    assert "function" not in schema  # Chat Completions nests it; the Realtime API doesn't.
    assert schema["parameters"]["required"] == ["plant"]
    assert "common name" in schema["parameters"]["properties"]["plant"]["description"]
    assert "Args:" not in schema["description"]  # parse_docstring moved it into the parameters.


def test_injected_config_stays_out_of_the_schema():
    assert list(realtime_tool(ask_gardener)["parameters"]["properties"]) == ["question"]


def test_session_update_sends_every_tool_and_the_input_format():
    session = session_update(TOOLS)["session"]
    assert [t["name"] for t in session["tools"]] == ["check_stock", "get_opening_hours", "ask_gardener"]
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["input"]["turn_detection"] == {"type": "server_vad"}
