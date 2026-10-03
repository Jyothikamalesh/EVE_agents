"""Offline tests for text-format tool-call parsing. Run: python -m evals.test_text_tool_calls"""

from agents.graphs.utils import parse_text_tool_calls

SEEN = "<tool_use>\n<tool_name>geocode_location</tool_name>\n<parameter=query>Paris, France</parameter>\n</function>\n</tool_call>"


def test_tool_use_xml_as_seen_from_qwen():
    (c,) = parse_text_tool_calls(SEEN)
    assert c["name"] == "geocode_location" and c["args"] == {"query": "Paris, France"} and c["id"]


def test_native_function_xml_with_typed_values():
    text = ('<tool_call><function=get_weather><parameter=lat>48.85</parameter><parameter=lon>2.35</parameter>'
            '<parameter=start_date>2023-07-01</parameter></function></tool_call>')
    (c,) = parse_text_tool_calls(text)
    assert c["name"] == "get_weather"
    assert c["args"] == {"lat": 48.85, "lon": 2.35, "start_date": "2023-07-01"}


def test_list_and_multiple_calls():
    text = (SEEN + "\n<tool_call><function=search_stac_items><parameter=bbox>[2.2, 48.8, 2.4, 48.9]</parameter>"
            "<parameter=limit>5</parameter></function></tool_call>")
    a, b = parse_text_tool_calls(text)
    assert a["name"] == "geocode_location" and b["name"] == "search_stac_items"
    assert b["args"] == {"bbox": [2.2, 48.8, 2.4, 48.9], "limit": 5}
    assert a["id"] != b["id"]


def test_plain_text_and_existing_formats_unchanged():
    assert parse_text_tool_calls("Hello, here is your answer: 3 scenes.") == []
    (c,) = parse_text_tool_calls('[TOOL_CALLS]geocode_location{"query": "Rome"}')
    assert c["name"] == "geocode_location" and c["args"] == {"query": "Rome"}
    (d,) = parse_text_tool_calls('[TOOL_CALLS][{"name": "get_weather", "arguments": {"lat": 1}}]')
    assert d["name"] == "get_weather" and d["args"] == {"lat": 1}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} passed")
