"""Offline tests: A2A tools are described by the Agent Card, and the EO prompt names no tools.

Run: python -m evals.test_a2a_client
"""

from pathlib import Path
from types import SimpleNamespace

import yaml

from service.a2a_client import tools_from_card


def card(*skills):
    return SimpleNamespace(skills=[SimpleNamespace(id=i, description=d) for i, d in skills])


def test_tool_descriptions_come_from_the_card():
    tools = tools_from_card(card(("embed_scene", "CARD TEXT: embed a scene"), ("compare_embeddings", "CARD TEXT: compare")),
                            "http://x")
    assert {t.name: t.description for t in tools} == {"embed_scene": "CARD TEXT: embed a scene",
                                                      "compare_embeddings": "CARD TEXT: compare"}
    assert set(tools[0].args_schema.model_fields) == {"collection", "item_id", "patch_size"}


def test_only_advertised_skills_become_tools():
    assert [t.name for t in tools_from_card(card(("rank_similar", "d")), "http://x")] == ["rank_similar"]
    assert tools_from_card(card(), "http://x") == []   # card with no skills: no tools, nothing for the prompt to mention


def test_skill_without_a_client_schema_is_skipped_not_guessed():
    got = tools_from_card(card(("fly_to_moon", "d"), ("embed_scene", "d")), "http://x")
    assert [t.name for t in got] == ["embed_scene"]


def test_the_real_agent_card_is_the_source_of_truth():
    from terramind_agent.server import SKILL_CARDS   # needs a2a-sdk only, not torch
    tools = tools_from_card(SimpleNamespace(skills=SKILL_CARDS), "http://x")
    assert [t.name for t in tools] == ["embed_scene", "compare_embeddings", "rank_similar"]
    by = {t.name: t.description for t in tools}
    assert "same_footprint" in by["compare_embeddings"] and "reading_guide" in by["compare_embeddings"]
    assert "only when the user asks" in by["embed_scene"]    # the usage rule that used to live in the prompt


def test_eo_prompt_hardcodes_no_tool_it_may_not_have():
    prompt = yaml.safe_load((Path(__file__).resolve().parent.parent / "service/eo_agent/prompts.yaml").read_text())["system"]
    for name in ("embed_scene", "compare_embeddings", "rank_similar", "geocode_location", "list_stac_collections",
                 "get_stac_item", "TerraMind"):
        assert name not in prompt, f"prompt names {name}: tool guidance belongs in the tool's own description"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} passed")
