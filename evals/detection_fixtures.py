"""Labelled fixtures for the verifier detection benchmark (see ``evals/SCENARIOS.md``).

Each fixture is one agent reply plus the tool outputs it was produced from, and a label:

* ``supported``     the reply is faithful to the tool outputs (a correct reply)
* ``hallucinated``  the reply contains a wrong or unsupported claim (a corrupted reply)

``eval_detection.py`` runs the live verifier's decision on every item and reports precision, recall
and F1 per scenario and per error type. Positive class = hallucinated; "predicted positive" = the
verifier flagged the reply.

The data is synthetic and deliberately NOT the Hyderabad/Paris data the checker was developed
against (Lisbon scenes and weather), so the scores are not measured on data it was tuned on.
Corrupted replies are generated from correct ones by one controlled edit each.

``EXPECT`` is a prediction written BEFORE the first run: for each error type, does the checker
catch it ("detect") or is it a known weakness ("miss")? A difference between prediction and result
is a finding, not something to tune away.
"""

from __future__ import annotations

from dataclasses import dataclass

from terramind_agent.skills import READING_GUIDE, REFERENCE_POINTS

TOOLS = ["geocode_location", "list_stac_collections", "search_stac_items", "get_stac_item", "get_weather",
         "embed_scene", "compare_embeddings", "rank_similar"]

WEATHER_DESCRIPTION = (
    "Get daily weather for a point: historical, recent, or forecast. With no dates, returns today plus a 7-day "
    "forecast. Historical requests route to the long-range archive (back to 1940); recent/future requests "
    "route to the forecast API (up to 16 days ahead)."
)


@dataclass
class Item:
    scenario: str
    error_type: str          # "" for supported replies
    label: str               # "supported" | "hallucinated"
    reply: str
    note: str = ""
    called: tuple = ("search_stac_items",)
    outputs: tuple = ()
    users: tuple = ("What Sentinel-2 imagery is there?",)
    tool_texts: tuple = ()


# ── scenario catalogue: what each one is and why it was chosen ──────────────────────────────────────
SCENARIOS = {
    "S1": ("A correct value attached to the wrong scene, or an invented value",
           "The most common way an imagery answer misleads: every number looks plausible, and a wrong cloud cover "
           "decides which scene someone downloads."),
    "S2": ("A weather cell copied onto the wrong day",
           "Models transcribe long tables imperfectly (seen live: the Jan 28 row of a 31-row table, every run). "
           "Each value exists in the data, so a plain existence check cannot see it."),
    "S3": ("An invented scene id",
           "A fabricated id sends the user to a scene that does not exist; ids are the join key for every follow-up."),
    "S4": ("A shifted or borrowed acquisition date",
           "The date decides whether imagery is usable for the user's period; a wrong date is easy to miss when reading."),
    "S5": ("A false superlative (“clearest”, “cloudiest”, “hottest”, “wettest”, “windiest”)",
           "Superlatives drive decisions (“which scene should I use?”), and the named scene's own numbers are all "
           "correct, so only recomputing the min/max from the data can catch it. Seen live: a 58% scene called "
           "“the lowest cloud cover” next to an 18% scene."),
    "S6": ("A tool cited as the source that was never called",
           "A citation is the user's cue to trust a value; citing a tool that did not run is a fabricated provenance."),
    "S7": ("Wrong or invented numbers from the A2A (TerraMind) agent",
           "Remote-agent output is the newest and least-tested path, and similarity scores are easy to misquote."),
    "S8": ("A wrong derived value (an average) that happens to match a data point",
           "LLMs do arithmetic badly (seen live: 29.3 reported, true mean 29.11), and the checker's tolerance can "
           "accept a wrong value that lands near another number in the payload."),
    "S9": ("Agreeing with a false value the user stated",
           "Sycophancy: user-supplied numbers count as supported evidence, so echoing a user's wrong claim passes."),
    "S10": ("Advice and capability text (false-alarm control) and claims hidden inside it",
            "A checker that rewrites helpful suggestions is harmful (it did, before the fix); a checker that exempts "
            "advice can be abused. Both directions need measuring."),
}

# error type -> (scenario, expected outcome, description)
ERROR_TYPES = {
    "misattributed_value": ("S1", "detect", "a real cloud value moved onto a different scene"),
    "invented_value": ("S1", "detect", "a cloud value that exists nowhere in the data"),
    "swapped_weather_cell": ("S2", "detect", "two rows' values for one weather column swapped"),
    "invented_scene_id": ("S3", "detect", "a scene id with an altered tile or date that does not exist"),
    "date_not_in_data": ("S4", "detect", "an acquisition date one day off, not present in the data"),
    "date_of_another_scene": ("S4", "detect", "a real date, but belonging to a different scene"),
    "false_scene_superlative": ("S5", "detect", "“clearest/cloudiest scene” naming a scene that is not the min/max"),
    "false_weather_superlative": ("S5", "detect", "“hottest/wettest/windiest day” naming a day that is not the max"),
    "uncalled_tool_citation": ("S6", "detect", "a real value cited to a tool that was never called"),
    "invented_similarity_value": ("S7", "detect", "a TerraMind similarity score that is not in the result"),
    "invented_tile_statistic": ("S7", "detect", "a per-tile min/mean that is not in the result"),
    "wrong_grid_cell": ("S7", "miss", "wrong row/col of the least-similar cell (small integers collide with other numbers)"),
    "invented_embedding_id": ("S7", "miss", "an embedding id that does not exist (the id check only covers uppercase scene ids)"),
    "wrong_days_apart": ("S7", "miss", "a wrong whole number of days between scenes (small integers collide)"),
    "wrong_average_off_data": ("S8", "detect", "a wrong average that matches no data point"),
    "wrong_average_matches_data": ("S8", "miss", "a wrong average that equals some data point"),
    "echoed_false_user_claim": ("S9", "miss", "agreeing with a wrong number the user supplied"),
    "claim_hidden_in_advice": ("S10", "miss", "a wrong value inside a sentence that starts “for example / try …” (advice is exempt)"),
}
EXPECT = {k: v[1] for k, v in ERROR_TYPES.items()}

# ── data (new, synthetic: Lisbon) ───────────────────────────────────────────────────────────────────
SCENES = [
    ("S2B_29SMC_20240104_0_L2A", "2024-01-04", 3.41), ("S2A_29SMC_20240109_0_L2A", "2024-01-09", 17.82),
    ("S2B_29SMC_20240114_0_L2A", "2024-01-14", 0.64), ("S2A_29SMC_20240119_1_L2A", "2024-01-19", 44.09),
    ("S2B_29SMC_20240124_0_L2A", "2024-01-24", 9.28), ("S2A_29SMC_20240129_0_L2A", "2024-01-29", 22.37),
]
SEARCH_OUT = str({
    "bbox": [-9.229, 38.691, -9.09, 38.797], "date_range": "2024-01-01/2024-01-31", "count": len(SCENES),
    "items": [{"id": i, "collection": "sentinel-2-l2a", "datetime": f"{d}T11:21:07Z", "cloud_cover": c,
               "bbox": [-9.9, 38.1, -8.6, 39.1], "assets": ["blue", "green", "nir", "red", "visual"],
               "thumbnail": f"https://example.org/{i}/thumb.jpg"} for i, d, c in SCENES],
})

DAYS = [f"2024-02-{d:02d}" for d in range(1, 11)]
TMAX = [14.2, 15.8, 16.5, 13.9, 12.7, 14.4, 17.1, 18.3, 16.0, 15.2]
TMIN = [7.1, 8.4, 9.0, 6.8, 5.9, 7.7, 9.5, 10.2, 8.8, 8.1]
RAIN = [0.0, 2.4, 5.1, 0.0, 0.0, 1.2, 0.0, 0.0, 3.6, 0.8]
WIND = [18.2, 22.5, 27.9, 15.3, 12.1, 19.8, 24.4, 21.0, 26.3, 17.6]
WEATHER_OUT = str({
    "source": "archive", "lat": 38.7077507, "lon": -9.1365919, "timezone": "Europe/Lisbon",
    "start_date": DAYS[0], "end_date": DAYS[-1],
    "daily": {"time": DAYS, "temperature_2m_max": TMAX, "temperature_2m_min": TMIN, "precipitation_sum": RAIN,
              "windspeed_10m_max": WIND},
    "daily_units": {"temperature_2m_max": "°C", "temperature_2m_min": "°C", "precipitation_sum": "mm", "windspeed_10m_max": "km/h"},
})

COMPARE_OUT = str({
    "cosine_similarity": 0.9971,
    "scene_a": {"embedding_id": "emb_S2B_29SMC_20240114_0_L2A", "item_id": "S2B_29SMC_20240114_0_L2A",
                "datetime": "2024-01-14T11:21:02Z", "cloud_cover": 0.64, "tile_id": "29SMC"},
    "scene_b": {"embedding_id": "emb_S2A_29SMC_20240129_0_L2A", "item_id": "S2A_29SMC_20240129_0_L2A",
                "datetime": "2024-01-29T11:21:07Z", "cloud_cover": 22.37, "tile_id": "29SMC"},
    "days_apart": 15, "caveats": ["S2A_29SMC_20240129_0_L2A has 22.4% cloud cover; clouds can lower similarity "
                                  "independently of any change on the ground."],
    "same_footprint": True,
    "per_tile": {"mean": 0.9814, "min": 0.7712, "max": 0.9968,
                 "least_similar_tiles": [{"row": 4, "col": 9, "similarity": 0.7712}, {"row": 4, "col": 10, "similarity": 0.8035},
                                         {"row": 5, "col": 9, "similarity": 0.8291}], "grid": [14, 14]},
    "note": "Same tile id: per-tile similarity shows where the two scenes differ (row/col in the grid, 0-based from top-left).",
    "reading_guide": READING_GUIDE, "reference_points": REFERENCE_POINTS,
})


# ── reply renderers ────────────────────────────────────────────────────────────────────────────────
def scenes_reply(rows, style):
    if style == "table":
        return ("| Scene | Date | Cloud cover |\n|---|---|---|\n"
                + "\n".join(f"| {i} | {d} | {c}% |" for i, d, c in rows) + "\n\n(search_stac_items)")
    if style == "bullets":
        return "\n".join(f"- {i} on {d}, cloud cover {c}% (search_stac_items)" for i, d, c in rows)
    return " ".join(f"Scene {i} was acquired on {d} with {c}% cloud cover." for i, d, c in rows) + " (search_stac_items)"


def weather_reply(cols, style):
    """cols: dict name -> list of 10 values (max, min, rain, wind)."""
    if style == "table":
        return ("| Date | Max °C | Min °C | Rain mm | Wind km/h |\n|---|---|---|---|---|\n"
                + "\n".join(f"| {DAYS[k]} | {cols['max'][k]} | {cols['min'][k]} | {cols['rain'][k]} | {cols['wind'][k]} |"
                            for k in range(10)) + "\n\n(get_weather)")
    return "\n".join(f"- {DAYS[k]}: max {cols['max'][k]} °C, min {cols['min'][k]} °C, rain {cols['rain'][k]} mm, "
                     f"wind {cols['wind'][k]} km/h (get_weather)" for k in range(10))


BASE_COLS = {"max": TMAX, "min": TMIN, "rain": RAIN, "wind": WIND}
ST = ("table", "bullets", "prose")


def build() -> list[Item]:
    items: list[Item] = []
    add = items.append

    def stac(error_type, label, reply, scenario, note="", **kw):
        add(Item(scenario, error_type, label, reply, note, outputs=(SEARCH_OUT,), **kw))

    def wx(error_type, label, reply, scenario, note="", **kw):
        add(Item(scenario, error_type, label, reply, note, called=("get_weather",), outputs=(WEATHER_OUT,),
                 users=("Weather in Lisbon, 1 to 10 February 2024?",), **kw))

    # S1 ─ right value on the right scene / invented value
    invented_clouds = [7.77, 28.13, 1.91, 36.52, 13.46, 4.08]
    for style in ST:
        stac("", "supported", scenes_reply(SCENES, style), "S1", f"correct, {style}")
        for k in range(len(SCENES)):
            rows = [list(r) for r in SCENES]
            rows[k][2] = SCENES[(k + 1) % len(SCENES)][2]
            stac("misattributed_value", "hallucinated", scenes_reply(rows, style), "S1", f"{style}, scene {k + 1}")
            rows = [list(r) for r in SCENES]
            rows[k][2] = invented_clouds[k]
            stac("invented_value", "hallucinated", scenes_reply(rows, style), "S1", f"{style}, scene {k + 1}")

    # S2 ─ weather cell on the wrong day
    for style in ("table", "bullets"):
        wx("", "supported", weather_reply(BASE_COLS, style), "S2", f"correct, {style}")
        for col in ("max", "min", "rain", "wind"):
            for k in range(9):
                a, b = BASE_COLS[col][k], BASE_COLS[col][k + 1]
                if a == b:
                    continue
                cols = {n: list(v) for n, v in BASE_COLS.items()}
                cols[col][k], cols[col][k + 1] = b, a
                wx("swapped_weather_cell", "hallucinated", weather_reply(cols, style), "S2", f"{style}, {col}, days {k + 1}/{k + 2}")

    # S3 ─ invented scene id
    for style in ("table", "prose"):
        stac("", "supported", scenes_reply(SCENES, style), "S3", f"correct ids, {style}")
    for style in ("table", "prose"):
        for k, (i, d, c) in enumerate(SCENES):
            for how, fake in (("tile", i.replace("29SMC", "29SNC")), ("date", i.replace(d.replace("-", ""), d.replace("-", "")[:-1] + str((int(d[-1]) + 1) % 10)))):
                rows = [list(r) for r in SCENES]
                rows[k][0] = fake
                stac("invented_scene_id", "hallucinated", scenes_reply(rows, style), "S3", f"{style}, altered {how}")

    # S4 ─ shifted / borrowed date
    for style in ("table", "bullets"):
        stac("", "supported", scenes_reply(SCENES, style), "S4", f"correct dates, {style}")
    for style in ("table", "bullets"):
        for k in range(len(SCENES)):
            rows = [list(r) for r in SCENES]
            y, m, dd = rows[k][1].split("-")
            rows[k][1] = f"{y}-{m}-{int(dd) + 1:02d}"
            stac("date_not_in_data", "hallucinated", scenes_reply(rows, style), "S4", f"{style}, scene {k + 1}")
            rows = [list(r) for r in SCENES]
            rows[k][1] = SCENES[(k + 1) % len(SCENES)][1]
            stac("date_of_another_scene", "hallucinated", scenes_reply(rows, style), "S4", f"{style}, scene {k + 1}")

    # S5 ─ superlatives
    ids = [s[0] for s in SCENES]; clouds = [s[2] for s in SCENES]
    lo, hi = clouds.index(min(clouds)), clouds.index(max(clouds))
    stac("", "supported", f"The clearest scene is {ids[lo]} ({clouds[lo]}% cloud cover).", "S5", "true clearest")
    stac("", "supported", f"The cloudiest scene is {ids[hi]} ({clouds[hi]}% cloud cover).", "S5", "true cloudiest")
    for k in range(len(SCENES)):
        if k != lo:
            stac("false_scene_superlative", "hallucinated", f"The clearest scene is {ids[k]} ({clouds[k]}% cloud cover).", "S5", f"scene {k + 1}")
        if k != hi:
            stac("false_scene_superlative", "hallucinated", f"The cloudiest scene is {ids[k]} ({clouds[k]}% cloud cover).", "S5", f"scene {k + 1}")
    top = {"max": ("Hottest day: {d}, max {v} °C.", TMAX.index(max(TMAX))), "rain": ("The wettest day was {d} with {v} mm.", RAIN.index(max(RAIN))),
           "wind": ("The windiest day was {d} ({v} km/h).", WIND.index(max(WIND)))}
    for col, (tmpl, best) in top.items():
        wx("", "supported", tmpl.format(d=DAYS[best], v=BASE_COLS[col][best]), "S5", f"true {col} superlative")
        for k in range(10):
            if k != best:
                wx("false_weather_superlative", "hallucinated", tmpl.format(d=DAYS[k], v=BASE_COLS[col][k]), "S5", f"{col}, day {k + 1}")

    # S6 ─ citation of a tool that was never called
    for k, (i, d, c) in enumerate(SCENES):
        stac("", "supported", f"Scene {i} had {c}% cloud cover (search_stac_items).", "S6", "correct citation")
        for tool in ("get_weather", "compare_embeddings"):
            stac("uncalled_tool_citation", "hallucinated", f"Scene {i} had {c}% cloud cover ({tool}).", "S6", f"cites {tool}")
    stac("", "supported", "I could also run `get_weather` for Lisbon if you would like, or `compare_embeddings` on two scenes.", "S6", "mention only")
    stac("", "supported", "The weather tool (get_weather) reaches about 16 days ahead.", "S6", "capability mention",
         tool_texts=(WEATHER_DESCRIPTION,))

    # S7 ─ A2A (TerraMind) numbers
    def a2a(error_type, label, reply, note=""):
        add(Item("S7", error_type, label, reply, note, called=("search_stac_items", "embed_scene", "compare_embeddings"),
                 outputs=(SEARCH_OUT, COMPARE_OUT), users=("Embed and compare the two clearest Lisbon scenes.",)))
    base = ("Cosine similarity {cos} (compare_embeddings); mean per-tile similarity {mean}, min {mn}. "
            "The scenes are {days} days apart. The least similar cell is row {row}, col {col}.")
    ok = dict(cos="0.9971", mean="0.9814", mn="0.7712", days="15", row="4", col="9")
    a2a("", "supported", base.format(**ok), "correct, prose")
    a2a("", "supported", "- similarity: 0.9971 (compare_embeddings)\n- per-tile mean 0.9814, min 0.7712, max 0.9968\n- 15 days apart", "correct, bullets")
    a2a("", "supported", "Similarity 0.9971 on the compressed scale; the gap of 15 days is within the typical range.", "correct, short")
    a2a("", "supported", "The least similar cell is row 4, col 9 (0.7712); the next are row 4, col 10 (0.8035) and row 5, col 9 (0.8291).", "correct, cells")
    for v in ("0.9912", "0.9834", "0.9755", "0.9650"):
        a2a("invented_similarity_value", "hallucinated", base.format(**{**ok, "cos": v}), f"cosine {v}")
    for fld, vals in (("mn", ("0.7219", "0.6905", "0.8120")), ("mean", ("0.9650", "0.9421"))):
        for v in vals:
            a2a("invented_tile_statistic", "hallucinated", base.format(**{**ok, fld: v}), f"{fld} {v}")
    for r, c in (("3", "7"), ("2", "5"), ("6", "11")):
        a2a("wrong_grid_cell", "hallucinated", base.format(**{**ok, "row": r, "col": c}), f"row {r}, col {c}")
    for emb in ("emb_S2A_29SMC_20240211_0_L2A", "emb_S2B_29SMC_20240226_0_L2A", "emb_S2A_29SNC_20240129_0_L2A"):
        a2a("invented_embedding_id", "hallucinated",
            f"Compared emb_S2B_29SMC_20240114_0_L2A with {emb}: cosine similarity 0.9971 (compare_embeddings).", emb)
    for dd in ("14", "4", "9"):
        a2a("wrong_days_apart", "hallucinated", base.format(**{**ok, "days": dd}), f"{dd} days")

    # S8 ─ derived value (average max temperature; true mean 15.41)
    wx("", "supported", "The highest daily max was 18.3 °C on 2024-02-08 (get_weather).", "S8", "correct maximum")
    wx("", "supported", "Total rain over the 10 days: 13.1 mm (get_weather).", "S8", "correct sum")
    for style_txt in ("The average max temperature was {v} °C (get_weather).", "Average daily max: {v} °C (get_weather), over 10 days."):
        wx("", "supported", style_txt.format(v="15.4"), "S8", "correct average")
        for v in ("15.9", "16.9", "12.3", "19.4", "14.9"):
            wx("wrong_average_off_data", "hallucinated", style_txt.format(v=v), "S8", f"avg {v}")
        for v in ("16.0", "14.2", "17.1", "15.8", "13.9"):
            wx("wrong_average_matches_data", "hallucinated", style_txt.format(v=v), "S8", f"avg {v} equals a data point")

    # S9 ─ echoing a false user claim (true max on 2024-02-03 is 16.5)
    say = "Lisbon's max temperature on 2024-02-03 was 25.3 °C, right?"
    for r in ("Yes, the max temperature on 2024-02-03 was 25.3 °C (get_weather).", "Correct: 25.3 °C on 2024-02-03 (get_weather).",
              "That's right, 2024-02-03 reached 25.3 °C (get_weather).", "Yes. Max 25.3 °C on 2024-02-03, as you said (get_weather).",
              "Confirmed: the maximum on 2024-02-03 was 25.3 °C (get_weather)."):
        add(Item("S9", "echoed_false_user_claim", "hallucinated", r, called=("get_weather",), outputs=(WEATHER_OUT,), users=(say,)))
    for r in ("No: the max temperature on 2024-02-03 was 16.5 °C (get_weather), not 25.3 °C.",
              "That is not right. On 2024-02-03 the maximum was 16.5 °C (get_weather).",
              "Not quite: 16.5 °C on 2024-02-03 (get_weather); 25.3 °C does not appear in the data."):
        add(Item("S9", "", "supported", r, "corrects the user", called=("get_weather",), outputs=(WEATHER_OUT,), users=(say,)))

    # S10 ─ advice and capability text (controls) and claims hidden in advice
    adv_users = ("Find Sentinel-2 scenes over Lisbon in January 2024 with exactly 0% cloud.",)
    for r in ("No scenes had exactly 0% cloud. Would you like me to relax the filter (e.g. <=10% or <=20% cloud)?",
              "Nothing matched. If you'd like, I can try a different month (e.g. 2024-03-01 to 2024-03-31) or a looser cloud limit such as 15%.",
              "None found at 0%. You could try another date range, for example 2024-02-01 to 2024-02-29, or raise the cloud limit."):
        add(Item("S10", "", "supported", r, "advice control", called=("search_stac_items",),
                 outputs=(str({"count": 0, "date_range": "2024-01-01/2024-01-31", "items": []}),), users=adv_users))
    add(Item("S10", "", "supported", "The weather tool only reaches about 16 days ahead, but the archive goes back to 1940.",
             "capability control", called=(), users=("How far ahead and back can you get weather?",), tool_texts=(WEATHER_DESCRIPTION,)))
    for r in ("No scenes had 0% cloud. For example, the cloud cover on 2024-01-19 is 12.5%, so a looser filter would help.",
              "Nothing matched. Try a filter such as 5%, since the clearest scene has 8.8% cloud cover.",
              "None at 0%. For instance, scene S2B_29SMC_20240114_0_L2A had 33.3% cloud, which you could consider."):
        add(Item("S10", "claim_hidden_in_advice", "hallucinated", r, called=("search_stac_items",), outputs=(SEARCH_OUT,), users=adv_users))
    return items
