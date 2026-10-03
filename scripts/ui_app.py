"""Chat UI for the EO agent: a thin client of the HTTP API.

It holds no agent of its own, so everything the API does (MCP tools, TerraMind A2A skills,
verifier, SQLite-backed sessions) is what you see here. Tool calls and results are shown
per turn; geocode / STAC / embedding results are drawn on a map.

Session handling:
  * every chat tab shows its session id; paste an id in the sidebar to reload that session
    (the API reads it from its SQLite checkpoint, so it survives restarts);
  * "Export" downloads the whole session as one JSON (messages, trace, TerraMind log,
    embeddings); "Open session file" replays such a file read-only, no API history needed.

Usage:
    scripts/start_all.sh                      # or start the processes yourself, see README
    streamlit run scripts/ui_app.py           # API at $EVE_API_BASE (default http://127.0.0.1:8000)
"""

import ast
import json
import os
import re
import uuid

import folium
import httpx
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

API = os.environ.get("EVE_API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="EO Agent", page_icon="🛰️", layout="wide")
st.title("🛰️ EO Agent")


def api(method: str, path: str, **kw) -> httpx.Response:
    return httpx.request(method, f"{API}{path}", timeout=kw.pop("timeout", 30), **kw)


try:
    health = api("GET", "/health", timeout=5).json()
except httpx.HTTPError:
    st.error(f"API not reachable at {API}. Start it with `scripts/start_all.sh` (see README).")
    st.stop()
st.caption(f"Model: `{health.get('model')}` · API: {API}")


# ── rendering of tool results ────────────────────────────────────────────────


def _bbox_bounds(bbox):
    """[west, south, east, north] -> folium [[south, west], [north, east]]."""
    w, so, e, n = bbox
    return [[so, w], [n, e]]


def render_map(tool_name: str, parsed: dict, key: str) -> bool:
    """Draw geocode / STAC-search / embedding results on a world map. True if a map was drawn."""
    boxes, markers = [], []  # (bounds, label, color) / (lat, lon, label)

    if tool_name == "geocode_location" and parsed.get("results"):
        for r in parsed["results"]:
            if r.get("bbox"):
                boxes.append((_bbox_bounds(r["bbox"]), r["display_name"], "#2b6cb0"))
            markers.append((r["lat"], r["lon"], r["display_name"]))
    elif tool_name == "search_stac_items" and parsed.get("items"):
        if parsed.get("bbox"):
            boxes.append((_bbox_bounds(parsed["bbox"]), "Search area", "#2b6cb0"))
        for it in parsed["items"]:
            if it.get("bbox"):
                label = f"{it['id']} · {(it.get('datetime') or '')[:10]} · cloud {it.get('cloud_cover', '?')}%"
                boxes.append((_bbox_bounds(it["bbox"]), label, "#dd6b20"))
    elif tool_name == "embed_scene" and parsed.get("crop_bbox"):
        boxes.append((_bbox_bounds(parsed["crop_bbox"]), f"{parsed.get('item_id')} (embedded crop)", "#38a169"))
    else:
        return False

    if not boxes and not markers:
        return False

    m = folium.Map(tiles="OpenStreetMap", zoom_control=True)
    for bounds, label, color in boxes:
        folium.Rectangle(bounds, color=color, weight=2, fill=True, fill_opacity=0.08, tooltip=label).add_to(m)
    for lat, lon, label in markers:
        folium.Marker([lat, lon], tooltip=label).add_to(m)
    all_pts = [pt for b, _, _ in boxes for pt in b] + [[la, lo] for la, lo, _ in markers]
    m.fit_bounds(all_pts, padding=(20, 20))
    st_folium(m, height=320, use_container_width=True, returned_objects=[], key=key)
    return True


def _parse_result(result_str: str):
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(result_str)
        except (ValueError, SyntaxError, TypeError):
            continue
    return None


def render_tool_result(tool_name: str, result_str: str, key: str = "") -> None:
    parsed = _parse_result(result_str)
    if isinstance(parsed, dict) and "error" not in parsed:
        render_map(tool_name, parsed, key=f"map-{key}")

    if not isinstance(parsed, dict):
        st.code(result_str, language="json")
        return

    if tool_name == "search_stac_items" and parsed.get("items"):
        st.caption(f"{parsed['count']} scene(s) · {parsed.get('date_range') or 'no date filter'}")
        cols = st.columns(3)
        for i, item in enumerate(parsed["items"]):
            with cols[i % 3]:
                if item.get("thumbnail"):
                    st.image(item["thumbnail"], width="stretch")
                else:
                    st.caption("(no thumbnail asset)")
                st.caption(f"**{item['id']}**\n\n{item.get('datetime', '')[:10]} · cloud {item.get('cloud_cover', '?')}%")
        return

    if tool_name == "list_stac_collections" and parsed.get("collections"):
        rows = [{"id": c.get("id"), "title": c.get("title"), "description": (c.get("description") or "")[:120]}
                for c in parsed["collections"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        return

    if tool_name == "get_stac_item" and parsed.get("id"):
        props = parsed.get("properties") or {}
        thumb = ((parsed.get("assets") or {}).get("thumbnail") or {}).get("href")
        left, right = st.columns([1, 2])
        with left:
            if thumb:
                st.image(thumb, width="stretch")
        with right:
            st.markdown(f"**{parsed['id']}**  \n{parsed.get('collection')} · {str(parsed.get('datetime') or props.get('datetime') or '')[:19]}")
            st.markdown(f"Cloud cover: **{props.get('eo:cloud_cover', '?')}%** · Platform: {props.get('platform', '?')}")
            st.caption("Assets: " + ", ".join(sorted((parsed.get("assets") or {}).keys())))
        with st.expander("Full item JSON"):
            st.code(result_str, language="json")
        return

    if tool_name == "embed_scene" and parsed.get("embedding_id"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Date", str(parsed.get("datetime") or "")[:10])
        c2.metric("Cloud cover", f"{parsed.get('cloud_cover', '?')}%")
        c3.metric("Tile", parsed.get("tile_id") or "?")
        st.caption(f"`{parsed['embedding_id']}` · shape {parsed.get('embedding_shape')} · session {parsed.get('session_id')}")
        return

    if tool_name == "compare_embeddings" and "cosine_similarity" in parsed:
        c1, c2 = st.columns(2)
        c1.metric("Cosine similarity", parsed["cosine_similarity"])
        c2.metric("Same footprint", "yes" if parsed.get("same_footprint") else "no")
        pt = parsed.get("per_tile")
        if pt:
            st.caption(f"Per tile: mean {pt['mean']} · min {pt['min']} · max {pt['max']}")
            st.dataframe(pd.DataFrame(pt["least_similar_tiles"]), hide_index=True)
        else:
            st.caption(parsed.get("note", ""))
        return

    if tool_name == "rank_similar" and parsed.get("ranking"):
        st.dataframe(pd.DataFrame(parsed["ranking"]), hide_index=True, width="stretch")
        return

    st.code(result_str, language="json")


# ── conversation model ───────────────────────────────────────────────────────


def messages_to_display(messages: list) -> list:
    """API/exported messages -> [{"role","content","steps"}]: one user entry per user message,
    one assistant entry per turn holding only that turn's tool calls (matched by tool_call_id)."""
    entries, current = [], None
    for m in messages:
        if m["role"] == "user":
            entries.append({"role": "user", "content": m["content"], "steps": []})
            current = None
            continue
        if current is None:
            current = {"role": "assistant", "content": "", "steps": []}
            entries.append(current)
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                current["steps"].append({"id": tc.get("id"), "tool": tc["name"], "args": tc["args"], "result": None})
            if not m.get("tool_calls") and m["content"]:
                current["content"] = re.sub(r"<think>.*?</think>", "", m["content"], flags=re.S).strip()
        elif m["role"] == "tool":
            for step in current["steps"]:
                if step["id"] == m["tool_call_id"]:
                    step["result"] = m["content"]
                    break
    return entries


def new_chat(thread_id=None, label=None, messages=None, readonly=False) -> None:
    st.session_state.chat_counter += 1
    st.session_state.chats.insert(0, {  # newest first -> lands on the first tab
        "thread_id": thread_id or str(uuid.uuid4()),
        "label": label or f"Chat {st.session_state.chat_counter}",
        "messages": messages or [],
        "readonly": readonly,
    })


def render_steps(steps, key_prefix, expanded) -> None:
    for si, step in enumerate(steps):
        with st.expander(f"🔧 {step['tool']}({step['args']})", expanded=expanded):
            render_tool_result(step["tool"], step["result"] or "", key=f"{key_prefix}-{si}")


def verify_note(trace: list) -> str | None:
    v = next((s for s in trace if s.get("step") == "verify"), None)
    if not v:
        return None
    a = v.get("args") or {}
    if a.get("caveat"):
        return "🛡️ Some values could not be verified against tool results (see the note in the answer)."
    if a.get("rewritten"):
        return "🛡️ The first draft contained unsupported values; the answer was rewritten from the tool results."
    return None


def render_chat(chat) -> None:
    tid = chat["thread_id"]
    st.caption("Session ID (paste it in the sidebar to reload this chat):")
    st.code(tid, language=None)

    for mi, msg in enumerate(chat["messages"]):
        with st.chat_message(msg["role"]):
            render_steps(msg.get("steps", []), f"{tid}-h{mi}", expanded=False)
            st.markdown(msg["content"])
            if msg.get("note"):
                st.caption(msg["note"])

    if chat.get("readonly"):
        st.info("Read-only replay of an exported session file.")
        return

    # Export (fetched on demand: it reads logs and the TerraMind agent's embeddings)
    if st.button("Prepare session export", key=f"prep-{tid}"):
        r = api("GET", f"/sessions/{tid}/export")
        if r.status_code == 200:
            st.session_state[f"export-{tid}"] = json.dumps(r.json(), indent=2)
        else:
            st.warning("Nothing to export yet: send a message first.")
    if f"export-{tid}" in st.session_state:
        st.download_button("⬇ Download session JSON", st.session_state[f"export-{tid}"],
                           file_name=f"session-{tid}.json", mime="application/json", key=f"dl-{tid}")

    prompt = st.chat_input("Ask about a place, imagery, weather, or embed and compare scenes...", key=f"input-{tid}")
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Working..."):
            try:
                resp = api("POST", "/chat", json={"session_id": tid, "message": prompt}, timeout=600)
                resp.raise_for_status()
                data = resp.json()
                history = api("GET", f"/sessions/{tid}").json()["messages"]
            except (httpx.HTTPError, KeyError) as exc:
                st.error(f"Request failed: {exc}")
                return
        last_user = max(i for i, m in enumerate(history) if m["role"] == "user")
        turn = messages_to_display(history[last_user:])
        assistant = next((e for e in turn if e["role"] == "assistant"), None) or {
            "role": "assistant", "content": data.get("reply") or "(no response generated)", "steps": []}
        assistant["note"] = verify_note(data.get("trace", []))
        render_steps(assistant["steps"], f"{tid}-n{len(chat['messages'])}", expanded=True)
        st.markdown(assistant["content"])
        if assistant["note"]:
            st.caption(assistant["note"])

    chat["messages"] += [{"role": "user", "content": prompt, "steps": []}, assistant]


# ── app ──────────────────────────────────────────────────────────────────────

if "chats" not in st.session_state:
    st.session_state.chats = []
    st.session_state.chat_counter = 0
    new_chat()

with st.sidebar:
    st.subheader("Chats")
    if st.button("➕ New chat", width="stretch"):
        new_chat()
        st.rerun()

    st.markdown("**Load a session by ID**")
    load_id = st.text_input("Session ID", key="load_id", label_visibility="collapsed", placeholder="paste a session ID")
    if st.button("Load session", width="stretch") and load_id.strip():
        tid = load_id.strip()
        if any(c["thread_id"] == tid for c in st.session_state.chats):
            st.info("That session is already open in a tab.")
        else:
            r = api("GET", f"/sessions/{tid}")
            if r.status_code == 404:
                st.error("No stored history for that session ID on this API's database.")
            else:
                new_chat(thread_id=tid, label=f"Session {tid[:6]}", messages=messages_to_display(r.json()["messages"]))
                st.rerun()

    st.markdown("**Open a session file**")
    up = st.file_uploader("Exported session JSON", type="json", label_visibility="collapsed")
    if up is not None and st.button("Replay file", width="stretch"):
        try:
            doc = json.load(up)
            new_chat(thread_id=doc["session_id"], label=f"File {doc['session_id'][:6]}",
                     messages=messages_to_display(doc["messages"]), readonly=True)
            st.rerun()
        except (ValueError, KeyError):
            st.error("Not a session export (expected the JSON from 'Download session JSON').")

    st.subheader("Try")
    for example in [
        "Show me cloud-free Sentinel-2 imagery of Hyderabad from January 2024.",
        "What's the historical weather in Paris for the first week of July 2023?",
        "Find two Sentinel-2 scenes over Hyderabad in January 2024, embed both with TerraMind and compare them.",
    ]:
        st.code(example, language=None)

tabs = st.tabs([c["label"] for c in st.session_state.chats])
for tab, chat in zip(tabs, st.session_state.chats):
    with tab:
        render_chat(chat)
