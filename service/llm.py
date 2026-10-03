"""Model factory: one place that decides which LLM the agent talks to.

The graph only needs a LangChain chat model with ``bind_tools``; nothing else in the repo
names a provider. Pick one with environment variables:

    EVE_LLM_PROVIDER   groq (default) | ollama | vllm | openrouter | hf | openai
    EVE_MODEL          model name as the provider spells it (default only for groq/ollama)
    EVE_LLM_BASE_URL   override the endpoint (any OpenAI-compatible server)
    EVE_LLM_API_KEY    override the key (else the provider's usual variable)

Every provider except ``groq`` goes through the OpenAI-compatible chat API, which is what
Ollama, vLLM, OpenRouter and Hugging Face's router/endpoints all serve. Models that don't
emit native tool calls still work: the graph parses text-format calls (see
``agents/graphs/utils.py``).
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.language_models import BaseChatModel

# provider -> (base_url, api-key env var, key used when none is needed, default model)
PRESETS = {
    "ollama": ("http://localhost:11434/v1", None, "ollama", "qwen3:8b"),
    "vllm": ("http://localhost:8000/v1", None, "EMPTY", None),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", None, None),
    "hf": ("https://router.huggingface.co/v1", "HF_TOKEN", None, None),
    "openai": (None, "OPENAI_API_KEY", None, None),
}
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"


def resolve(env: Optional[dict] = None) -> dict:
    """Settings the factory would use (separate from construction so it is testable)."""
    env = os.environ if env is None else env
    provider = env.get("EVE_LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        return {"provider": "groq", "model": env.get("EVE_MODEL", DEFAULT_GROQ_MODEL)}
    if provider not in PRESETS:
        raise ValueError(f"EVE_LLM_PROVIDER={provider!r}; expected groq or one of {sorted(PRESETS)}")
    base_url, key_var, no_key, default_model = PRESETS[provider]
    model = env.get("EVE_MODEL") or default_model
    if not model:
        raise ValueError(f"EVE_MODEL is required for provider {provider!r} (e.g. a model id as {provider} names it)")
    api_key = env.get("EVE_LLM_API_KEY") or (env.get(key_var) if key_var else None) or no_key
    if not api_key:
        raise ValueError(f"{key_var} (or EVE_LLM_API_KEY) is required for provider {provider!r}")
    return {"provider": provider, "model": model, "base_url": env.get("EVE_LLM_BASE_URL") or base_url, "api_key": api_key}


def make_llm(env: Optional[dict] = None) -> BaseChatModel:
    cfg = resolve(env)
    if cfg["provider"] == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=cfg["model"], temperature=0)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=cfg["model"], base_url=cfg["base_url"], api_key=cfg["api_key"], temperature=0)


def describe(env: Optional[dict] = None) -> str:
    cfg = resolve(env)
    return f"{cfg['provider']}:{cfg['model']}"
