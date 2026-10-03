"""Offline tests for provider selection. Run: python -m evals.test_llm_factory"""

from service.llm import make_llm, resolve


def test_default_is_groq():
    assert resolve({}) == {"provider": "groq", "model": "qwen/qwen3.8-27b"}
    assert resolve({"EVE_MODEL": "x/y"})["model"] == "x/y"


def test_ollama_needs_no_key_and_has_default_model():
    c = resolve({"EVE_LLM_PROVIDER": "ollama"})
    assert c["base_url"] == "http://localhost:11434/v1" and c["model"] == "qwen3:8b" and c["api_key"]


def test_vllm_needs_model_and_honours_base_url():
    try:
        resolve({"EVE_LLM_PROVIDER": "vllm"})
    except ValueError as exc:
        assert "EVE_MODEL" in str(exc)
    else:
        raise AssertionError
    c = resolve({"EVE_LLM_PROVIDER": "vllm", "EVE_MODEL": "Qwen/Qwen3-8B", "EVE_LLM_BASE_URL": "http://gpu:9000/v1"})
    assert c["base_url"] == "http://gpu:9000/v1"


def test_openrouter_and_hf_keys():
    for prov, var in (("openrouter", "OPENROUTER_API_KEY"), ("hf", "HF_TOKEN")):
        try:
            resolve({"EVE_LLM_PROVIDER": prov, "EVE_MODEL": "m"})
        except ValueError as exc:
            assert var in str(exc)
        else:
            raise AssertionError(prov)
        assert resolve({"EVE_LLM_PROVIDER": prov, "EVE_MODEL": "m", var: "k"})["api_key"] == "k"


def test_unknown_provider_rejected_and_models_build():
    try:
        resolve({"EVE_LLM_PROVIDER": "nope"})
    except ValueError:
        pass
    else:
        raise AssertionError
    llm = make_llm({"EVE_LLM_PROVIDER": "ollama"})
    assert hasattr(llm, "bind_tools") and type(llm).__name__ == "ChatOpenAI"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} passed")
