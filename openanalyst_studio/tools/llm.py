# tools/llm.py
import os
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

from langchain_openai import ChatOpenAI, AzureChatOpenAI

# ------- tiny helpers -------


def _secrets() -> Dict[str, Any]:
    try:
        import streamlit as st  # type: ignore
        return dict(st.secrets)  # copy into a plain dict
    except Exception:
        return {}


def _cfg(key: str, default: Optional[str]=None) -> Optional[str]:
    """Config lookup: Streamlit secrets -> env -> default. Empty string -> None."""
    s = _secrets()
    v = s.get(key)
    if v not in (None, ""):
        return str(v)
    v = os.getenv(key)
    return v if v not in (None, "") else default


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        v = value.strip()
        if v == "":
            return None
        try:
            return int(v)
        except ValueError:
            return None
    try:
        return int(value)
    except Exception:
        return None


def _clean_model_kwargs(mk_in):
    if not mk_in: return {}
    reserved = {"seed", "response_format"}
    return {k: v for k, v in mk_in.items() if v not in ("", None) and k not in reserved}


def _normalize_model_kwargs(
    extra: Optional[Dict[str, Any]]=None,
    response_format: Optional[Dict[str, Any]]=None,
) -> Dict[str, Any]:
    mk = _clean_model_kwargs(extra)
    if response_format:
        mk["response_format"] = response_format
    return mk


def _mk_cache_key(**kw: Any) -> Tuple:
    """Stable, hashable cache key."""

    def _freeze(x: Any) -> Any:
        if isinstance(x, dict):
            return tuple(sorted((k, _freeze(v)) for k, v in x.items()))
        if isinstance(x, (list, tuple, set)):
            return tuple(_freeze(v) for v in x)
        return x

    return tuple(sorted((k, _freeze(v)) for k, v in kw.items()))

# tools/llm.py

def _unfreeze(x):
    """Convert the frozen (hashable) structures back into Python containers."""
    if isinstance(x, tuple):
        # dict-like? tuple of (key, value) pairs
        if all(isinstance(i, tuple) and len(i) == 2 for i in x):
            return {k: _unfreeze(v) for k, v in x}
        # list/tuple-like
        return [ _unfreeze(v) for v in x ]
    return x


@lru_cache(maxsize=32)
def _cached_make_llm(key_tuple: tuple) -> ChatOpenAI:
    cfg = dict(key_tuple)  # because _mk_cache_key returns tuple of (k, v)

    # --- IMPORTANT: turn frozen tuples back into dicts/lists ---
    mk_frozen = cfg.get("model_kwargs", ())
    mk = _unfreeze(mk_frozen) if isinstance(mk_frozen, tuple) else (mk_frozen or {})
    if not isinstance(mk, dict):
        mk = {}
    cfg["model_kwargs"] = mk
    # -----------------------------------------------------------

    provider = cfg["provider"]
    common_kwargs = dict(
        temperature=cfg["temperature"],
        timeout=cfg["timeout"],
        max_retries=cfg["max_retries"],
        streaming=cfg["streaming"],
        seed=cfg["seed"],
        model_kwargs=cfg["model_kwargs"],  # now a dict
    )
    api_key = cfg["api_key"]

    if provider == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=cfg["azure_deployment"],
            api_version=cfg["azure_api_version"],
            azure_endpoint=cfg["azure_endpoint"],
            api_key=api_key,
            **common_kwargs,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=cfg["model"],
            api_key=api_key,
            base_url=cfg["base_url"],
            organization=cfg["organization"],
            **common_kwargs,
        )

# ------- main API -------


def get_llm(
    model: Optional[str]=None,
    temperature: Optional[float]=None,
    timeout: Optional[int]=None,
    api_key: Optional[str]=None,
    streaming: bool=False,
    response_format: Optional[Dict[str, Any]]=None,  # e.g. {"type": "json_object"}
    model_kwargs: Optional[Dict[str, Any]]=None,
) -> ChatOpenAI:
    """
    Return a cached ChatOpenAI/AzureChatOpenAI client configured from:
    1) function args (highest precedence)
    2) Streamlit secrets
    3) Environment variables
    4) sane defaults

    Supported env/secrets:
      - OPENAI_API_KEY (or pass api_key=)
      - MODEL_NAME (default: gpt-4o-mini)
      - TEMPERATURE (default: 0.2)
      - OPENAI_TIMEOUT (default: 60)
      - OPENAI_MAX_RETRIES (default: 2)
      - OPENAI_ORGANIZATION (optional)
      - OPENAI_BASE_URL (optional; for self-hosted/OpenRouter/Ollama gateway)
      - OPENAI_SEED (optional int)
      - OPENAI_API_TYPE=azure (switch to AzureChatOpenAI)
        * AZURE_OPENAI_ENDPOINT
        * OPENAI_API_VERSION
        * AZURE_OPENAI_DEPLOYMENT (deployment name)

    Notes:
      - If response_format is {"type":"json_object"} streaming will be disabled.
      - Identical configs are cached.
    """
    # API key
    key = api_key or _cfg("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is not set.")

    # Base config
    m = model or _cfg("MODEL_NAME", "gpt-4o-mini") or "gpt-4o-mini"
    t = float(temperature if temperature is not None else _cfg("TEMPERATURE", "0.2") or 0.2)
    to = int(timeout if timeout is not None else _cfg("OPENAI_TIMEOUT", "60") or 60)
    retries = int(_cfg("OPENAI_MAX_RETRIES", "2") or 2)
    org = _cfg("OPENAI_ORGANIZATION")
    base_url = _cfg("OPENAI_BASE_URL")
    seed = _parse_int(_cfg("OPENAI_SEED"))

    # Provider selection
    provider = (_cfg("OPENAI_API_TYPE", "openai") or "openai").strip().lower()

    # Normalize kwargs & JSON mode guard
    mk = _normalize_model_kwargs(model_kwargs, response_format)
    if mk.get("response_format", {}).get("type") == "json_object" and streaming:
        streaming = False  # JSON mode + streaming don't mix

    # Azure specifics
    azure_endpoint = _cfg("AZURE_OPENAI_ENDPOINT")
    azure_api_version = _cfg("OPENAI_API_VERSION")
    azure_deployment = _cfg("AZURE_OPENAI_DEPLOYMENT") or m  # fall back to model name if not provided

    # Build cache key (include only serializable values)
    key_tuple = _mk_cache_key(
        provider=provider,
        api_key=key,  # cached; if you prefer not to cache the raw key, hash it instead
        model=m,
        temperature=t,
        timeout=to,
        max_retries=retries,
        streaming=streaming,
        base_url=base_url,
        organization=org,
        seed=seed,
        model_kwargs=mk,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        azure_deployment=azure_deployment,
    )

    # Create (or fetch cached) client
    llm = _cached_make_llm(key_tuple)

    # Keep env in sync for libs that read OPENAI_API_KEY lazily
    os.environ["OPENAI_API_KEY"] = key
    if provider == "azure":
        if azure_endpoint:
            os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
        if azure_api_version:
            os.environ["OPENAI_API_VERSION"] = azure_api_version

    return llm
