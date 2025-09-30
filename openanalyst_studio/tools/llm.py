# openanalyst_studio/tools/llm.py
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

from langchain_openai import ChatOpenAI

# Optional Azure support (only used if OPENAI_API_TYPE=azure)
try:  # pragma: no cover
    from langchain_openai import AzureChatOpenAI
except Exception:  # pragma: no cover
    AzureChatOpenAI = None  # type: ignore[misc,assignment]

# -----------------------
# Secrets / Config helpers
# -----------------------


def _secrets() -> Dict[str, Any]:
    """Return Streamlit secrets if available, else empty dict."""
    try:  # pragma: no cover
        import streamlit as st
        return dict(st.secrets)  # shallow copy
    except Exception:
        return {}


def _cfg(key: str, default: Optional[str]=None) -> Optional[str]:
    """
    Read from Streamlit secrets first, then environment; finally default.
    Empty strings are treated as unset.
    """
    s = _secrets()
    if key in s and s[key] not in (None, ""):
        return str(s[key])
    v = os.getenv(key)
    return v if v not in (None, "") else default


def _normalize_model_kwargs(
    extra: Optional[Dict[str, Any]]=None,
    response_format: Optional[Dict[str, Any]]=None,
) -> Dict[str, Any]:
    """
    Merge arbitrary model kwargs with response_format in a new dict.
    Ensures we never return None (important for cache key stability).
    """
    mk: Dict[str, Any] = dict(extra or {})
    if response_format:
        mk["response_format"] = response_format
    return mk


def _bool_flag(val: Optional[str], default: bool=False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

# -----------------------
# Internal cached factory
# -----------------------


def _mk_cache_key(
    provider: str,
    model: str,
    temperature: float,
    timeout: int,
    streaming: bool,
    base_url: Optional[str],
    organization: Optional[str],
    seed: Optional[int],
    max_retries: int,
    model_kwargs: Dict[str, Any],
    azure_endpoint: Optional[str],
    azure_api_version: Optional[str],
) -> Tuple[Any, ...]:
    """
    Build a hashable cache key. Dicts are converted to sorted JSON strings.
    """
    mk_json = json.dumps(model_kwargs or {}, sort_keys=True, separators=(",", ":"))
    return (
        provider,
        model,
        round(temperature, 6),
        int(timeout),
        bool(streaming),
        base_url or "",
        organization or "",
        seed if seed is not None else "",
        int(max_retries),
        mk_json,
        azure_endpoint or "",
        azure_api_version or "",
    )


@lru_cache(maxsize=16)
def _cached_make_llm(key: Tuple[Any, ...]) -> ChatOpenAI:
    """
    Create and return a LangChain LLM client using a normalized, hashable key.
    We pass everything we need via the key itself (unpacked below).
    """
    (
        provider,
        model,
        temperature,
        timeout,
        streaming,
        base_url,
        organization,
        seed,
        max_retries,
        mk_json,
        azure_endpoint,
        azure_api_version,
    ) = key

    model_kwargs = json.loads(mk_json) if mk_json else {}

    # Construct kwargs common to both OpenAI & AzureOpenAI variants (where supported)
    common_kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "timeout": timeout,
        "streaming": streaming,
        "max_retries": max_retries,
        "model_kwargs": model_kwargs,
    }
    if base_url:
        common_kwargs["base_url"] = base_url  # openai v1 supports base_url
    if organization:
        common_kwargs["organization"] = organization
    if seed is not None:
        # Supported by OpenAI API v1 via model_kwargs or top-level depending on LC version.
        # Put in model_kwargs to be safest across versions.
        common_kwargs["model_kwargs"] = {**model_kwargs, "seed": seed}

    if provider == "azure":
        if AzureChatOpenAI is None:  # pragma: no cover
            raise RuntimeError("Azure provider requested but AzureChatOpenAI is unavailable.")
        # Azure-specific kwargs
        azure_kwargs: Dict[str, Any] = {}
        if azure_endpoint:
            azure_kwargs["azure_endpoint"] = azure_endpoint
        if azure_api_version:
            azure_kwargs["api_version"] = azure_api_version
        # Note: Azure key is read from OPENAI_API_KEY env var or provided by OpenAI SDK config.
        return AzureChatOpenAI(**common_kwargs, **azure_kwargs)  # type: ignore[call-arg]
    else:
        # Default OpenAI-compatible (incl. self-hosted/OpenRouter/etc via base_url)
        return ChatOpenAI(**common_kwargs)

# -----------------------
# Public factory function
# -----------------------


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
        * OPENAI_API_VERSION (Azure API version)

    Notes:
      - If response_format is set to {"type":"json_object"} you should NOT enable streaming.
      - This function is safe to call repeatedly; identical configs reuse the same client.
    """
    # Key/API setup
    key = api_key or _cfg("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is not set.")

    # The OpenAI Python SDK reads from env; set it here for consumers that rely on it.
    os.environ.setdefault("OPENAI_API_KEY", key)

    # Base config
    m = model or _cfg("MODEL_NAME", "gpt-4o-mini") or "gpt-4o-mini"
    t = float(temperature if temperature is not None else _cfg("TEMPERATURE", "0.2") or 0.2)
    to = int(timeout if timeout is not None else _cfg("OPENAI_TIMEOUT", "60") or 60)
    retries = int(_cfg("OPENAI_MAX_RETRIES", "2") or 2)
    org = _cfg("OPENAI_ORGANIZATION")
    base_url = _cfg("OPENAI_BASE_URL")
    seed_str = _cfg("OPENAI_SEED")
    seed = int(seed_str) if seed_str and seed_str.isdigit() else None

    # Provider selection
    provider = (_cfg("OPENAI_API_TYPE", "openai") or "openai").strip().lower()
    azure_endpoint = _cfg("AZURE_OPENAI_ENDPOINT")
    azure_api_version = _cfg("OPENAI_API_VERSION")  # Azure uses this variable name in many setups

    # Merge kwargs & JSON mode
    mk = _normalize_model_kwargs(model_kwargs, response_format)

    # Guard: JSON mode + streaming is a bad combo
    if mk.get("response_format", {}).get("type") == "json_object" and streaming:
        # Don't raise—just disable streaming to be helpful.
        streaming = False

    # Build a cache key and return a cached client
    key_tuple = _mk_cache_key(
        provider=provider,
        model=m,
        temperature=t,
        timeout=to,
        streaming=streaming,
        base_url=base_url,
        organization=org,
        seed=seed,
        max_retries=retries,
        model_kwargs=mk,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
    )

    llm = _cached_make_llm(key_tuple)

    # The underlying SDK also needs the API key at runtime; ensure env is set.
    # (We don't pass api_key to ChatOpenAI directly so that caching remains stable.
    #  If you prefer explicit passing, you can add api_key=key to common_kwargs and
    #  include the key (or a hash) in the cache key—be mindful of logs/leaks.)
    os.environ["OPENAI_API_KEY"] = key

    return llm
