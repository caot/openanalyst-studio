# openanalyst_studio/tools/analysis_tool.py
from __future__ import annotations

import json
from typing import Any

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field
from tenacity import Retrying, stop_after_attempt, wait_exponential_jitter

from openanalyst_studio.tools.llm import get_llm
from openanalyst_studio.tools.schemas import AnalysisInput, AnalysisOutput
from openanalyst_studio.utils.logging_utils import timed, tool_logger

SYSTEM_PROMPT = (
    "You are a concise business analyst. Use the provided data_summary to answer clearly.\n"
    "Return ONLY a JSON object with keys: `answer` (string), `evidence` (array of strings).\n"
    "No markdown, no extra prose."
)


def _compact_summary(summary: dict[str, Any], max_chars: int=8000) -> dict[str, Any]:
    """
    Reduce summary size to avoid hitting model/context limits:
    - Drop very large arrays and keep short samples
    - Convert non-serializable types to strings defensively
    """
    try:
        payload = json.dumps(summary, default=str)
        if len(payload) <= max_chars:
            return summary
    except Exception:
        # If summary isn't fully serializable, fall through to compaction
        pass

    compact: dict[str, Any] = {}
    for k, v in summary.items():
        try:
            if isinstance(v, (list, tuple)):
                compact[k] = v[:5]  # keep a short head
            elif isinstance(v, dict):
                # keep a few keys; also try to compress nested large lists
                trimmed = {}
                for i, (kk, vv) in enumerate(v.items()):
                    if i >= 15:
                        break
                    if isinstance(vv, (list, tuple)):
                        trimmed[kk] = vv[:5]
                    else:
                        trimmed[kk] = vv
                compact[k] = trimmed
            else:
                compact[k] = v
        except Exception:
            compact[k] = str(v)

    # Ensure it fits; if not, put a final hard cap on serialized size
    try:
        s = json.dumps(compact, default=str)
        if len(s) > max_chars:
            compact = {"truncated": True, "keys": list(compact.keys())[:20]}
    except Exception:
        compact = {"truncated": True}
    return compact


def _extract_json(text: str) -> dict | None:
    """
    Robustly parse a JSON object from model text. Supports:
    - Raw JSON
    - Fenced ```json blocks
    - First {...} slice if mixed prose sneaks in
    """
    if not text:
        return None
    t = text.strip()
    # fenced json
    if "```json" in t:
        try:
            t = t.split("```json", 1)[1].split("```", 1)[0].strip()
            return json.loads(t)
        except Exception:
            pass
    if "```" in t:
        try:
            t = t.split("```", 1)[1].split("```", 1)[0].strip()
            return json.loads(t)
        except Exception:
            pass
    # raw or embedded
    try:
        if t.startswith("{") and t.endswith("}"):
            return json.loads(t)
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(t[start: end + 1])
    except Exception:
        return None
    return None


class AnalysisTool(BaseTool):
    """
    Generate concise business insights from a provided data summary.

    Returns a JSON string:
      {"ok": true, "answer": "...", "evidence": ["..."]}

    Notes:
    - Uses OpenAI JSON mode when available (falls back gracefully).
    - Keeps the public tool name as 'generate_insights_tool' for prompt compatibility.
    """
    # Keep prompt compatibility without editing COMPREHENSIVE_AGENT_PROMPT
    name: str = "generate_insights_tool"
    description: str = "Generate concise business insights using data_summary."
    args_schema: type[AnalysisInput] = AnalysisInput
    return_direct: bool = False

    # Tunables (pydantic fields so you can override at instantiation)
    temperature: float = Field(default=0.2, description="LLM temperature")
    timeout_s: int = Field(default=120, description="Request timeout (seconds)")
    max_retries: int = Field(default=2, description="Max retries with backoff")
    use_json_mode: bool = Field(default=True, description="Prefer JSON mode (OpenAI JSON object)")
    stream_tokens: bool = Field(default=False, description="Stream tokens (not used with JSON mode)")
    max_summary_chars: int = Field(default=8000, description="Hard cap for summary payload")

    # ---- internals ----
    def _llm(self):
        """
        Create an LLM client, preferring JSON mode; fall back if unsupported by the model/client.
        """
        if self.use_json_mode:
            try:
                return get_llm(
                    temperature=self.temperature,
                    timeout=self.timeout_s,
                    streaming=False,
                    response_format={"type": "json_object"},
                )
            except TypeError:
                # Older clients may not support response_format
                pass
        return get_llm(
            temperature=self.temperature,
            timeout=self.timeout_s,
            streaming=(False if self.use_json_mode else self.stream_tokens),
        )

    def _infer(
        self,
        question: str,
        data_summary: dict[str, Any],
        run_manager: CallbackManagerForToolRun | None,
    ) -> str:
        llm = self._llm()
        compact = _compact_summary(data_summary or {}, max_chars=self.max_summary_chars)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps({"question": question, "data_summary": compact}, default=str)),
        ]

        # 1) Prefer single-shot (JSON mode or plain)
        try:
            resp = llm.invoke(messages)
            text = getattr(resp, "content", None) or str(resp)
        except Exception as _e:
            # 2) Optional: streaming fallback when not in JSON mode
            if self.use_json_mode:
                # If JSON mode call failed, try plain mode once
                llm = get_llm(temperature=self.temperature, timeout=self.timeout_s, streaming=False)
                resp = llm.invoke(messages)
                text = getattr(resp, "content", None) or str(resp)
            else:
                chunks: list[str] = []
                for ch in llm.stream(messages):
                    token = getattr(ch, "content", "") or ""
                    if token and run_manager:
                        run_manager.on_text(token, verbose=False)
                    chunks.append(token)
                text = "".join(chunks)

        # Parse model output -> AnalysisOutput
        payload = _extract_json(text) or {}
        try:
            out = AnalysisOutput(
                answer=str(payload.get("answer", "")).strip() or text.strip(),
                evidence=[str(x) for x in (payload.get("evidence") or [])],
            ).model_dump()
        except Exception:
            out = AnalysisOutput(answer=text.strip(), evidence=[]).model_dump()

        return json.dumps(out)

    # LangChain sync entrypoint
    def _run(
        self,
        question: str,
        data_summary: dict[str, Any],
        chart_context: dict[str, Any] | None=None,  # accepted but unused
        run_manager: CallbackManagerForToolRun | None=None,
    ) -> str:
        log = tool_logger(self.name).bind(question=(question or "")[:120])
        with timed(log, action="analysis"):
            for attempt in Retrying(
                stop=stop_after_attempt(self.max_retries + 1),
                wait=wait_exponential_jitter(initial=0.5, max=3.0),
                reraise=True,
            ):
                with attempt:
                    return self._infer(question, data_summary, run_manager)

    # LangChain async entrypoint
    async def _arun(self, *args, **kwargs) -> str:  # pragma: no cover
        return self._run(*args, **kwargs)


# Optional: export a ready-to-use instance for convenience/back-compat imports
# e.g., from openanalyst_studio.tools.analysis_tool import generate_insights_tool
generate_insights_tool = AnalysisTool()
