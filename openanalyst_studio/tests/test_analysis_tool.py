# tools/analysis_tool.py

from __future__ import annotations

import json
from typing import Any

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import Retrying, stop_after_attempt, wait_exponential_jitter

from openanalyst_studio.tools.llm import get_llm
from openanalyst_studio.tools.schemas import AnalysisInput, AnalysisOutput
from openanalyst_studio.utils.logging_utils import timed, tool_logger

SYSTEM = (
    "You are a concise business analyst. Use the provided data_summary to answer clearly.\n"
    "Return ONLY a JSON object with keys: `answer` (string), `evidence` (array of strings).\n"
    "No markdown, no extra prose."
)


class AnalysisTool(BaseTool):
    name: str = "analysis"
    description: str = "Generate concise business insights using data_summary."
    args_schema: type[AnalysisInput] = AnalysisInput
    return_direct: bool = False

    temperature: float = 0.2
    timeout_s: int = 120
    max_retries: int = 2
    use_json_mode: bool = True
    stream_tokens: bool = False

    def _llm(self):
        if self.use_json_mode:
            return get_llm(
                temperature=self.temperature,
                timeout=self.timeout_s,
                streaming=False,
                response_format={"type": "json_object"},
            )
        return get_llm(temperature=self.temperature, timeout=self.timeout_s, streaming=self.stream_tokens)

    def _infer(self, question: str, data_summary: dict[str, Any], run_manager: CallbackManagerForToolRun | None):
        llm = self._llm()
        messages = [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=json.dumps({"question": question, "data_summary": data_summary})),
        ]

        # <- always compute a default evidence from the summary
        default_ev = list((data_summary or {}).get("business_metrics_statistics", {}).keys())[:3]

        if self.use_json_mode:
            # Non-streaming, JSON mode
            resp = llm.invoke(messages)
            text = (resp.content or "").strip()
            try:
                obj = json.loads(text)
                answer = str(obj.get("answer", "")).strip()
                evidence = obj.get("evidence", default_ev)
                out = AnalysisOutput(answer=answer, evidence=[str(x) for x in evidence]).model_dump()
            except Exception:
                # Fallback: keep the raw text but still provide default evidence
                out = AnalysisOutput(answer=text, evidence=default_ev).model_dump()
            return json.dumps(out)

        # Streaming (plain text) path: DO NOT attempt JSON parse
        chunks: list[str] = []
        for ch in llm.stream(messages):
            token = getattr(ch, "content", "") or ""
            if token:
                chunks.append(token)
                if run_manager:
                    run_manager.on_text(token, verbose=False)

        answer = "".join(chunks).strip()
        out = AnalysisOutput(answer=answer, evidence=default_ev).model_dump()
        return json.dumps(out)

    def _run(self, question, data_summary, chart_context=None, run_manager: CallbackManagerForToolRun | None=None) -> str:
        log = tool_logger(self.name).bind(question=question[:80])
        with timed(log, action="analysis"):
            for attempt in Retrying(
                stop=stop_after_attempt(self.max_retries + 1),
                wait=wait_exponential_jitter(initial=0.5, max=3.0),
                reraise=True,
            ):
                with attempt:
                    return self._infer(question, data_summary, run_manager)

    async def _arun(self, *args, **kwargs):
        return self._run(*args, **kwargs)
