# openanalyst_studio/tools/visualization_tool.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import BaseTool
from tenacity import Retrying, stop_after_attempt, wait_exponential_jitter

from openanalyst_studio.tools.schemas import (
    ChartHints,
    ChartSpec,
    ChartType,
    DatasetContext,
    VisualizationInput,
    VisualizationOutput,
)
from openanalyst_studio.utils.logging_utils import timed, tool_logger


def _lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _safe_get(
    d: Optional[Dict[str, Any]], key: str, default: Any=None
) -> Any:
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def _classify_columns(ctx: DatasetContext) -> Tuple[List[str], List[str], List[str]]:
    """Return (categorical, numeric, temporal) columns based on dtypes/names."""
    dtypes = {k: _lower(v) for k, v in ctx.dtypes.items()} if isinstance(ctx.dtypes, dict) else {}
    cols = list(ctx.columns) if isinstance(ctx.columns, list) else []

    cats, nums, temps = [], [], []
    for c in cols:
        dt = dtypes.get(c, "")
        name = _lower(c)

        is_cat = any(k in dt for k in ("object", "category", "bool"))
        is_num = any(k in dt for k in ("int", "float", "number"))
        is_tmp = ("datetime" in dt) or any(w in name for w in ("date", "time", "month", "year"))

        if is_tmp:
            temps.append(c)
        elif is_num:
            nums.append(c)
        elif is_cat:
            cats.append(c)
        else:
            # fallback heuristics by name when dtype missing
            if any(w in name for w in ("region", "state", "city", "category", "segment", "brand", "type", "group")):
                cats.append(c)
            elif any(w in name for w in ("price", "cost", "revenue", "sales", "amount", "profit", "value", "units", "volume", "quantity")):
                nums.append(c)
            else:
                # Unknown → treat as categorical to enable bar/pie defaults
                cats.append(c)

    return cats, nums, temps


def _detect_intent(query_lc: str) -> Dict[str, bool]:
    """Lightweight intent flags from the query text."""
    return {
        "wants_trend": any(k in query_lc for k in ("trend", "over time", "by month", "monthly", "weekly", "daily", "year", "timeline")),
        "wants_distribution": any(k in query_lc for k in ("distribution", "histogram", "spread", "density")),
        "wants_composition": any(k in query_lc for k in ("share", "composition", "breakdown", "proportion", "market share", "percent", "ratio")),
        "wants_correlation": any(k in query_lc for k in ("correlation", "relationship", "scatter", "vs", "impact of")),
    }


def _apply_hints_first(
    hints: Optional[ChartHints],
) -> Tuple[Optional[ChartType], Optional[str], Optional[str], Optional[str]]:
    """Take any explicit hints and return a partial decision tuple."""
    if not isinstance(hints, ChartHints):
        try:
            # Attempt to coerce plain dict → ChartHints
            hints = ChartHints.model_validate(hints or {})
        except Exception:
            hints = None

    if not hints:
        return None, None, None, None

    return hints.chart_type, hints.x, hints.y, hints.title


def choose_chart_spec(
    query: str,
    dataset_context: DatasetContext,
    chart_hints: Optional[ChartHints]=None,
) -> ChartSpec:
    """
    Pure, testable chart selector:
      - Honors hints if provided (and fills any missing fields intelligently).
      - Uses dataset dtypes/column names to pick a sensible chart.
      - Keeps output compatible with ChartSpec validations.
    """
    q = _lower(query)
    cats, nums, temps = _classify_columns(dataset_context)
    intent = _detect_intent(q)

    hint_type, hint_x, hint_y, hint_title = _apply_hints_first(chart_hints)

    # 1) If hints specify a full, valid pair (type + x [+ y if required]), prefer them.
    if hint_type and hint_x:
        chart_type = hint_type
        x = hint_x
        # For charts that need Y, prefer hint_y or the first numeric
        if chart_type in {ChartType.bar, ChartType.line, ChartType.scatter, ChartType.box}:
            y = hint_y or (nums[0] if nums else None)
            if y is None:
                # fallback: if we don't have numeric y, choose pie to express counts by category
                chart_type = ChartType.pie
        else:
            y = hint_y  # pie/histogram can ignore y
        title = hint_title or (query[:80] if query else "Chart")
        return ChartSpec(chart_type=chart_type, x=x, y=y, title=title)

    # 2) No (or partial) hints → infer from intent + available columns.
    # Trends over time → line (temporal + numeric)
    if intent["wants_trend"] and temps and nums:
        return ChartSpec(
            chart_type=ChartType.line, x=temps[0], y=nums[0], title=(hint_title or query[:80])
        )

    # Explicit correlation / “vs” → scatter (numeric + numeric)
    if intent["wants_correlation"] and len(nums) >= 2:
        return ChartSpec(
            chart_type=ChartType.scatter, x=nums[0], y=nums[1], title=(hint_title or query[:80])
        )

    # Distribution → histogram (any numeric)
    if intent["wants_distribution"] and nums:
        return ChartSpec(
            chart_type=ChartType.histogram, x=nums[0], y=None, title=(hint_title or query[:80])
        )

    # Composition / share → pie if categorical present; fall back to bar if many categories + numeric
    if intent["wants_composition"] and cats:
        # If we do not know values, let the renderer count occurrences (pie allows y=None)
        return ChartSpec(chart_type=ChartType.pie, x=cats[0], y=None, title=(hint_title or query[:80]))

    # 3) Generic fallbacks by data shape
    # categorical + numeric → bar
    if cats and nums:
        return ChartSpec(chart_type=ChartType.bar, x=cats[0], y=nums[0], title=(hint_title or query[:80]))

    # temporal + numeric → line
    if temps and nums:
        return ChartSpec(chart_type=ChartType.line, x=temps[0], y=nums[0], title=(hint_title or query[:80]))

    # numeric + numeric → scatter
    if len(nums) >= 2:
        return ChartSpec(chart_type=ChartType.scatter, x=nums[0], y=nums[1], title=(hint_title or query[:80]))

    # single numeric only → histogram
    if len(nums) == 1:
        return ChartSpec(chart_type=ChartType.histogram, x=nums[0], y=None, title=(hint_title or query[:80]))

    # only categorical → pie (counts)
    if cats:
        return ChartSpec(chart_type=ChartType.pie, x=cats[0], y=None, title=(hint_title or query[:80]))

    # last resort: synthetic category name (frontends may remap later)
    return ChartSpec(chart_type=ChartType.pie, x="category", y=None, title=(hint_title or (query[:80] or "Chart")))


class VisualizationTool(BaseTool):
    """
    LangChain tool that returns a JSON-serialized VisualizationOutput containing a ChartSpec.
    It is robust to partial/missing context and will always return a valid spec.
    """
    name: str = "visualization"
    description: str = "Create a ChartSpec JSON using dataset_context."
    args_schema: type[VisualizationInput] = VisualizationInput
    return_direct: bool = False

    max_examples: int = 50
    max_retries: int = 2

    def _coerce_ctx(self, ctx_like: Any) -> DatasetContext:
        # Accept dict / DatasetContext interchangeably
        if isinstance(ctx_like, DatasetContext):
            return ctx_like
        try:
            return DatasetContext.model_validate(ctx_like or {})
        except Exception:
            return DatasetContext()  # empty context

    def _coerce_hints(self, hints_like: Any) -> Optional[ChartHints]:
        if hints_like is None:
            return None
        if isinstance(hints_like, ChartHints):
            return hints_like
        try:
            return ChartHints.model_validate(hints_like)
        except Exception:
            return None

    def _choose_spec(
        self,
        query: str,
        dataset_context: Any,  # dict or DatasetContext
        chart_hints: Any | None,
    ) -> ChartSpec:
        ctx = self._coerce_ctx(dataset_context)
        hints = self._coerce_hints(chart_hints)
        return choose_chart_spec(query=query, dataset_context=ctx, chart_hints=hints)

    def _run(self, query, dataset_context, chart_hints=None, run_manager=None) -> str:
        log = tool_logger(self.name).bind(query=(query or "")[:100])
        with timed(log, action="choose_chart"):
            for attempt in Retrying(
                stop=stop_after_attempt(self.max_retries + 1),
                wait=wait_exponential_jitter(initial=0.2, max=2.0),
                reraise=True,
            ):
                with attempt:
                    spec = self._choose_spec(query, dataset_context, chart_hints)
                    out = VisualizationOutput(spec=spec)
                    # Return a compact JSON for tool-to-tool interop
                    return out.model_dump_json()

    async def _arun(self, *args, **kwargs):
        return self._run(*args, **kwargs)
