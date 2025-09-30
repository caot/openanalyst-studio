# openanalyst_studio/tools/analysis.py
from __future__ import annotations

import json
from typing import Any, Dict

from langchain.tools import tool

# Re-export the class-based tool instance to keep backwards compatibility:
#   from openanalyst_studio.tools.analysis import generate_insights_tool
# now returns the instance defined in analysis_tool.py
from openanalyst_studio.tools.analysis_tool import generate_insights_tool  # noqa: F401

# Prefer package-local prompt; fall back to a minimal one if unavailable.
try:
    from openanalyst_studio.prompts import INSIGHTS_ANALYSIS_PROMPT as INSIGHT_GENERATION_PROMPT  # type: ignore
except Exception:  # pragma: no cover
    INSIGHT_GENERATION_PROMPT = (
        "Generate business insights based on the data and visualization. Provide 3-5 key insights."
    )

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def _parse_kv_input(input_text: str) -> Dict[str, str]:
    """
    Parse strings like:
      "query: What drives sales? | df_summary: {...} | chart_context: bar over region"
    into a dict {'query': 'What drives sales?', 'df_summary': '{...}', 'chart_context': '...'}.
    Robust to missing sections and extra whitespace.
    """
    out: Dict[str, str] = {}
    if not isinstance(input_text, str):
        return out
    for chunk in input_text.split("|"):
        part = chunk.strip()
        if not part:
            continue
        if ":" not in part:
            # If someone passes only the query, treat whole text as query once.
            if "query" not in out and not out:
                out["query"] = part
            continue
        key, value = part.split(":", 1)
        out[key.strip().lower()] = value.strip()
    if "query" not in out:
        out["query"] = input_text.strip()
    return out

# ------------------------------------------------------------------------------
# Lightweight tools (string-in -> JSON string-out)
# ------------------------------------------------------------------------------


@tool
def create_correlation_analysis(input_text: str) -> str:
    """
    Return instructions to create a correlation heatmap for numeric variables.
    Input: "query: [question] | df_summary: [summary]"
    Output: JSON string with analysis_type='correlation', chart_type='heatmap'.
    """
    try:
        parsed = _parse_kv_input(input_text)
        query = parsed.get("query", input_text.strip())
        payload = {
            "tool_used": "correlation_analysis",
            "analysis_type": "correlation",
            "chart_type": "heatmap",
            "message": "Create correlation heatmap of all numeric variables.",
            "success": True,
            "query": query,
        }
        return json.dumps(payload)
    except Exception as e:  # pragma: no cover
        return json.dumps({"success": False, "error": f"Error in correlation analysis: {e}"})


@tool
def create_data_summary_tool(input_text: str) -> str:
    """
    Signal that the agent should rely on the already-provided comprehensive dataset summary.
    Useful as a no-op acknowledgment in flows that expect a 'summary' step.
    """
    try:
        payload = {
            "tool_used": "create_data_summary",
            "analysis_type": "summary",
            "message": (
                "Data summary analysis acknowledged. Use the comprehensive dataset summary "
                "provided in the agent input for detailed statistics."
            ),
            "success": True,
        }
        return json.dumps(payload)
    except Exception as e:  # pragma: no cover
        return json.dumps({"success": False, "error": f"Error in data summary tool: {e}"})


@tool
def create_chart_context_tool(input_text: str) -> str:
    """
    Build a compact chart context JSON from a simple KV string:
      "chart_type: bar | x_col: Region | y_col: Total Sales"
    """
    try:
        parsed = _parse_kv_input(input_text)
        chart_type = parsed.get("chart_type", "unknown")
        x_col = parsed.get("x_col", "")
        y_col = parsed.get("y_col", "")

        context = f"Chart Type: {chart_type}\n"
        if x_col:
            context += f"X-axis: {x_col}\n"
        if y_col:
            context += f"Y-axis: {y_col}\n"

        payload = {
            "tool_used": "create_chart_context",
            "chart_context": context.strip(),
            "message": "Chart context prepared for analysis.",
            "success": True,
        }
        return json.dumps(payload)
    except Exception as e:  # pragma: no cover
        return json.dumps({"success": False, "error": f"Error in chart context tool: {e}"})

# ------------------------------------------------------------------------------
# Optional textual helpers (not LangChain tools)
# ------------------------------------------------------------------------------


def create_data_summary(df) -> str:
    """
    Create a concise, human-readable dataset summary for LLM context.
    (Not a LangChain tool; just a utility used elsewhere.)
    """
    try:
        import pandas as pd  # local import to avoid hard dependency at tool import time
        if not hasattr(df, "columns"):
            return "Invalid DataFrame"

        lines = [
            "Dataset Overview:",
            f"- {len(df):,} rows, {len(df.columns)} columns",
            f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
            "",
            "Column Information:",
        ]

        for col in df.columns:
            s = df[col]
            dtype = str(s.dtype)
            non_null = int(s.count())
            unique_vals = int(s.nunique(dropna=True))
            lines.append(f"- {col}: {dtype}, {non_null} non-null, {unique_vals} unique")

            if pd.api.types.is_object_dtype(s):
                top = s.value_counts(dropna=True).head(3).index.tolist()
                lines.append(f"  Top values: {top}")
            elif pd.api.types.is_numeric_dtype(s):
                try:
                    lines.append(f"  Range: {float(s.min()):.2f} to {float(s.max()):.2f}")
                except Exception:
                    pass

        return "\n".join(lines)
    except Exception as e:  # pragma: no cover
        return f"Error creating data summary: {e}"


def create_chart_context(chart_type: str, decision: Dict[str, Any]) -> str:
    """
    Compose a small chart context string from a decision dict.
    """
    try:
        parts = [f"Chart Type: {chart_type}"]
        x = decision.get("x_column")
        y = decision.get("y_column")
        agg = decision.get("agg_func")
        if x:
            parts.append(f"X-axis: {x}")
        if y:
            parts.append(f"Y-axis: {y}")
        if agg:
            parts.append(f"Aggregation: {agg}")
        return "\n".join(parts)
    except Exception as e:  # pragma: no cover
        return f"Error creating chart context: {e}"


__all__ = [
    # class-based tool instance re-export
    "generate_insights_tool",
    # function tools
    "create_correlation_analysis",
    "create_data_summary_tool",
    "create_chart_context_tool",
    # helpers
    "create_data_summary",
    "create_chart_context",
]
