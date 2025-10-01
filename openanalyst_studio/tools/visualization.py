# openanalyst_studio/tools/visualization.py
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from typing import Any

import pandas as pd
import plotly.express as px
from langchain.tools import tool

from .llm import get_llm

try:
    import os as _os
    import sys as _sys

    _sys.path.append(_os.path.dirname(_os.path.dirname(__file__)))
    from ..prompts import CHART_DECISION_PROMPT  # noqa: E402
except Exception:  # pragma: no cover
    CHART_DECISION_PROMPT = (
        "You are a data visualization expert. Use EXACT column names from dataset. "
        "Return JSON with chart_type, x_column, y_column, title, agg_func, reasoning."
    )

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _fmt_large(v: Any) -> Any:
    if isinstance(v, (int, float)):
        a = abs(v)
        if a >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if a >= 1_000:
            return f"{v/1_000:.1f}K"
    return v


def _safe_str(x: Any) -> str:
    return x if isinstance(x, str) else ("" if x is None else str(x))


def _sanitize_json_block(text: str) -> str:
    """Strip code fences and trim."""
    t = _safe_str(text).strip()
    if "```" in t:
        # try json fence first
        if "```json" in t:
            t = t.split("```json", 1)[-1]
        t = t.split("```", 1)[-1]
    return t.strip()


def _parse_df_info(df_info: str) -> dict[str, Any]:
    """Parse df_info string → dict with robust fallbacks."""
    s = _safe_str(df_info).strip()
    if not s:
        return {"columns": [], "dtypes": {}, "sample": {}}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            s2 = s.replace("'", '"').replace("True", "true").replace("False", "false")
            return json.loads(s2)
        except Exception:
            try:
                import ast
                return ast.literal_eval(s)
            except Exception:
                # minimal safe fallback
                return {"columns": [], "dtypes": {}, "sample": {}}


def _ensure_numeric(df: pd.DataFrame, col: str) -> str | None:
    """If column isn't numeric, try a clean-to-numeric conversion copy; return a temp name if created."""
    if col not in df.columns:
        return None
    if pd.api.types.is_numeric_dtype(df[col]):
        return col
    # try convert
    series = (
        df[col]
        .astype(str)
        .str.replace(r"[^\d\.\-]", "", regex=True)
        .replace("", pd.NA)
    )
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().sum() > max(1, int(0.5 * len(df))):  # >50% convertible
        tmp = f"__num__{col}"
        df[tmp] = num
        return tmp
    return None


def _ensure_datetime(df: pd.DataFrame, col: str) -> str | None:
    if col not in df.columns:
        return None
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return col
    try:
        dt = pd.to_datetime(df[col], errors="coerce")
        if dt.notna().sum() > 0:
            tmp = f"__dt__{col}"
            df[tmp] = dt
            return tmp
    except Exception:
        pass
    return None


def _best_match(available: Iterable[str], target: str | None) -> str | None:
    if not target:
        return None
    t = target.lower()
    # exact case-insensitive
    for c in available:
        if c.lower() == t:
            return c
    # partial
    for c in available:
        cl = c.lower()
        if t in cl or cl in t:
            return c
    # semantic-ish
    words = t.replace("_", " ").split()
    for c in available:
        cl = c.lower().replace("_", " ")
        for w in words:
            if w and (w in cl):
                return c
    # typed heuristics by keywords
    revenue_words = ("sales", "revenue", "amount", "total", "value", "income", "profit")
    if any(w in t for w in revenue_words):
        for c in available:
            cl = c.lower()
            if any(w in cl for w in revenue_words):
                return c
    time_words = ("date", "time", "month", "year", "day", "created", "updated")
    if any(w in t for w in time_words):
        for c in available:
            cl = c.lower()
            if any(w in cl for w in time_words):
                return c
    return None


def _title_from_rules(chart_type: str, x: str, y: str | None, agg: str | None) -> str:
    if chart_type == "histogram":
        return f"Distribution of {x}"
    if chart_type == "pie":
        return f"{x} Composition"
    if chart_type == "line":
        return f"{_safe_str(y) or 'Value'} over {x}"
    if chart_type == "scatter":
        return f"{_safe_str(y)} vs {x}"
    if chart_type == "box":
        return f"{_safe_str(y)} by {x}"
    # bar
    if agg and agg != "none" and y:
        return f"{agg.title()} {y} by {x}"
    return f"{_safe_str(y) or 'Count'} by {x}"


def _maybe_llm_title(chart_type: str, x_col: str, y_col: str | None, df: pd.DataFrame, agg_func: str | None) -> str | None:
    """Optional LLM title. Disabled by default unless OA_USE_LLM_TITLES=1."""
    if os.getenv("OA_USE_LLM_TITLES", "0") not in ("1", "true", "True"):
        return None
    try:
        llm = get_llm(response_format={"type": "text"})  # plain text
        sample = df.head(3).to_dict("records")
        ds = f"Rows={len(df)}, Columns={list(df.columns)}"
        prompt = (
            "You are a BI expert. Generate a concise (≤8 words), professional chart title.\n"
            f"Type: {chart_type}\nX: {x_col}\nY: {y_col}\nAggregation: {agg_func or 'none'}\n"
            f"Dataset: {ds}\nSample: {sample}\n"
            "Output: title only (no quotes, no punctuation at end)."
        )
        resp = llm.invoke(prompt)
        title = getattr(resp, "content", None) or str(resp)
        title = _safe_str(title).strip().strip('"').strip("'")
        return title if 0 < len(title) <= 60 else None
    except Exception as e:  # pragma: no cover
        log.debug("llm_title_failed", exc=str(e))
        return None


def _apply_layout(fig, title: str, x_title: str, y_title: str):
    fig.update_layout(
        template="plotly_white",
        title={"text": f"<b>{title}</b>", "x": 0.5, "xanchor": "center", "font": {"size": 16}},
        xaxis_title=f"<b>{x_title}</b>",
        yaxis_title=f"<b>{y_title}</b>",
        font={"size": 11},
        height=500,
        margin=dict(l=60, r=40, t=80, b=80),
    )
    fig.update_xaxes(
        linecolor="black",
        linewidth=0.2,
        gridwidth=1,
        gridcolor="lightgray",
        tickangle=45,
        tickfont_size=10,
        title_font_size=12,
    )
    fig.update_yaxes(linecolor="black", linewidth=0.2, gridwidth=1, gridcolor="lightgray")
    return fig

# ---------------------------------------------------------------------
# Chart creators (return plotly Figure or None)
# ---------------------------------------------------------------------


def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str | None, title: str | None, agg_func: str="sum"):
    try:
        if not x_col:
            return None
        if y_col is None or agg_func == "count":
            data = df[x_col].value_counts().reset_index()
            data.columns = [x_col, "count"]
            y_col_name = "count"
        else:
            if agg_func not in {"sum", "mean", "count", "max", "min"}:
                agg_func = "sum"
            # coerce y to numeric if needed
            y_fixed = _ensure_numeric(df, y_col) or y_col
            data = df.groupby(x_col, dropna=False)[y_fixed].agg(agg_func).reset_index()
            y_col_name = y_col

        # limit categories to top 12 for readability
        if len(data) > 12 and y_col_name in data.columns:
            data = data.nlargest(12, y_col_name)
            title = f"{_safe_str(title)} (Top 12)"

        data = data.sort_values(y_col_name, ascending=False)
        fig = px.bar(data, x=x_col, y=y_col_name, title=title or _title_from_rules("bar", x_col, y_col, agg_func),
                     color=y_col_name, color_continuous_scale="viridis")

        xt = x_col.replace("_", " ").title()
        yt = (y_col or "Count").replace("_", " ").title()
        fig = _apply_layout(fig, fig.layout.title.text or "", xt, yt)

        fig.update_traces(
            texttemplate="%{y:,.0f}",
            textposition="outside",
            marker_line_width=0.5,
            marker_line_color="white",
            hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>",
        )
        return fig
    except Exception as e:
        log.exception("bar_chart_failed", exc=str(e))
        return None


def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str | None):
    try:
        if not x_col or not y_col:
            return None

        # ensure temporal x
        x_fixed = _ensure_datetime(df, x_col) or x_col
        # ensure numeric y
        y_fixed = _ensure_numeric(df, y_col) or y_col

        df_sorted = df.sort_values(x_fixed)
        fig = px.line(df_sorted, x=x_fixed, y=y_fixed, title=title or _title_from_rules("line", x_col, y_col, None))

        xt = x_col.replace("_", " ").title()
        yt = y_col.replace("_", " ").title()
        fig = _apply_layout(fig, fig.layout.title.text or "", xt, yt)

        fig.update_traces(
            line=dict(width=3, color="#2E86AB"),
            marker=dict(size=6, color="#2E86AB"),
            hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>",
        )

        # annotate min/max if possible
        yvals = pd.to_numeric(df_sorted[y_fixed], errors="coerce").dropna()
        if not yvals.empty:
            i_max = yvals.idxmax()
            i_min = yvals.idxmin()
            fig.add_annotation(
                x=df_sorted.loc[i_max, x_fixed],
                y=df_sorted.loc[i_max, y_fixed],
                text=f"Max: {_fmt_large(df_sorted.loc[i_max, y_fixed])}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                arrowwidth=2,
                bgcolor="white",
                bordercolor="red",
            )
            fig.add_annotation(
                x=df_sorted.loc[i_min, x_fixed],
                y=df_sorted.loc[i_min, y_fixed],
                text=f"Min: {_fmt_large(df_sorted.loc[i_min, y_fixed])}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="blue",
                arrowwidth=2,
                bgcolor="white",
                bordercolor="blue",
            )
        return fig
    except Exception as e:
        log.exception("line_chart_failed", exc=str(e))
        return None


def create_pie_chart(df: pd.DataFrame, category_col: str, value_col: str | None, title: str | None):
    try:
        if not category_col:
            return None
        if value_col:
            # ensure numeric values
            v_fixed = _ensure_numeric(df, value_col) or value_col
            agg = df.groupby(category_col, dropna=False)[v_fixed].sum().reset_index()
            agg = agg.sort_values(v_fixed, ascending=False)
            if len(agg) > 12:
                top = agg.head(11)
                others_sum = agg.iloc[11:][v_fixed].sum()
                agg = pd.concat([top, pd.DataFrame({category_col: ["Others"], v_fixed: [others_sum]})], ignore_index=True)
            names = agg[category_col]
            values = agg[v_fixed]
        else:
            vc = df[category_col].value_counts(dropna=False)
            if len(vc) > 12:
                top = vc.head(11)
                others = vc.iloc[11:].sum()
                vc = top.append(pd.Series({"Others": others}))
            names = vc.index
            values = vc.values

        fig = px.pie(values=values, names=names, title=title or _title_from_rules("pie", category_col, value_col, None))
        fig.update_layout(
            template="plotly_white",
            title={"text": f"<b>{fig.layout.title.text}</b>", "x": 0.5, "xanchor": "center", "font": {"size": 18}},
            font={"size": 12},
            height=600,
            margin=dict(l=60, r=60, t=80, b=60),
            showlegend=True,
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont_size=11,
            marker=dict(line=dict(color="white", width=2)),
            hovertemplate="<b>%{label}</b><br>Value: %{value}<br>Percentage: %{percent}<extra></extra>",
        )
        return fig
    except Exception as e:
        log.exception("pie_chart_failed", exc=str(e))
        return None


def create_histogram(df: pd.DataFrame, column: str, title: str | None, bins: int=50):
    try:
        if not column:
            return None
        x_fixed = _ensure_numeric(df, column) or column
        fig = px.histogram(
            df, x=x_fixed, title=title or _title_from_rules("histogram", column, "Frequency", None), nbins=bins,
            color_discrete_sequence=["#2E86AB"],
        )
        xt = column.replace("_", " ").title()
        fig = _apply_layout(fig, fig.layout.title.text or "", xt, "Frequency")
        fig.update_traces(
            marker_line_width=1,
            marker_line_color="white",
            opacity=0.75,
            hovertemplate="<b>Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>",
        )
        fig.update_layout(bargap=0.1)
        return fig
    except Exception as e:
        log.exception("histogram_failed", exc=str(e))
        return None


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str | None):
    try:
        if not x_col or not y_col:
            return None
        x_fixed = _ensure_numeric(df, x_col) or x_col
        y_fixed = _ensure_numeric(df, y_col) or y_col

        fig = px.scatter(
            df,
            x=x_fixed,
            y=y_fixed,
            title=title or _title_from_rules("scatter", x_col, y_col, None),
            color_discrete_sequence=["#2E86AB"],
            opacity=0.7,
        )
        xt = x_col.replace("_", " ").title()
        yt = y_col.replace("_", " ").title()
        fig = _apply_layout(fig, fig.layout.title.text or "", xt, yt)

        # optional trendline (requires statsmodels)
        try:
            fig_tr = px.scatter(df, x=x_fixed, y=y_fixed, trendline="ols")
            if len(fig_tr.data) > 1:
                fig.add_traces(fig_tr.data[1:])
                fig.update_traces(line=dict(width=3, color="red", dash="dash"), selector=dict(mode="lines"))
        except Exception:
            pass

        fig.update_traces(
            marker=dict(size=8, line=dict(width=1, color="white"), opacity=0.7),
            hovertemplate="<b>%{x}</b>, <b>%{y}</b><extra></extra>",
            selector=dict(mode="markers"),
        )
        return fig
    except Exception as e:
        log.exception("scatter_failed", exc=str(e))
        return None


def create_box_plot(df: pd.DataFrame, category_col: str, value_col: str, title: str | None):
    try:
        if not category_col or not value_col:
            return None
        v_fixed = _ensure_numeric(df, value_col) or value_col
        fig = px.box(
            df,
            x=category_col,
            y=v_fixed,
            title=title or _title_from_rules("box", category_col, value_col, None),
            color=category_col,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        xt = category_col.replace("_", " ").title()
        yt = value_col.replace("_", " ").title()
        fig = _apply_layout(fig, fig.layout.title.text or "", xt, yt)
        fig.update_layout(showlegend=False)
        fig.update_traces(
            marker_outliercolor="red",
            marker_line_width=2,
            line_width=2,
            hovertemplate="<b>%{x}</b><br>Value: %{y}<extra></extra>",
        )
        return fig
    except Exception as e:
        log.exception("box_plot_failed", exc=str(e))
        return None


CHART_MAPPING = {
    "bar": create_bar_chart,
    "line": create_line_chart,
    "pie": create_pie_chart,
    "histogram": create_histogram,
    "scatter": create_scatter_plot,
    "box": create_box_plot,
}

# ---------------------------------------------------------------------
# LangChain tools (stringified JSON outputs)
# ---------------------------------------------------------------------


@tool
def create_bar_chart_tool(input_text: str) -> str:
    """Create professional bar chart parameters from `input_text`.
    Input: "x_col: [col] | y_col: [col] | title: [title] | agg_func: [sum|count|mean]"
    Returns: JSON string describing the prepared bar-chart parameters.
    """
    try:
        return json.dumps({
            "tool_used": "create_bar_chart",
            "chart_type": "bar",
            "message": "Bar chart parameters prepared",
            "instruction": "Use create_visualization_tool for comprehensive chart creation with LLM guidance",
            "success": True,
        })
    except Exception as e:  # pragma: no cover
        return json.dumps({"error": f"Error in bar chart tool: {e}"})


@tool
def create_line_chart_tool(input_text: str) -> str:
    """Prepare line chart parameters from `input_text`.
    Input format: "x_col: [column] | y_col: [column] | title: [title]"
    Returns a JSON string describing the prepared line-chart parameters.
    """
    try:
        return json.dumps(
            {
                "tool_used": "create_line_chart",
                "chart_type": "line",
                "message": "Line chart parameters prepared",
                "instruction": "Use create_visualization_tool for comprehensive chart creation with LLM guidance",
                "success": True,
            }
        )
    except Exception as e:  # pragma: no cover
        return json.dumps({"error": f"Error in line chart tool: {e}"})


@tool
def create_pie_chart_tool(input_text: str) -> str:
    """Prepare pie chart parameters from `input_text`.
    Input format: "category_col: [column] | value_col: [column] | title: [title]"
    Returns a JSON string describing the prepared pie-chart parameters.
    """
    try:
        return json.dumps(
            {
                "tool_used": "create_pie_chart",
                "chart_type": "pie",
                "message": "Pie chart parameters prepared",
                "instruction": "Use create_visualization_tool for comprehensive chart creation with LLM guidance",
                "success": True,
            }
        )
    except Exception as e:  # pragma: no cover
        return json.dumps({"error": f"Error in pie chart tool: {e}"})


@tool
def create_histogram_tool(input_text: str) -> str:
    """Prepare histogram parameters from `input_text`.
    Input format: "column: [column] | title: [title] | bins: [number]"
    Returns a JSON string describing the prepared histogram parameters.
    """
    try:
        return json.dumps(
            {
                "tool_used": "create_histogram",
                "chart_type": "histogram",
                "message": "Histogram parameters prepared",
                "instruction": "Use create_visualization_tool for comprehensive chart creation with LLM guidance",
                "success": True,
            }
        )
    except Exception as e:  # pragma: no cover
        return json.dumps({"error": f"Error in histogram tool: {e}"})


@tool
def create_scatter_plot_tool(input_text: str) -> str:
    """Prepare scatter plot parameters from `input_text`.
    Input format: "x_col: [column] | y_col: [column] | title: [title]"
    Returns a JSON string describing the prepared scatter-plot parameters.
    """
    try:
        return json.dumps(
            {
                "tool_used": "create_scatter_plot",
                "chart_type": "scatter",
                "message": "Scatter plot parameters prepared",
                "instruction": "Use create_visualization_tool for comprehensive chart creation with LLM guidance",
                "success": True,
            }
        )
    except Exception as e:  # pragma: no cover
        return json.dumps({"error": f"Error in scatter plot tool: {e}"})


@tool
def create_box_plot_tool(input_text: str) -> str:
    """Prepare box plot parameters from `input_text`.
    Input format: "category_col: [column] | value_col: [column] | title: [title]"
    Returns a JSON string describing the prepared box-plot parameters.
    """
    try:
        return json.dumps(
            {
                "tool_used": "create_box_plot",
                "chart_type": "box",
                "message": "Box plot parameters prepared",
                "instruction": "Use create_visualization_tool for comprehensive chart creation with LLM guidance",
                "success": True,
            }
        )
    except Exception as e:  # pragma: no cover
        return json.dumps({"error": f"Error in box plot tool: {e}"})


@tool
def create_visualization_tool(input_text: str) -> str:
    """
    Decide a chart from: "query: [question] | df_info: [json]"
    Always returns STRINGIFIED JSON with keys:
      chart_type, x_column, y_column, title, agg_func, reasoning
    """
    try:
        # 1) Split schema
        query = input_text
        df_info = '{"columns": [], "dtypes": {}, "sample": {}}'
        if " | df_info: " in input_text:
            parts = input_text.split(" | df_info: ", 1)
            query = parts[0].replace("query:", "", 1).strip()
            df_info = parts[1].strip()

        df_data = _parse_df_info(df_info)

        # 2) Call LLM in JSON mode (safer)
        llm = get_llm(response_format={"type": "json_object"})
        prompt = CHART_DECISION_PROMPT.format(
            columns=df_data.get("columns", []),
            dtypes=str(df_data.get("dtypes", {})),
            sample=str(df_data.get("sample", {})),
            question=query,
        )
        raw = llm.invoke(prompt)
        content = getattr(raw, "content", None) or str(raw)
        content = _sanitize_json_block(content)

        # 3) Parse decision JSON; fallback defaults if needed
        decision: dict[str, Any]
        try:
            decision = json.loads(content)
        except Exception:
            decision = {
                "chart_type": "bar",
                "x_column": (_best_match(df_data.get("columns", []), "region") or (df_data.get("columns", []) or ["Category"])[0]),
                "y_column": _best_match(df_data.get("columns", []), "sales"),
                "title": "Sales Analysis",
                "agg_func": "sum",
                "reasoning": "Fallback due to parsing error",
            }

        # Ensure minimal keys exist
        decision.setdefault("chart_type", "bar")
        decision.setdefault("x_column", (df_data.get("columns", []) or ["Category"])[0] if df_data.get("columns") else "Category")
        decision.setdefault("y_column", None)
        decision.setdefault("title", _title_from_rules(decision["chart_type"], decision["x_column"], decision["y_column"], decision.get("agg_func")))
        decision.setdefault("agg_func", "sum")
        decision.setdefault("reasoning", "")

        return json.dumps(decision)
    except Exception as e:  # pragma: no cover
        log.exception("create_visualization_tool_failed", exc=str(e))
        return json.dumps({"error": f"Error in visualization tool: {e}"})


def execute_chart_template(df: pd.DataFrame, decision: dict[str, Any]):
    """
    Execute template based on a decision dict:
      {chart_type, x_column, y_column, title, agg_func}
    Returns a plotly Figure or None.
    """
    try:
        chart_type = decision.get("chart_type")
        if chart_type not in CHART_MAPPING:
            return None

        x_col = decision.get("x_column")
        y_col = decision.get("y_column")

        # List → first element
        if isinstance(x_col, list) and x_col:
            x_col = x_col[0]
        if isinstance(y_col, list) and y_col:
            y_col = y_col[0]

        # fix columns if needed
        if x_col not in df.columns:
            x_col = _best_match(df.columns, x_col) or (df.select_dtypes(include=["object"]).columns.tolist() or [df.columns[0]])[0]
        if y_col and y_col not in df.columns:
            y_col = _best_match(df.columns, y_col) or (df.select_dtypes(include=["number"]).columns.tolist() or [None])[0]

        # normalized params
        title = decision.get("title")
        agg = decision.get("agg_func", "sum")

        func = CHART_MAPPING[chart_type]
        if chart_type == "bar":
            title = _maybe_llm_title("bar", x_col, y_col, df, agg) or title
            return func(df, x_col, y_col, title, agg)
        if chart_type == "line":
            title = _maybe_llm_title("line", x_col, y_col, df, None) or title
            return func(df, x_col, y_col, title)
        if chart_type == "pie":
            title = _maybe_llm_title("pie", x_col, y_col, df, None) or title
            return func(df, x_col, y_col, title)
        if chart_type == "histogram":
            title = _maybe_llm_title("histogram", x_col, None, df, None) or title
            return func(df, x_col, title)
        if chart_type == "scatter":
            title = _maybe_llm_title("scatter", x_col, y_col, df, None) or title
            return func(df, x_col, y_col, title)
        if chart_type == "box":
            title = _maybe_llm_title("box", x_col, y_col, df, None) or title
            return func(df, x_col, y_col, title)

        return None
    except Exception as e:
        log.exception("execute_chart_template_failed", exc=str(e))
        return None
