# openanalyst_studio/openanalyst.py
# How to run: PYTHONPATH=. streamlit run openanalyst_studio/openanalyst.py

from __future__ import annotations

import json
import uuid
from typing import Any

import pandas as pd
import streamlit as st
import structlog
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

from openanalyst_studio.prompts import COMPREHENSIVE_AGENT_PROMPT
from openanalyst_studio.tools import get_llm
from openanalyst_studio.tools.analysis import (
    create_chart_context_tool,
    create_correlation_analysis,  # <-- updated name
    create_data_summary_tool,
    generate_insights_tool,
)
from openanalyst_studio.tools.data_processor import (
    classify_column_types,
    clean_financial_columns,
    coerce_date_columns,
    data_cleaning,
    data_quality_assessment_tool,
    get_data_summary_tool,
    load_file,
    prepare_data_context,
    quick_data_check,
    save_uploaded_file,
    validate_columns_tool,
)
from openanalyst_studio.tools.visualization import (
    create_bar_chart_tool,
    create_box_plot_tool,
    create_histogram_tool,
    create_line_chart_tool,
    create_pie_chart_tool,
    create_scatter_plot_tool,
    create_visualization_tool,
    execute_chart_template,
)
from openanalyst_studio.utils.logging_utils import configure_logging

log = structlog.get_logger()
configure_logging()


# --------------------------------------------------------------------------------------
# Agent factory
# --------------------------------------------------------------------------------------
REAL_VARS = {"input", "agent_scratchpad", "tools", "tool_names"}


def _escape_curly_braces_except(template: str, keep: set[str]) -> str:
    # 1) Escape all braces
    t = template.replace("{", "{{").replace("}", "}}")
    # 2) Unescape the placeholders we really want
    for v in keep:
        t = t.replace("{{" + v + "}}", "{" + v + "}")
    return t

@st.cache_resource(show_spinner=False)
def create_comprehensive_data_analysis_agent() -> AgentExecutor:
    """Create and cache the ReAct-style agent with all tools wired."""
    llm = get_llm()

    tools = [
        # Core visualization + analysis
        create_visualization_tool,
        create_correlation_analysis,
        generate_insights_tool,

        # Per-chart fallbacks
        create_bar_chart_tool,
        create_line_chart_tool,
        create_pie_chart_tool,
        create_histogram_tool,
        create_scatter_plot_tool,
        create_box_plot_tool,

        # Data utilities
        get_data_summary_tool,
        validate_columns_tool,
        data_quality_assessment_tool,
        create_data_summary_tool,
        create_chart_context_tool,
    ]

    safe_prompt_text = _escape_curly_braces_except(COMPREHENSIVE_AGENT_PROMPT, REAL_VARS)
    prompt = PromptTemplate.from_template(safe_prompt_text)  # only needs {input}, {agent_scratchpad}, {tools?}
    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3,
        return_intermediate_steps=True,
    )


# --------------------------------------------------------------------------------------
# Dataset summary for the agent
# --------------------------------------------------------------------------------------
def create_comprehensive_data_summary(df: pd.DataFrame) -> str:
    """Create JSON dataset summary with a business focus for the agent."""
    try:
        df_cleaned = clean_financial_columns(df)

        classification = classify_column_types(df_cleaned)
        business_metrics = classification.get("business_metrics", [])
        identifiers = classification.get("identifiers", [])
        categorical_cols = classification.get("categorical", [])
        temporal_cols = classification.get("temporal", [])

        # Missing values
        missing = df_cleaned.isnull().sum()
        missing_summary = {c: int(v) for c, v in missing.items() if v > 0}

        # Stats for key metrics
        business_stats: dict[str, Any] = {}
        for col in business_metrics[:8]:
            if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                stats = df_cleaned[col].describe()
                business_stats[col] = {
                    "mean": float(stats.get("mean", 0.0)),
                    "std": float(stats.get("std", 0.0)),
                    "min": float(stats.get("min", 0.0)),
                    "max": float(stats.get("max", 0.0)),
                    "median": float(stats.get("50%", 0.0)),
                    "column_type": "business_metric",
                }

        # Categorical quick profile
        categorical_stats: dict[str, Any] = {}
        for col in categorical_cols[:8]:
            vc = df_cleaned[col].value_counts(dropna=True).head(5)
            mode = df_cleaned[col].mode()
            categorical_stats[col] = {
                "unique_count": int(df_cleaned[col].nunique(dropna=True)),
                "top_values": {str(k): int(v) for k, v in vc.to_dict().items()},
                "mode": None if mode.empty else str(mode.iloc[0]),
            }

        summary = {
            "basic_info": {
                "shape": f"{df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns",
                "total_rows": int(df_cleaned.shape[0]),
                "total_columns": int(df_cleaned.shape[1]),
                "memory_usage_mb": float(df_cleaned.memory_usage(deep=True).sum() / 1024 ** 2),
                "data_cleaned": True,
            },
            "intelligent_column_classification": {
                "business_metrics": business_metrics,
                "identifiers": identifiers,
                "categorical_columns": categorical_cols,
                "temporal_columns": temporal_cols,
            },
            "data_quality": {
                "missing_values": missing_summary,
                "duplicate_rows": int(df_cleaned.duplicated().sum()),
                "completeness_percent": float(
                    100.0 - (missing.sum() / max(1, df_cleaned.size) * 100.0)
                ),
            },
            "sample_data": df_cleaned.head(3).to_dict("records"),
            "business_metrics_statistics": business_stats,
            "categorical_summary": categorical_stats,
            "analysis_guidance": {
                "focus_on": "Use business_metrics for statistical analysis and KPI calculations",
                "exclude_from_stats": "Do not include identifiers (IDs, codes) in statistical analysis",
                "visualization_ready": "Data is cleaned and ready for BI visualizations",
            },
        }

        return json.dumps(summary, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error creating comprehensive summary: {e}"


# --------------------------------------------------------------------------------------
# Helpers for agent outputs
# --------------------------------------------------------------------------------------
def _s(x: Any) -> str:
    """Safe coerce to string; None -> ''."""
    return x if isinstance(x, str) else ""


def _parse_chart_blob(text: str) -> dict | None:
    """Extract and parse a minimal JSON object that contains 'chart_type'."""
    try:
        import re

        # Easy path: exact JSON object with "chart_type"
        for m in re.finditer(r'\{[^{}]*"chart_type"[^{}]*\}', text):
            blob = (
                m.group(0)
                .replace("'", '"')
                .replace("True", "true")
                .replace("False", "false")
            )
            return json.loads(blob)
    except Exception:
        return None
    return None


def process_agent_result(agent_result: Any, df: pd.DataFrame) -> tuple[str, Any | None]:
    """
    Process an agent's final answer. If it includes a chart spec, render it.
    Returns (text, plotly_figure_or_None)
    """
    try:
        # 1) Direct dict result
        if isinstance(agent_result, dict) and agent_result.get("chart_type"):
            chart = execute_chart_template(df, agent_result)
            if chart:
                return "", chart

        # 2) String result; try to extract a chart spec
        agent_text = _s(agent_result)
        if agent_text:
            spec = _parse_chart_blob(agent_text)
            if spec and spec.get("chart_type"):
                chart = execute_chart_template(df, spec)
                if chart:
                    return agent_text, chart

            # Fallback: broad slice of the first {...}
            if "{" in agent_text and "}" in agent_text:
                try:
                    start = agent_text.find("{")
                    end = agent_text.rfind("}") + 1
                    jt = (
                        agent_text[start:end]
                        .replace("'", '"')
                        .replace("True", "true")
                        .replace("False", "false")
                    )
                    spec2 = json.loads(jt)
                    if isinstance(spec2, dict) and spec2.get("chart_type"):
                        chart = execute_chart_template(df, spec2)
                        if chart:
                            return agent_text, chart
                except Exception:
                    pass

        return agent_text or str(agent_result), None
    except Exception as e:
        return f"Error processing agent result: {e}", None


# --------------------------------------------------------------------------------------
# Query execution
# --------------------------------------------------------------------------------------
def process_query(query: str, df: pd.DataFrame) -> tuple[str, Any | None]:
    """Run the agent against the dataset and post-process the output."""
    try:
        agent_executor = create_comprehensive_data_analysis_agent()

        # Context for tools and direct answering
        comprehensive_summary = _s(create_comprehensive_data_summary(df))
        basic_context = _s(prepare_data_context(df))
        data_quality = _s(quick_data_check(df))

        agent_input = f"""
USER QUERY: {query}

COMPREHENSIVE DATASET SUMMARY:
{comprehensive_summary}

DATA QUALITY ASSESSMENT:
{data_quality}

BASIC DATASET CONTEXT (for tools):
{basic_context}

INTELLIGENT INSTRUCTIONS:
- BUSINESS METRICS: Focus statistical analysis on these (sales, profit, prices, units, margins)
- IDENTIFIERS: Exclude from statistical analysis (IDs, codes, references)
- CATEGORICAL: Use for grouping and segmentation analysis
- TEMPORAL: Use for time-based analysis

MANDATORY RULES:
1) "statistical analysis" -> ONLY generate_insights_tool
2) "show/create chart" -> ONLY create_visualization_tool
3) Focus: Total Sales, Operating Profit, Price per Unit, Units Sold
4) Exclude identifiers from statistics
5) Max 2 tool calls
6) If answerable from business_metrics_statistics, answer directly
"""

        result = agent_executor.invoke({"input": agent_input}) or {}
        if not isinstance(result, dict):
            result = {"_raw": result}

        # Normalize keys defensively
        agent_output = (
            result.get("output")
            or result.get("final_output")
            or result.get("response")
            or ""
        )
        intermediate_steps = result.get("intermediate_steps") or []

        # Try to discover a chart spec from tool observations first
        chart_data = None
        if isinstance(intermediate_steps, (list, tuple)):
            for step in intermediate_steps:
                obs = None
                try:
                    if isinstance(step, (list, tuple)) and len(step) >= 2:
                        obs = step[1]  # (AgentAction, observation)
                    elif isinstance(step, dict):
                        obs = step.get("observation")
                except Exception:
                    obs = None

                # Observation may be dict or str
                if isinstance(obs, dict) and obs.get("chart_type"):
                    chart_data = obs
                    break
                obs_text = _s(obs)
                if obs_text and "chart_type" in obs_text:
                    try:
                        jt = (
                            obs_text.strip()
                            .replace("'", '"')
                            .replace("True", "true")
                            .replace("False", "false")
                        )
                        if jt.startswith("{") and jt.endswith("}"):
                            maybe = json.loads(jt)
                            if isinstance(maybe, dict) and maybe.get("chart_type"):
                                chart_data = maybe
                                break
                    except Exception:
                        pass

        if isinstance(chart_data, dict) and chart_data.get("chart_type"):
            chart = execute_chart_template(df, chart_data)
            if chart:
                return _s(agent_output), chart

        # Otherwise parse whatever the agent said
        return process_agent_result(agent_output, df)

    except Exception as e:
        log.exception("process_query_failed", query=query)
        return f"Comprehensive agent execution error: {e}", None


# --------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------
def main() -> None:
    title = "OpenAnalyst Studio"
    st.set_page_config(page_title=title, page_icon="📊")
    st.title(f"📊 {title}")

    # Session state
    if "df" not in st.session_state:
        st.session_state.df = None
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {id, role, content, chart}

    # Sidebar: upload
    with st.sidebar:
        st.header("📁 Upload Dataset")
        st.markdown("*Upload a CSV, Excel, or JSON file to start the analysis*")

        uploaded_file = st.file_uploader(
            "Choose your dataset",
            type=["csv", "xlsx", "xls", "json"],
            help="Supported formats: CSV, Excel, JSON",
        )

        if uploaded_file:
            try:
                file_path = save_uploaded_file(uploaded_file)
                df_raw = load_file(file_path)

                if df_raw is not None and isinstance(df_raw, pd.DataFrame):
                    # Coerce common date/timestamp columns
                    common_date_names = [
                        c for c in df_raw.columns
                        if c.lower() in {"date", "dt", "timestamp", "time", "created_at", "updated_at"}
                    ]
                    df_raw = coerce_date_columns(df_raw, force_cols=common_date_names)

                    df_cleaned = data_cleaning(df_raw)
                    st.session_state.df = df_cleaned

                    st.success(f"✅ Loaded: {len(df_cleaned):,} rows × {len(df_cleaned.columns)} columns")
                    st.write(
                        "**Columns:** ",
                        ", ".join(map(str, df_cleaned.columns[:5])) + ("..." if len(df_cleaned.columns) > 5 else ""),
                    )

                    with st.expander("📊 Dataset Preview"):
                        # st.dataframe(df_cleaned.head(3), use_container_width=True)
                        st.dataframe(df_cleaned.head(3), width="stretch")
                else:
                    st.error("Uploaded file could not be parsed as a DataFrame.")
            except Exception as e:
                st.error(f"Upload error: {e}")

    # Main pane
    if st.session_state.df is not None:
        st.subheader("💬 Chat with your data")
        st.markdown("*Ask questions about your dataset — the agent will analyze and visualize the results.*")

        # History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("chart") is not None:
                    st.plotly_chart(
                        msg["chart"],
                        width="stretch",
                        key=f"chart_{msg['id']}",
                    )

        # New input
        if user_query := st.chat_input("Ask about your data..."):
            user_id = uuid.uuid4().hex
            st.session_state.messages.append({"id": user_id, "role": "user", "content": user_query, "chart": None})

            with st.chat_message("user"):
                st.write(user_query)

            assistant_id = uuid.uuid4().hex
            with st.chat_message("assistant"):
                text_slot = st.empty()
                chart_slot = st.empty()
                with st.spinner("🤖 Analyzing with LangChain agent..."):
                    response, chart = process_query(user_query, st.session_state.df)

                text_slot.write(response)
                if chart is not None:
                    # chart_slot.plotly_chart(chart, use_container_width=True)
                    chart_slot.plotly_chart(chart, width="stretch")

            st.session_state.messages.append(
                {"id": assistant_id, "role": "assistant", "content": response, "chart": chart}
            )

        with st.expander("💡 Intelligent Agent Example Questions"):
            st.markdown(
                """
**📊 Business Intelligence & Insights**
• Analyze profitability by region and create visualizations
• What are the key business insights from this data?
• Show me the best performing retailers with charts
• Generate comprehensive insights on sales performance

**📈 Smart Visualizations**
• Create a bar chart showing total sales by region
• Show profit margins across different states
• Visualize the relationship between price and units sold
• Plot sales trends over time

**🔍 Quick Business Questions**
• What's the average profit margin?
• Which region has the highest sales?
• How many different brands are there?
• What are the top 5 performing cities?

**🎯 Advanced Analytics**
• Show correlations between business metrics
• Analyze sales performance by retailer and visualize
• Compare operating margins across regions with charts
• Identify outliers in pricing and profitability
"""
            )
            st.info(
                "🤖 **Smart Agent**: Cleans financial data, focuses on business metrics (not IDs), "
                "and creates intelligent visualizations with actionable insights."
            )

    else:
        st.info("👆 Upload a CSV/Excel/JSON file to start analyzing with the LangChain agent!")
        st.markdown(
            """
### 🤖 About This App
- **LangChain Agent Architecture**
- **Tool-based AI Workflows**
- **Intelligent Query Routing**
- **Template-driven Visualizations**
"""
        )


if __name__ == "__main__":
    main()
