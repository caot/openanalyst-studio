# openanalyst_studio/tools/data_processor.py
from __future__ import annotations

import csv
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from langchain.tools import tool
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

# Streamlit is optional: core functions should not depend on it.
try:  # pragma: no cover
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # sentinel; use _warn/_error helpers below

# ---------------------------
# Helpers / Logging shims
# ---------------------------


def _warn(msg: str) -> None:
    if st is not None:  # pragma: no cover
        st.warning(msg)
    else:
        print(f"[WARN] {msg}")


def _error(msg: str) -> None:
    if st is not None:  # pragma: no cover
        st.error(msg)
    else:
        print(f"[ERROR] {msg}")

# ---------------------------
# Regexes for date detection
# ---------------------------


ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:\d{2})?$"
)

# Currency / numeric-like strings
_CURRENCY_STRIP_RE = re.compile(r"[\$\€\£\¥%,\s]")
_NON_NUMERIC_RE = re.compile(r"[^\d\.\-]")

# ---------------------------
# Type Coercion Utilities
# ---------------------------


def coerce_date_columns(
    df: pd.DataFrame,
    force_cols: Optional[List[str]]=None,
    sample_size: int=200,
    threshold: float=0.7,
) -> pd.DataFrame:
    """
    Convert columns that look like ISO dates/datetimes into pandas datetime.
    - force_cols: always attempt to parse these names (e.g., ['date','timestamp'])
    - sample_size: non-null rows to inspect per column
    - threshold: fraction of sampled rows that must match date/datetime regex
    """
    out = df.copy()
    candidates: set[str] = set(force_cols or [])

    for col in out.columns:
        s = out[col]
        if is_datetime64_any_dtype(s) or not is_object_dtype(s):
            continue  # already datetime or clearly not a date-like object

        non_null = s.dropna().astype(str)
        if non_null.empty:
            continue

        sample = non_null.head(sample_size)
        match_ratio = (sample.str.match(ISO_DATE_RE) | sample.str.match(ISO_DATETIME_RE)).mean()
        if match_ratio >= threshold:
            candidates.add(col)

    for col in candidates:
        try:
            out[col] = pd.to_datetime(out[col], errors="coerce", utc=False)
        except Exception:
            # leave column as-is if parsing fails
            pass
    return out


def _coerce_numeric_like_series(s: pd.Series) -> pd.Series:
    """
    Attempt to convert object dtype series with currency/commas/percentages to numeric.
    - Keeps only digits, dot, minus after stripping common currency/format chars.
    - Returns a numeric series where possible; otherwise returns original.
    """
    if not is_object_dtype(s):
        return s

    sample = s.dropna().astype(str).head(10)
    looks_numeric = any(ch in "".join(sample.tolist()) for ch in ["$", ",", "%"]) or any(
        c.isdigit() for c in "".join(sample.tolist())
    )
    if not looks_numeric:
        return s

    cleaned = s.astype(str)
    # Remove quotes/spaces first to avoid artifacts
    cleaned = cleaned.str.replace(r'["\']', "", regex=True).str.strip()
    # Strip currency/commas/whitespace/percentage signs
    cleaned = cleaned.str.replace(_CURRENCY_STRIP_RE, "", regex=True)
    # Keep digits/dot/minus only
    cleaned = cleaned.str.replace(_NON_NUMERIC_RE, "", regex=True)
    cleaned = cleaned.replace("", np.nan)

    numeric = pd.to_numeric(cleaned, errors="coerce")
    # If too few conversions succeed, revert (avoid damaging categorical text)
    non_null = s.notna().sum()
    success = numeric.notna().sum()
    if non_null == 0 or (success / max(1, non_null)) < 0.7:
        return s
    return numeric


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive data cleaning for general business datasets.
    Steps:
      1) Drop fully-empty rows/cols
      2) Normalize column names
      3) Trim strings and standardize missing tokens
      4) Coerce numeric-like object columns
      5) Parse common date columns (MM/DD/YYYY and ISO)
      6) Drop duplicates
      7) Conservative outlier capping (IQR * 4)
      8) Keep ID-like columns as strings if high-cardinality
    """
    try:
        out = df.copy()

        # 1. Remove fully empty rows/columns
        out = out.dropna(how="all")
        out = out.loc[:, ~out.isna().all()]

        # 2. Clean column names
        out.columns = (
            pd.Index(out.columns)
            .map(lambda c: str(c).strip())
            .map(lambda c: re.sub(r"\s+", " ", c))
        )

        # 3. Clean object columns: trim and normalize null-like tokens
        for col in out.select_dtypes(include=["object"]).columns:
            s = out[col].astype(str).str.strip().str.strip('"').str.strip("'").str.strip()
            s = s.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "NULL": np.nan, "null": np.nan})
            out[col] = s

        # 4. Coerce numeric-like strings
        for col in out.columns:
            if is_object_dtype(out[col]):
                out[col] = _coerce_numeric_like_series(out[col])

        # 5. Parse dates: first attempt common business format, then ISO/autodetect
        date_name_hints = ("date", "time", "day", "month", "year", "period", "created", "updated", "timestamp")
        for col in out.columns:
            if is_object_dtype(out[col]) and any(k in col.lower() for k in date_name_hints):
                try:
                    parsed = pd.to_datetime(out[col], format="%m/%d/%Y", errors="coerce")
                    if parsed.notna().any():
                        out[col] = parsed
                        continue
                except Exception:
                    pass
        out = coerce_date_columns(out, force_cols=[c for c in out.columns if any(k in c.lower() for k in date_name_hints)])

        # 6. Remove duplicates
        out = out.drop_duplicates()

        # 7. Conservative outlier capping
        for col in out.select_dtypes(include=[np.number]).columns:
            s = out[col]
            if s.notna().sum() == 0:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr == 0:
                continue
            lo, hi = q1 - 4 * iqr, q3 + 4 * iqr
            # Cap only if there are truly extreme values
            if (s < lo).any() or (s > hi).any():
                out[col] = s.clip(lo, hi)

        # 8. Preserve ID-like columns as strings if very high cardinality
        id_hints = ("id", "code", "number", "ref", "key")
        n = len(out)
        if n > 0:
            for col in out.columns:
                if any(k in col.lower() for k in id_hints) and not is_object_dtype(out[col]):
                    # high-cardinality (heuristic): >50% uniques suggests identifier
                    if out[col].nunique(dropna=True) / n > 0.5:
                        out[col] = out[col].astype(str)

        return out

    except Exception as e:  # pragma: no cover
        _warn(f"Error in data cleaning: {e}")
        return df


def clean_financial_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and clean financial columns with currency formatting (price, cost, revenue, sales, profit, margin, amount, value, total, unit).
    Converts to numeric when >=80% of non-null entries can be parsed.
    """
    try:
        out = df.copy()
        financial_terms = ("price", "cost", "revenue", "sales", "profit", "margin", "amount", "value", "total", "unit")
        for col in out.columns:
            if is_object_dtype(out[col]) and any(t in col.lower() for t in financial_terms):
                s = out[col].astype(str)
                s = s.str.replace(_CURRENCY_STRIP_RE, "", regex=True).str.replace(_NON_NUMERIC_RE, "", regex=True)
                s = s.replace("", np.nan)
                numeric = pd.to_numeric(s, errors="coerce")
                non_null = out[col].notna().sum()
                success = numeric.notna().sum()
                if non_null > 0 and (success / non_null) >= 0.8:
                    out[col] = numeric
        return out
    except Exception as e:  # pragma: no cover
        _warn(f"Error cleaning financial columns: {e}")
        return df


def classify_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Classify columns into business categories:
      - business_metrics: numeric columns with KPI-ish names (sales, revenue, profit, etc.) or other numeric columns
      - identifiers: numeric-looking ID/code/ref/key/number
      - categorical: object/bool
      - temporal: datetime-like
      - unknown: anything else
    """
    try:
        result = {"business_metrics": [], "identifiers": [], "categorical": [], "temporal": [], "unknown": []}
        kpi_terms = ("price", "cost", "revenue", "sales", "profit", "margin", "amount", "value", "total", "units", "volume", "quantity")
        id_terms = ("id", "code", "number", "ref", "key")
        time_terms = ("date", "time", "day", "month", "year", "period")

        for col in df.columns:
            name = col.lower()
            s = df[col]
            if is_datetime64_any_dtype(s):
                result["temporal"].append(col)
            elif is_bool_dtype(s) or is_object_dtype(s):
                result["categorical"].append(col)
            elif is_numeric_dtype(s):
                if any(t in name for t in id_terms):
                    result["identifiers"].append(col)
                elif any(t in name for t in kpi_terms):
                    result["business_metrics"].append(col)
                else:
                    # default other numeric columns to business metrics
                    result["business_metrics"].append(col)
            else:
                if any(t in name for t in time_terms):
                    result["temporal"].append(col)
                else:
                    result["unknown"].append(col)
        return result
    except Exception as e:  # pragma: no cover
        _warn(f"Error classifying columns: {e}")
        return {"business_metrics": [], "identifiers": [], "categorical": [], "temporal": [], "unknown": []}

# ---------------------------
# File IO
# ---------------------------


def _sniff_csv_dialect(path: str, sample_bytes: int=8192) -> Tuple[str, Optional[str]]:
    """
    Sniff delimiter and encoding for CSV. Returns (delimiter, encoding).
    Tries utf-8-sig first; falls back to utf-8.
    """
    encodings = ("utf-8-sig", "utf-8")
    for enc in encodings:
        try:
            with open(path, "rb") as fb:
                head = fb.read(sample_bytes)
            text = head.decode(enc, errors="ignore")
            dialect = csv.Sniffer().sniff(text)
            return dialect.delimiter, enc
        except Exception:
            continue
    return ",", "utf-8"


def load_file(file_path: str) -> Optional[pd.DataFrame]:
    """Load CSV/Excel/JSON/Parquet with sensible defaults and light heuristics."""
    try:
        lower = file_path.lower()
        if lower.endswith(".csv"):
            delim, enc = _sniff_csv_dialect(file_path)
            return pd.read_csv(file_path, encoding=enc or "utf-8", sep=delim, engine="python")
        if lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(file_path, engine=None)  # let pandas choose
        if lower.endswith(".json"):
            return pd.read_json(file_path, lines=False)
        if lower.endswith((".parquet", ".pq")):
            return pd.read_parquet(file_path)  # requires pyarrow/fastparquet
        raise ValueError(f"Unsupported file format: {os.path.splitext(file_path)[1]}")
    except Exception as e:  # pragma: no cover
        _error(f"Error loading file: {e}")
        return None


def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file to ./document safely (prevents path traversal)."""
    try:
        os.makedirs("document", exist_ok=True)
        name = os.path.basename(getattr(uploaded_file, "name", "uploaded.bin"))
        path = os.path.join("document", name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return path
    except Exception as e:  # pragma: no cover
        _error(f"Error saving file: {e}")
        return None

# ---------------------------
# Summaries & Validation
# ---------------------------


@tool
def get_data_summary_tool(input_text: str) -> str:
    """Return a small control JSON indicating dataset summary is available."""
    try:
        payload = {
            "tool_used": "data_summary",
            "message": "Enhanced dataset summary with business intelligence completed.",
            "instruction": (
                "Data has been cleaned and columns classified into business metrics, "
                "identifiers, categorical, and temporal types. Focus statistics on business metrics."
            ),
            "business_focus": "Use sales/profit/prices; exclude identifiers.",
            "success": True,
        }
        return json.dumps(payload)
    except Exception as e:  # pragma: no cover
        return json.dumps({"success": False, "error": f"Error in data summary tool: {e}"})


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a compact DataFrame summary for LLMs (JSON-serializable)."""
    try:
        numerics = df.select_dtypes(include="number")
        numeric_summary: Dict[str, Dict[str, float]] = {}
        if not numerics.empty:
            for col, stats in numerics.describe().to_dict().items():
                numeric_summary[str(col)] = {
                    str(k): float(v) for k, v in stats.items() if pd.notna(v)
                }
        return {
            "shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
            "columns": [str(c) for c in df.columns],
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "sample": df.head(2).to_dict(),
            "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1024 ** 2), 2),
            "numeric_summary": numeric_summary,
        }
    except Exception as e:  # pragma: no cover
        return {"error": f"Error creating summary: {e}"}


@tool
def validate_columns_tool(input_text: str) -> str:
    """
    Validate if specific columns exist in the dataset.
    Input: "columns: [col1,col2]" or "check: col"
    Note: This tool signals intent; actual validation requires the runtime dataset.
    """
    try:
        cols: List[str] = []
        text = input_text.strip()
        if "columns:" in text:
            part = text.split("columns:", 1)[1].strip()
            part = part.strip("[]")
            cols = [c.strip().strip('"').strip("'") for c in part.split(",") if c.strip()]
        elif "check:" in text:
            cols = [text.split("check:", 1)[1].strip()]
        payload = {
            "tool_used": "column_validation",
            "columns_requested": cols,
            "message": "Column validation requested. Refer to dataset summary for available columns.",
            "instruction": "Use exact column names from the provided dataset summary.",
            "success": True,
        }
        return json.dumps(payload)
    except Exception as e:  # pragma: no cover
        return json.dumps({"success": False, "error": f"Error in column validation tool: {e}"})


def validate_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> Tuple[bool, str]:
    """Validate that required_cols are present in df.columns."""
    try:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return False, f"Missing columns: {missing}"
        return True, "All columns found"
    except Exception as e:  # pragma: no cover
        return False, f"Validation error: {e}"


def prepare_data_context(df: pd.DataFrame) -> str:
    """Format minimal context for LLM (columns, dtypes, tiny sample) as JSON string."""
    try:
        context = {
            "columns": [str(c) for c in df.columns],
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "sample": df.head(2).to_dict(),
        }
        return json.dumps(context)
    except Exception as e:  # pragma: no cover
        return json.dumps({"error": f"Error preparing context: {e}"})


@tool
def data_quality_assessment_tool(input_text: str) -> str:
    """Return a scaffold with data quality checks & recommendations."""
    try:
        payload = {
            "tool_used": "data_quality_assessment",
            "analysis_type": "quality_check",
            "message": "Data quality assessment scaffold.",
            "instruction": (
                "Refer to missing values, duplicates, dtype validation, and range checks in the dataset summary."
            ),
            "recommendations": [
                "Inspect missing value patterns per column",
                "Check duplicate rows/keys",
                "Validate data types and permissible value ranges",
                "Review outlier handling policy",
            ],
            "success": True,
        }
        return json.dumps(payload)
    except Exception as e:  # pragma: no cover
        return json.dumps({"success": False, "error": f"Error in data quality assessment tool: {e}"})


def quick_data_check(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute lightweight quality indicators for dashboards/agents."""
    try:
        return {
            "total_rows": int(len(df)),
            "total_cols": int(len(df.columns)),
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_cols": int(len(df.select_dtypes(include="number").columns)),
            "categorical_cols": int(len(df.select_dtypes(include=["object", "bool"]).columns)),
            "temporal_cols": int(len(df.select_dtypes(include=["datetime"]).columns)),
        }
    except Exception as e:  # pragma: no cover
        return {"error": f"Error in data check: {e}"}


def format_data_info(df: pd.DataFrame) -> str:
    """Human-friendly dataset info block for UI."""
    try:
        mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
        info = (
            "📊 **Dataset Info:**\n"
            f"- **Size:** {len(df):,} rows × {len(df.columns)} columns\n"
            f"- **Memory:** {mem_mb:.1f} MB\n"
            f"- **Missing values:** {df.isna().sum().sum():,} cells\n"
            f"- **Numeric columns:** {len(df.select_dtypes(include='number').columns)}\n"
            f"- **Text/Bool columns:** {len(df.select_dtypes(include=['object','bool']).columns)}\n"
            f"- **Temporal columns:** {len(df.select_dtypes(include=['datetime']).columns)}\n"
        )
        return info
    except Exception as e:  # pragma: no cover
        return f"Error formatting data info: {e}"


__all__ = [
    # loaders
    "load_file",
    "save_uploaded_file",
    # cleaning/coercion
    "data_cleaning",
    "clean_financial_columns",
    "coerce_date_columns",
    "classify_column_types",
    # summaries/tools
    "get_data_summary_tool",
    "get_data_summary",
    "validate_columns_tool",
    "validate_columns",
    "prepare_data_context",
    "data_quality_assessment_tool",
    "quick_data_check",
    "format_data_info",
]
