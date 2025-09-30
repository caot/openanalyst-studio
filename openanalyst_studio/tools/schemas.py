# openanalyst_studio/tools/schemas.py
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# -----------------------
# Core enums / type models
# -----------------------


class ChartType(str, Enum):
    bar = "bar"
    line = "line"
    pie = "pie"
    histogram = "histogram"
    scatter = "scatter"
    box = "box"


class ChartOptions(BaseModel):
    """
    Generic chart options that frontends can safely read.
    Keep this conservative—tool-specific options can be merged in the UI layer.
    """
    agg: Optional[Literal["sum", "mean", "count", "min", "max"]] = None
    bins: Optional[int] = Field(default=None, ge=1, description="Number of bins (histogram).")
    stacked: bool = False
    orientation: Literal["v", "h"] = "v"
    trendline: Optional[Literal["ols", "lowess"]] = None
    sort: Optional[Literal["asc", "desc"]] = None

    model_config = dict(extra="ignore")


class DatasetContext(BaseModel):
    """
    Minimal dataset context the viz-decider needs.
    """
    columns: List[str] = Field(default_factory=list)
    dtypes: Dict[str, str] = Field(default_factory=dict)
    sample: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("columns", mode="before")
    @classmethod
    def _coerce_columns(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        return [str(v)]

    @field_validator("dtypes", mode="before")
    @classmethod
    def _coerce_dtypes(cls, v: Any) -> Dict[str, str]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return {str(k): str(vv).lower() for k, vv in v.items()}
        return {}

    model_config = dict(extra="ignore")


class ChartHints(BaseModel):
    """
    Optional hints to steer chart selection.
    """
    chart_type: Optional[ChartType] = None
    x: Optional[str] = None
    y: Optional[str] = None
    title: Optional[str] = None

    @field_validator("x", "y", "title", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    model_config = dict(extra="ignore")

# -----------------------
# I/O schemas for tools
# -----------------------


class ChartSpec(BaseModel):
    """
    Render-ready chart specification that the frontend can execute.
    Note: we do not validate column existence here (no DataFrame available).
    """
    chart_type: ChartType
    x: str = Field(..., min_length=1, description="Column name for X axis / category.")
    y: Optional[str] = Field(default=None, description="Column name for Y axis (if applicable).")
    group: Optional[str] = Field(default=None, description="Optional grouping column.")
    title: Optional[str] = Field(default=None, max_length=120)
    options: ChartOptions = Field(default_factory=ChartOptions)

    @field_validator("title", mode="before")
    @classmethod
    def _clean_title(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            return v or None
        return v

    @model_validator(mode="after")
    def _validate_axes(self) -> "ChartSpec":
        """
        Enforce minimal axis requirements based on chart type:
          - bar/line/scatter/box: require y
          - histogram/pie: y is optional/ignored
        """
        needs_y = self.chart_type in {ChartType.bar, ChartType.line, ChartType.scatter, ChartType.box}
        if needs_y and not (self.y and isinstance(self.y, str) and self.y.strip()):
            raise ValueError(f"chart_type '{self.chart_type.value}' requires a non-empty 'y' field.")
        if not (self.x and isinstance(self.x, str) and self.x.strip()):
            raise ValueError("A non-empty 'x' field is required.")
        return self

    model_config = dict(extra="ignore")


class VisualizationInput(BaseModel):
    """
    Input to the visualization tool (chart decision).
    """
    query: str = Field(..., min_length=2, description="User question or instruction.")
    dataset_context: DatasetContext
    chart_hints: Optional[ChartHints] = None

    @field_validator("query", mode="before")
    @classmethod
    def _trim_query(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    model_config = dict(extra="ignore")


class VisualizationOutput(BaseModel):
    ok: bool = True
    spec: ChartSpec

    model_config = dict(extra="ignore")


class AnalysisInput(BaseModel):
    question: str = Field(..., min_length=2)
    data_summary: Dict[str, Any]
    chart_context: Optional[Dict[str, Any]] = None

    @field_validator("question", mode="before")
    @classmethod
    def _trim_question(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    model_config = dict(extra="ignore")


class AnalysisOutput(BaseModel):
    ok: bool = True
    answer: str = Field(..., min_length=1)
    # Optional evidence the UI can show (e.g., which metrics informed the answer)
    evidence: List[str] = Field(default_factory=list)

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        return []

    model_config = dict(extra="ignore")
