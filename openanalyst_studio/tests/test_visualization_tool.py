import json

import pytest

from openanalyst_studio.tools.schemas import ChartSpec, VisualizationOutput
from openanalyst_studio.tools.visualization_tool import VisualizationTool


@pytest.fixture
def dataset_context():
    # Minimal context your tool expects
    return {
        "columns": {
            "region": "category",
            "sales": "float64",
            "units": "int64",
        }
    }


def test_visualization_basic_selection(dataset_context):
    tool = VisualizationTool()
    out = tool.invoke({
        "query": "Show sales by region",
        "dataset_context": dataset_context,
        "chart_hints": None,
    })
    # Tool returns a JSON string; parse and validate with schema
    obj = json.loads(out)
    vis = VisualizationOutput.model_validate(obj)
    assert vis.ok is True
    assert isinstance(vis.spec, ChartSpec)
    # With a categorical + numeric present, default should be bar (per your heuristic)
    assert vis.spec.chart_type in {"bar", "pie"}
    assert vis.spec.x in {"region", "sales", "units"}  # depends on heuristic
    if vis.spec.chart_type != "pie":
        assert vis.spec.y in {"sales", "units"}


def test_visualization_hints_override(dataset_context):
    tool = VisualizationTool()
    out = tool.invoke({
        "query": "Any chart",
        "dataset_context": dataset_context,
        "chart_hints": {"chart_type": "scatter", "x": "sales", "y": "units", "title": "My Chart"},
    })
    obj = json.loads(out)
    vis = VisualizationOutput.model_validate(obj)
    assert vis.spec.chart_type == "scatter"
    assert vis.spec.x == "sales"
    assert vis.spec.y == "units"
    assert vis.spec.title == "My Chart"
