from __future__ import annotations

from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

from openanalyst_studio.tools.llm import get_llm


def get_tools() -> list:
    from openanalyst_studio.tools.analysis_tool import AnalysisTool
    from openanalyst_studio.tools.visualization_tool import VisualizationTool
    return [VisualizationTool(), AnalysisTool()]

def create_agent(
    agent_type: AgentType=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose: bool=False,
    max_iterations: int=3,
    handle_parsing_errors: bool=True,
    llm=None,
):
    llm = llm or get_llm()
    return initialize_agent(
        tools=get_tools(),
        llm=llm,
        agent=agent_type,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=handle_parsing_errors,
    )
