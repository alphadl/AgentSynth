"""Tests for core Pydantic types."""

import pytest
from pydantic import ValidationError

from agentsynth.core.types import (
    AgentStep,
    GenerationConfig,
    SynthesizedDataPair,
    ToolDefinition,
    ToolParameter,
)


def test_tool_parameter() -> None:
    p = ToolParameter(name="query", type="string", description="Search query", required=True)
    assert p.name == "query"
    assert p.required is True


def test_tool_definition() -> None:
    td = ToolDefinition(
        name="search_item",
        description="Search for products",
        parameters=[
            ToolParameter(name="query", type="string", description="Query", required=True),
        ],
    )
    assert td.name == "search_item"
    assert len(td.parameters) == 1
    assert td.parameters[0].name == "query"


def test_agent_step() -> None:
    step = AgentStep(
        step_index=1,
        role="assistant",
        content="I will search for screws.",
        tool_calls=[{"name": "search_item", "arguments": {"query": "screws"}}],
    )
    assert step.step_index == 1
    assert step.role == "assistant"
    assert step.tool_calls is not None
    assert step.tool_calls[0]["name"] == "search_item"


def test_agent_step_role_invalid() -> None:
    with pytest.raises(ValidationError):
        AgentStep(step_index=1, role="system", content="x")  # type: ignore[arg-type]


def test_synthesized_data_pair() -> None:
    pair = SynthesizedDataPair(
        id="sample-001",
        source_method="forward_teacher",
        user_prompt="Find a supplier for screws.",
        trajectory=[
            AgentStep(step_index=1, role="user", content="Find a supplier for screws."),
            AgentStep(
                step_index=2,
                role="assistant",
                content="I'll search.",
                tool_calls=[{"name": "search_item", "arguments": {"query": "screws"}}],
            ),
        ],
        execution_success=True,
        verifier_score=0.9,
    )
    assert pair.id == "sample-001"
    assert pair.source_method == "forward_teacher"
    assert pair.execution_success is True
    assert len(pair.trajectory) == 2


def test_synthesized_data_pair_source_method_invalid() -> None:
    with pytest.raises(ValidationError):
        SynthesizedDataPair(
            id="x",
            source_method="invalid",
            user_prompt="q",
            trajectory=[AgentStep(step_index=1, role="user", content="q")],
            execution_success=True,
        )  # type: ignore[arg-type]


def test_generation_config_defaults() -> None:
    config = GenerationConfig()
    assert "gpt-4o" in config.teacher_models
    assert config.back_translation_model == "gpt-4o"
    assert config.max_steps == 15
    assert config.temperature == 0.7


def test_generation_config_custom() -> None:
    config = GenerationConfig(
        teacher_models=["claude-3-opus"],
        max_steps=20,
        temperature=0.5,
    )
    assert config.teacher_models == ["claude-3-opus"]
    assert config.max_steps == 20
    assert config.temperature == 0.5
