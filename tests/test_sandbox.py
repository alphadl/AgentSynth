"""Tests for execution sandbox and validate_trajectory."""

import pytest

from agentsynth.core.types import AgentStep, ToolDefinition, ToolParameter
from agentsynth.execution import DummyToolRegistry, validate_trajectory


def test_dummy_registry() -> None:
    tools = [
        ToolDefinition(
            name="search",
            description="Search",
            parameters=[ToolParameter(name="q", type="string", description="Query", required=True)],
        ),
    ]
    reg = DummyToolRegistry(tools)
    assert reg.get_tool("search") is not None
    assert reg.get_tool("unknown") is None
    assert reg.list_tool_names() == ["search"]


def test_validate_trajectory_ok() -> None:
    reg = DummyToolRegistry([
        ToolDefinition("search", "Search", [ToolParameter("q", "string", "Query", True)]),
    ])
    trajectory = [
        AgentStep(step_index=1, role="user", content="Hi"),
        AgentStep(
            step_index=2,
            role="assistant",
            content="Thinking...",
            tool_calls=[{"name": "search", "arguments": {"q": "screws"}}],
        ),
    ]
    ok, errs = validate_trajectory(trajectory, reg)
    assert ok is True
    assert len(errs) == 0


def test_validate_trajectory_unknown_tool() -> None:
    reg = DummyToolRegistry([
        ToolDefinition("search", "Search", []),
    ])
    trajectory = [
        AgentStep(
            step_index=1,
            role="assistant",
            content="x",
            tool_calls=[{"name": "hallucinated_tool", "arguments": {}}],
        ),
    ]
    ok, errs = validate_trajectory(trajectory, reg)
    assert ok is False
    assert any("unknown tool" in e for e in errs)


def test_validate_trajectory_invalid_arg() -> None:
    reg = DummyToolRegistry([
        ToolDefinition("search", "Search", [ToolParameter("q", "string", "Query", True)]),
    ])
    trajectory = [
        AgentStep(
            step_index=1,
            role="assistant",
            content="x",
            tool_calls=[{"name": "search", "arguments": {"q": "ok", "extra_key": "no"}}],
        ),
    ]
    ok, errs = validate_trajectory(trajectory, reg)
    assert ok is False
    assert any("no parameter" in e for e in errs)
