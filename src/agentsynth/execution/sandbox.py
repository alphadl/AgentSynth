"""Execution sandbox and trajectory validation (Reject Sampling).

Validates that every tool_call in a trajectory matches the schema of a
ToolRegistry; rejects hallucinated names or arguments.
"""

from __future__ import annotations

from typing import Protocol

from agentsynth.core.types import AgentStep, ToolDefinition


class ToolRegistry(Protocol):
    """Protocol for a registry of available tools (by name)."""

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Return the tool definition for the given name, or None if unknown."""
        ...

    def list_tool_names(self) -> list[str]:
        """Return all registered tool names."""
        ...


class DummyToolRegistry:
    """In-memory tool registry for testing and validation."""

    def __init__(self, tools: list[ToolDefinition] | None = None) -> None:
        self._by_name: dict[str, ToolDefinition] = {}
        for t in tools or []:
            self._by_name[t.name] = t

    def register(self, tool: ToolDefinition) -> None:
        self._by_name[tool.name] = tool

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self._by_name.get(name)

    def list_tool_names(self) -> list[str]:
        return list(self._by_name.keys())


def validate_trajectory(
    trajectory: list[AgentStep],
    registry: ToolRegistry,
) -> tuple[bool, list[str]]:
    """Check that every tool_call in the trajectory matches the registry schema.

    Returns (success, list of error messages).
    Rejects on: unknown tool name, argument key not in schema.
    """
    errors: list[str] = []
    for step in trajectory:
        if step.tool_calls is None:
            continue
        for i, call in enumerate(step.tool_calls):
            if not isinstance(call, dict):
                errors.append(f"Step {step.step_index} call {i}: not a dict")
                continue
            name = call.get("name") if isinstance(call.get("name"), str) else None
            if not name:
                errors.append(f"Step {step.step_index} call {i}: missing or invalid 'name'")
                continue
            tool = registry.get_tool(name)
            if tool is None:
                errors.append(f"Step {step.step_index} call {i}: unknown tool '{name}'")
                continue
            args = call.get("arguments")
            if args is None:
                args = {}
            if not isinstance(args, dict):
                errors.append(f"Step {step.step_index} call {i}: 'arguments' must be a dict")
                continue
            param_names = {p.name for p in tool.parameters}
            for key in args:
                if key not in param_names:
                    errors.append(
                        f"Step {step.step_index} call {i}: tool '{name}' has no parameter '{key}'"
                    )
            for p in tool.parameters:
                if p.required and p.name not in args:
                    errors.append(
                        f"Step {step.step_index} call {i}: "
                        f"tool '{name}' missing required '{p.name}'"
                    )
    return (len(errors) == 0, errors)
