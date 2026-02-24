"""Tool chain builder: programmatically create valid sequences of tool calls."""

from __future__ import annotations

import random
from typing import Any

from agentsynth.core.types import ToolDefinition, ToolParameter


def _sample_args_for_tool(tool: ToolDefinition) -> dict[str, Any]:
    """Generate placeholder arguments for a tool from its schema."""
    args: dict[str, Any] = {}
    for p in tool.parameters:
        if p.type in ("integer", "int", "number"):
            args[p.name] = 10
        elif p.type in ("array", "list"):
            args[p.name] = []
        elif p.type == "boolean":
            args[p.name] = True
        else:
            args[p.name] = f"<{p.name}>"
    return args


def build_chains(
    tools: list[ToolDefinition],
    max_length: int = 5,
    num_chains: int = 1,
    shuffle: bool = True,
) -> list[list[dict[str, Any]]]:
    """Build valid tool-call sequences from tool definitions.

    Each chain is a list of {"name": tool_name, "arguments": {...}} with
    arguments conforming to each tool's parameters.

    Args:
        tools: Available tool definitions.
        max_length: Max steps per chain.
        num_chains: Number of chains to generate.
        shuffle: If True, randomize tool order per chain.

    Returns:
        List of chains; each chain is a list of tool-call dicts.
    """
    if not tools:
        return []
    names = [t.name for t in tools]
    by_name = {t.name: t for t in tools}
    chains: list[list[dict[str, Any]]] = []
    for _ in range(num_chains):
        order = names.copy()
        if shuffle:
            random.shuffle(order)
        length = min(max_length, len(order))
        chain = []
        for name in order[:length]:
            tool = by_name[name]
            args = _sample_args_for_tool(tool)
            chain.append({"name": name, "arguments": args})
        chains.append(chain)
    return chains
