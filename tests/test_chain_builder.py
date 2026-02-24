"""Tests for chain_builder."""

from agentsynth.core.types import ToolDefinition, ToolParameter
from agentsynth.generators import build_chains


def test_build_chains_basic() -> None:
    tools = [
        ToolDefinition(name="search", description="Search", parameters=[ToolParameter(name="q", type="string", description="Query", required=True)]),
        ToolDefinition(name="get_detail", description="Detail", parameters=[ToolParameter(name="id", type="string", description="ID", required=True)]),
    ]
    chains = build_chains(tools, max_length=2, num_chains=2, shuffle=False)
    assert len(chains) == 2
    for chain in chains:
        assert len(chain) == 2
        assert chain[0]["name"] in ("search", "get_detail")
        assert "arguments" in chain[0]
        assert isinstance(chain[0]["arguments"], dict)


def test_build_chains_empty() -> None:
    assert build_chains([]) == []
