"""Tests for chain_builder."""

from agentsynth.core.types import ToolDefinition, ToolParameter
from agentsynth.generators import build_chains


def test_build_chains_basic() -> None:
    tools = [
        ToolDefinition(
            name="search",
            description="Search",
            parameters=[ToolParameter(name="q", type="string", description="Query", required=True)],
        ),
        ToolDefinition(
            name="get_detail",
            description="Detail",
            parameters=[ToolParameter(name="id", type="string", description="ID", required=True)],
        ),
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


def test_build_chains_seed_reproducible() -> None:
    tools = [
        ToolDefinition(name="a", description="A", parameters=[]),
        ToolDefinition(name="b", description="B", parameters=[]),
        ToolDefinition(name="c", description="C", parameters=[]),
    ]
    chains1 = build_chains(tools, num_chains=3, shuffle=True, seed=42)
    chains2 = build_chains(tools, num_chains=3, shuffle=True, seed=42)
    assert chains1 == chains2


def test_build_chains_max_length_capped_by_tools() -> None:
    """max_length > len(tools) is silently capped; each tool appears at most once."""
    tools = [
        ToolDefinition(name="x", description="X", parameters=[]),
        ToolDefinition(name="y", description="Y", parameters=[]),
    ]
    chains = build_chains(tools, max_length=10, num_chains=1, shuffle=False)
    assert len(chains[0]) == 2  # capped by number of tools
