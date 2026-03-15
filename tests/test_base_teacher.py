"""Tests for BaseTeacher (with mocked LLM)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agentsynth.core.types import ToolDefinition, ToolParameter
from agentsynth.teachers import BaseTeacher
from agentsynth.teachers.base_teacher import _parse_trajectory


def test_parse_trajectory_empty_or_invalid() -> None:
    assert _parse_trajectory(None) == []
    assert _parse_trajectory("") == []
    assert _parse_trajectory("  ") == []
    assert _parse_trajectory("{}") == []
    assert _parse_trajectory("[]") == []
    assert _parse_trajectory("not json at all {{{") == []
    assert _parse_trajectory('{"key": "value"}') == []


@pytest.fixture
def mock_litellm():
    with patch("agentsynth.teachers.base_teacher.litellm") as m:
        yield m


def test_base_teacher_parse_trajectory(mock_litellm: MagicMock) -> None:
    trajectory_json = [
        {"step_index": 1, "role": "user", "content": "Find screws"},
        {
            "step_index": 2,
            "role": "assistant",
            "content": "I will search.",
            "tool_calls": [{"name": "search", "arguments": {"q": "screws"}}],
        },
    ]
    mock_litellm.completion.return_value.choices = [
        MagicMock(message=MagicMock(content=json.dumps(trajectory_json)))
    ]
    teacher = BaseTeacher(model="gpt-4o", temperature=0.0)
    tools = [
        ToolDefinition(
            name="search",
            description="Search",
            parameters=[ToolParameter(name="q", type="string", description="Query", required=True)],
        )
    ]
    pair = teacher.generate("Find screws", tools, sample_id="test-1")
    assert pair.id == "test-1"
    assert pair.source_method == "forward_teacher"
    assert pair.user_prompt == "Find screws"
    assert len(pair.trajectory) == 2
    assert pair.trajectory[0].role == "user"
    assert pair.trajectory[1].tool_calls is not None
    assert pair.execution_success is False


def test_base_teacher_raises_on_empty_trajectory(mock_litellm: MagicMock) -> None:
    """generate() should raise ValueError (and retry) when LLM returns unparseable output."""
    mock_litellm.completion.return_value.choices = [
        MagicMock(message=MagicMock(content="not json {{{"))
    ]
    teacher = BaseTeacher(model="gpt-4o", temperature=0.0)
    tools = [ToolDefinition(name="search", description="Search", parameters=[])]
    with pytest.raises(ValueError, match="empty or unparseable"):
        teacher.generate("Find screws", tools, sample_id="test-err")
