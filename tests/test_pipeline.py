"""Tests for Pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from agentsynth.core.types import AgentStep, SynthesizedDataPair, ToolDefinition, ToolParameter
from agentsynth.pipeline import Pipeline


def test_pipeline_save_jsonl(tmp_path: Path) -> None:
    p = Pipeline()
    pairs = [
        SynthesizedDataPair(
            id="1",
            source_method="forward_teacher",
            user_prompt="Hi",
            trajectory=[AgentStep(step_index=1, role="user", content="Hi")],
            execution_success=True,
        ),
    ]
    out = tmp_path / "out.jsonl"
    p.save_jsonl(pairs, out)
    assert out.exists()
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 1
    assert "forward_teacher" in lines[0]


def test_pipeline_run_forward_no_type_error() -> None:
    """Regression: model_copy(update=...) must not raise TypeError on execution_success."""
    trajectory_json = [
        {"step_index": 1, "role": "user", "content": "Find screws"},
        {
            "step_index": 2,
            "role": "assistant",
            "content": "Searching.",
            "tool_calls": [{"name": "search", "arguments": {"q": "screws"}}],
        },
    ]
    tools = [
        ToolDefinition(
            name="search",
            description="Search",
            parameters=[ToolParameter(name="q", type="string", description="Query", required=True)],
        )
    ]
    p = Pipeline()
    p.set_tools(tools)
    with patch("agentsynth.teachers.base_teacher.litellm") as mock_llm:
        mock_llm.completion.return_value.choices = [
            MagicMock(message=MagicMock(content=json.dumps(trajectory_json)))
        ]
        pair = p.run_forward("Find screw suppliers")
    assert isinstance(pair.execution_success, bool)
    assert pair.source_method == "forward_teacher"


def test_pipeline_run_back_translation_no_type_error() -> None:
    """Regression: model_copy(update=...) must not raise TypeError on execution_success."""
    tools = [
        ToolDefinition(
            name="search",
            description="Search",
            parameters=[ToolParameter(name="q", type="string", description="Query", required=True)],
        )
    ]
    p = Pipeline()
    p.set_tools(tools)
    with patch("agentsynth.teachers.back_translator.litellm") as mock_llm:
        mock_llm.completion.return_value.choices = [
            MagicMock(message=MagicMock(content="Find screw suppliers and compare prices"))
        ]
        results = p.run_back_translation(num_samples=1)
    assert len(results) == 1
    assert isinstance(results[0].execution_success, bool)
    assert results[0].source_method == "back_translation"
