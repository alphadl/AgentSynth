"""Tests for Pipeline."""

from pathlib import Path

import pytest

from agentsynth.core.types import AgentStep, SynthesizedDataPair
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
