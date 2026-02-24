"""Tests for BackTranslator (mocked LLM)."""

from unittest.mock import MagicMock, patch

import pytest

from agentsynth.teachers import BackTranslator


@pytest.fixture
def mock_litellm():
    with patch("agentsynth.teachers.back_translator.litellm") as m:
        yield m


def test_back_translator(mock_litellm: MagicMock) -> None:
    mock_litellm.completion.return_value.choices = [
        MagicMock(message=MagicMock(content="Find screw suppliers and compare prices"))
    ]
    bt = BackTranslator(model="gpt-4o", temperature=0.0)
    chain = [{"name": "search", "arguments": {"q": "screws"}}, {"name": "get_detail", "arguments": {"id": "1"}}]
    pair = bt.generate(chain, sample_id="bt-1")
    assert pair.id == "bt-1"
    assert pair.source_method == "back_translation"
    assert "screw" in pair.user_prompt.lower() or "supplier" in pair.user_prompt.lower()
    assert len(pair.trajectory) == 4  # 2 assistant + 2 tool steps
    assert pair.trajectory[0].tool_calls is not None
    assert pair.trajectory[0].tool_calls[0]["name"] == "search"
