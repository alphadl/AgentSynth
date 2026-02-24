"""Back-translator: tool chain -> user prompt via LLM."""

from __future__ import annotations

import logging
import uuid

from agentsynth.core.types import (
    AgentStep,
    SynthesizedDataPair,
)
from agentsynth.teachers.prompts import BACK_TRANSLATOR_SYSTEM
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

try:
    import litellm
except ImportError:
    litellm = None  # type: ignore[assignment]


def _chain_to_text(chain: list[dict]) -> str:
    lines = []
    for i, call in enumerate(chain, 1):
        name = call.get("name", "?")
        args = call.get("arguments") or {}
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        lines.append(f"{i}. {name}({args_str})")
    return "\n".join(lines)


def _chain_to_trajectory(chain: list[dict]) -> list[AgentStep]:
    """Turn a tool chain into a minimal trajectory (user + assistant steps with tool_calls + tool obs)."""
    steps = []
    for i, call in enumerate(chain):
        step_idx = 2 * i + 1
        steps.append(
            AgentStep(
                step_index=step_idx,
                role="assistant",
                content=f"Calling {call.get('name', '?')}.",
                tool_calls=[{"name": call.get("name", ""), "arguments": call.get("arguments") or {}}],
            )
        )
        steps.append(
            AgentStep(
                step_index=step_idx + 1,
                role="tool",
                content=f"Observation for step {step_idx}.",
            )
        )
    return steps


class BackTranslator:
    """Reverse-engineer a user prompt from a sequence of tool calls."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.5) -> None:
        self.model = model
        self.temperature = temperature

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((ValueError,)),
        reraise=True,
    )
    def generate(
        self,
        tool_chain: list[dict],
        sample_id: str | None = None,
    ) -> SynthesizedDataPair:
        """Generate user_prompt from tool_chain; trajectory is derived from chain + mock observations."""
        if litellm is None:
            raise ImportError("litellm is required for BackTranslator. pip install litellm")
        sample_id = sample_id or uuid.uuid4().hex[:12]
        history_text = _chain_to_text(tool_chain)
        user_msg = f"Tool execution history:\n{history_text}\n\nGenerate the user query that would have led to this exact sequence."
        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": BACK_TRANSLATOR_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.temperature,
            max_tokens=1024,
        )
        content = response.choices[0].message.content  # type: ignore[union-attr]
        user_prompt = (content or "").strip().strip('"')
        trajectory = _chain_to_trajectory(tool_chain)
        return SynthesizedDataPair(
            id=sample_id,
            source_method="back_translation",
            user_prompt=user_prompt,
            trajectory=trajectory,
            execution_success=False,
            verifier_score=0.0,
        )
