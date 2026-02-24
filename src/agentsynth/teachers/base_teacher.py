"""Forward teacher: scenario + tool defs -> SynthesizedDataPair via LLM."""

from __future__ import annotations

import json
import logging
import uuid

from agentsynth.core.types import AgentStep, SynthesizedDataPair, ToolDefinition
from agentsynth.teachers.prompts import FORWARD_TEACHER_SYSTEM
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

try:
    import litellm
except ImportError:
    litellm = None  # type: ignore[assignment]


def _tools_to_text(tools: list[ToolDefinition]) -> str:
    lines = []
    for t in tools:
        params = ", ".join(f"{p.name}({p.type}, required={p.required})" for p in t.parameters)
        lines.append(f"- {t.name}: {t.description}. Params: {params}")
    return "\n".join(lines)


def _parse_trajectory(raw: str | None) -> list[AgentStep]:
    """Parse LLM JSON output into list of AgentStep."""
    if not raw or not raw.strip():
        return []
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw)
    if not isinstance(data, list):
        return []
    steps = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            continue
        step_index = obj.get("step_index", i + 1)
        role = obj.get("role", "assistant")
        if role not in ("assistant", "user", "tool"):
            role = "assistant"
        content = obj.get("content") or ""
        tool_calls = obj.get("tool_calls")
        if tool_calls is not None and not isinstance(tool_calls, list):
            tool_calls = None
        steps.append(
            AgentStep(
                step_index=int(step_index),
                role=role,
                content=str(content),
                tool_calls=tool_calls,
            )
        )
    return steps


class BaseTeacher:
    """Generate a single trajectory from a scenario using an LLM."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_steps: int = 15,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_steps = max_steps

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((json.JSONDecodeError, ValueError)),
        reraise=True,
    )
    def generate(
        self,
        scenario: str,
        tools: list[ToolDefinition],
        sample_id: str | None = None,
    ) -> SynthesizedDataPair:
        """Generate one (user_prompt, trajectory) pair. execution_success left False (set by validator)."""
        if litellm is None:
            raise ImportError("litellm is required for BaseTeacher. pip install litellm")
        sample_id = sample_id or uuid.uuid4().hex[:12]
        tools_text = _tools_to_text(tools)
        user_msg = f"Scenario:\n{scenario}\n\nAvailable tools:\n{tools_text}\n\nGenerate a trajectory (JSON array of steps)."
        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": FORWARD_TEACHER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.temperature,
            max_tokens=4096,
        )
        if not response.choices:
            raise ValueError("LLM returned no choices")
        content = getattr(response.choices[0].message, "content", None)
        trajectory = _parse_trajectory(content)
        if not trajectory:
            trajectory = [
                AgentStep(step_index=1, role="user", content=scenario),
                AgentStep(step_index=2, role="assistant", content="(parse failed)", tool_calls=None),
            ]
        return SynthesizedDataPair(
            id=sample_id,
            source_method="forward_teacher",
            user_prompt=scenario,
            trajectory=trajectory,
            execution_success=False,
            verifier_score=0.0,
        )
