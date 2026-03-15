"""Pipeline: generate -> validate -> filter -> save."""

from __future__ import annotations

import logging
from pathlib import Path

from agentsynth.core.types import (
    GenerationConfig,
    SynthesizedDataPair,
    ToolDefinition,
)
from agentsynth.execution import DummyToolRegistry, validate_trajectory
from agentsynth.generators import build_chains
from agentsynth.teachers import BackTranslator, BaseTeacher

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrate forward or back-translation synthesis with reject sampling."""

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()
        self.registry = DummyToolRegistry()

    def set_tools(self, tools: list[ToolDefinition]) -> None:
        self.registry = DummyToolRegistry(tools)

    def run_forward(self, scenario: str) -> SynthesizedDataPair:
        """One forward sample: scenario -> BaseTeacher -> validate -> set execution_success."""
        teacher = BaseTeacher(
            model=self.config.teacher_models[0],
            temperature=self.config.temperature,
            max_steps=self.config.max_steps,
        )
        tools_list = [self.registry.get_tool(n) for n in self.registry.list_tool_names()]
        tools_list = [t for t in tools_list if t is not None]
        pair = teacher.generate(scenario, tools_list, None)
        ok, _ = validate_trajectory(pair.trajectory, self.registry)
        return pair.model_copy(update={"execution_success": ok})

    def run_back_translation(self, num_samples: int = 1) -> list[SynthesizedDataPair]:
        """Generate chains, back-translate each, validate, return with execution_success set."""
        tools_list = [self.registry.get_tool(n) for n in self.registry.list_tool_names()]
        tools_list = [t for t in tools_list if t is not None]
        if not tools_list:
            return []
        chains = build_chains(
            tools_list,
            max_length=min(5, self.config.max_steps),
            num_chains=num_samples,
        )
        bt = BackTranslator(
            model=self.config.back_translation_model,
            temperature=self.config.temperature,
        )
        results = []
        for chain in chains:
            pair = bt.generate(chain, None)
            ok, _ = validate_trajectory(pair.trajectory, self.registry)
            results.append(pair.model_copy(update={"execution_success": ok}))
        return results

    def save_jsonl(self, pairs: list[SynthesizedDataPair], path: str | Path) -> None:
        """Append pairs to a JSONL file (one JSON object per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for p in pairs:
                f.write(p.model_dump_json() + "\n")
        logger.info("Wrote %d samples to %s", len(pairs), path)
