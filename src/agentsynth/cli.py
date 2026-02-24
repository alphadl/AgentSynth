"""CLI entry point for AgentSynth."""

import json
import sys
from pathlib import Path

import click

from agentsynth.core.types import GenerationConfig, ToolDefinition
from agentsynth.pipeline import Pipeline


@click.group()
@click.version_option(version="0.1.0", prog_name="agentsynth")
def main() -> None:
    """AgentSynth: Industrial-Grade Agent Data Synthesis Pipeline."""
    pass


def _load_tools(path: Path) -> list[ToolDefinition]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        click.echo(f"Invalid JSON in {path}: {e}", err=True)
        sys.exit(1)
    except OSError as e:
        click.echo(f"Cannot read {path}: {e}", err=True)
        sys.exit(1)
    try:
        if isinstance(data, list):
            return [ToolDefinition.model_validate(t) for t in data]
        return [ToolDefinition.model_validate(data)]
    except Exception as e:
        click.echo(f"Invalid tool schema in {path}: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--tools", "-t", type=click.Path(path_type=Path, exists=True), required=True, help="JSON file of tool definitions.")
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Output JSONL path.")
@click.option("--mode", type=click.Choice(["forward", "back"]), default="back", help="forward = scenario -> trajectory; back = chain -> prompt.")
@click.option("--scenarios", type=click.Path(path_type=Path, exists=True), help="JSON/JSONL of scenarios (for forward mode).")
@click.option("--num-samples", "-n", default=1, help="Number of samples (back mode).")
@click.option("--model", default=None, help="Override model (e.g. gpt-4o).")
def run(
    tools: Path,
    output: Path,
    mode: str,
    scenarios: Path | None,
    num_samples: int,
    model: str | None,
) -> None:
    """Run synthesis: Generate -> Validate -> Filter -> Save to JSONL."""
    config = GenerationConfig()
    if model:
        config.teacher_models = [model]
        config.back_translation_model = model
    pipeline = Pipeline(config)
    pipeline.set_tools(_load_tools(tools))

    if mode == "forward":
        if not scenarios:
            click.echo("--scenarios required for forward mode.", err=True)
            raise SystemExit(1)
        text = scenarios.read_text(encoding="utf-8")
        try:
            scenario_list = json.loads(text)
        except json.JSONDecodeError:
            scenario_list = [line.strip() for line in text.splitlines() if line.strip()]
        if isinstance(scenario_list, str):
            scenario_list = [scenario_list]
        pairs = [pipeline.run_forward(s) for s in scenario_list]
    else:
        pairs = pipeline.run_back_translation(num_samples=num_samples)

    accepted = [p for p in pairs if p.execution_success]
    if accepted:
        pipeline.save_jsonl(accepted, output)
        click.echo(f"Generated {len(pairs)}, accepted {len(accepted)}, saved to {output}")
    else:
        click.echo(f"Generated {len(pairs)}, accepted 0 (no samples written to {output})")
