"""CLI entry point for AgentSynth."""

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="agentsynth")
def main() -> None:
    """AgentSynth: Industrial-Grade Agent Data Synthesis Pipeline."""
    pass


@main.command()
def run() -> None:
    """Run the full synthesis pipeline (Generate -> Validate -> Filter -> Save)."""
    click.echo("Pipeline not yet implemented. Use Phase 4 to integrate.")


if __name__ == "__main__":
    main()
