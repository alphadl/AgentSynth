# AgentSynth

A pipeline for synthesizing high-quality agent training data from scratch. It focuses on **cold start** and **data scarcity** in tool-using agent scenarios: instead of mining value from existing logs, AgentSynth generates SFT-style trajectories (user intent → reasoning + tool calls) and validates them with execution-based reject sampling.

## Design principles

- **Synthetic over manual** — Strong LLMs (e.g. GPT-4, Qwen, Claude) can produce more consistent chain-of-thought and tool-calling behavior than typical human annotators.
- **Execution as ground truth** — Any trajectory that fails to run correctly (syntax errors, hallucinated tools or arguments) is rejected before it enters the dataset.
- **Bidirectional generation** — Data is produced in two ways: **forward** (multi-teacher: scenario → full trajectory) and **back-translation** (valid tool chain → reverse-engineered user query).

## Pipeline overview

- **Pipe A — Forward synthesis**  
  Seed scenario → multiple teacher models → consensus or selection → best trajectory.

- **Pipe B — Back-translation**  
  Tool definitions → valid tool-call sequences → simulated execution → LLM generates the user query that would justify that sequence.

- **Pipe C — Reject sampling**  
  Replay each candidate trajectory in a sandbox; reject on syntax errors, unknown tools/arguments, or empty observation loops.

## Installation

```bash
cd AgentSynth
pip install -e .
```

Requires **Python 3.10+**. See `pyproject.toml` and `requirements.txt` for dependencies.

## Usage

```bash
agentsynth --help
agentsynth run   # full pipeline (placeholder until Phase 4)
```

## Project layout

```
AgentSynth/
├── src/agentsynth/
│   ├── core/           # Types and config (Pydantic models)
│   ├── teachers/       # Forward teachers and back-translator
│   ├── execution/      # Sandbox and trajectory validation
│   └── generators/     # Tool-chain builder
├── tests/
├── pyproject.toml
└── README.md
```

## Dependencies

- **pydantic** (≥2) — schema and validation  
- **litellm** — LLM calls (OpenAI, Anthropic, Qwen, etc.)  
- **tenacity** — retries  
- **click**, **rich** — CLI

Optional: `datasets` for HuggingFace dataset I/O. See `pyproject.toml` for dev tools (pytest, ruff, mypy).

## License

Apache-2.0.
