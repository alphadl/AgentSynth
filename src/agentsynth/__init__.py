"""AgentSynth: Industrial-Grade Agent Data Synthesis Pipeline.

Generates high-quality SFT trajectories via Multi-Teacher Forward Synthesis
and Back-Translation with Reject Sampling.
"""

__version__ = "0.1.0"

from agentsynth.core.types import (
    AgentStep,
    GenerationConfig,
    SynthesizedDataPair,
    ToolDefinition,
    ToolParameter,
)

__all__ = [
    "__version__",
    "AgentStep",
    "GenerationConfig",
    "SynthesizedDataPair",
    "ToolDefinition",
    "ToolParameter",
]
