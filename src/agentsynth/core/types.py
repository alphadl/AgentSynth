"""Core Pydantic data models for AgentSynth.

Defines the data contract for the entire pipeline:
  ToolDefinition -> AgentStep -> SynthesizedDataPair, GenerationConfig
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# --- Tool Definitions ---


class ToolParameter(BaseModel):
    """Schema for a single parameter of a tool."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (e.g. string, integer, array)")
    description: str = Field(..., description="Human-readable description")
    required: bool = Field(..., description="Whether this parameter is required")


class ToolDefinition(BaseModel):
    """Schema for a single tool available to the agent."""

    name: str = Field(..., description="Tool identifier / API name")
    description: str = Field(..., description="What the tool does")
    parameters: list[ToolParameter] = Field(
        default_factory=list,
        description="List of parameters for this tool",
    )


# --- Trajectory Components ---


class AgentStep(BaseModel):
    """A single step in a synthesized or executed agent trajectory."""

    step_index: int = Field(..., description="Order of this step in the trajectory")
    role: Literal["assistant", "user", "tool"] = Field(
        ...,
        description=(
            "Who produced this step: assistant (CoT + tool_calls), user, or tool (observation)"
        ),
    )
    content: str = Field(
        ...,
        description="The thought process (CoT), user message, or tool output",
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description="Structured tool calls if this step is from the assistant and invokes tools",
    )


class SynthesizedDataPair(BaseModel):
    """A single (user_prompt, trajectory) pair produced by forward or back-translation synthesis."""

    id: str = Field(..., description="Unique identifier for this sample")
    source_method: Literal["forward_teacher", "back_translation"] = Field(
        ...,
        description="Whether this came from multi-teacher forward or back-translation",
    )
    user_prompt: str = Field(..., description="The user query / scenario")
    trajectory: list[AgentStep] = Field(
        ...,
        description="Ordered list of agent steps (thoughts, tool calls, observations)",
    )
    execution_success: bool = Field(
        ...,
        description="Did the tools actually run without error (Reject Sampling result)",
    )
    verifier_score: float = Field(
        default=0.0,
        description="Optional quality score from a judge model",
    )


# --- Configuration ---


class GenerationConfig(BaseModel):
    """Configuration for the synthesis pipeline (teachers, limits, sampling)."""

    teacher_models: list[str] = Field(
        default_factory=lambda: ["gpt-4o", "qwen-max"],
        description="Model IDs for multi-teacher forward synthesis",
    )
    back_translation_model: str = Field(
        default="gpt-4o",
        description="Model used for back-translation (tool chain -> user prompt)",
    )
    max_steps: int = Field(
        default=15,
        ge=1,
        le=100,
        description="Maximum steps per trajectory",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation",
    )
