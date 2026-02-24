AgentSynth: Industrial-Grade Agent Data Synthesis Pipeline1. Project ManifestoAgentSynth is the "Static Data Scaling" engine for the Aoxia Agent ecosystem.Unlike AgentHER (which recovers value from failed live logs), AgentSynth generates high-quality SFT trajectories ex nihilo (from scratch) to solve the "Cold Start" and "Data Scarcity" problems in B2B/E-commerce agent training.Core Philosophy:Synthetic > Human: GPT-4/Qwen-Max generates cleaner, more consistent CoT (Chain-of-Thought) and tool calls than average human annotators.Execution is Truth: Any synthesized trajectory that fails to execute (runtime error, hallucinations) is immediately rejected (Reject Sampling).Bidirectional Generation: We generate data in two directions:Forward (Multi-Teacher): User Intent $\rightarrow$ Expert Agent Execution.Backward (Back-translation): Valid Tool Chain $\rightarrow$ Reverse-Engineered User Intent.2. Architecture & Modules2.1. The "Factory Floor" (Core Pipelines)The system consists of three main processing pipes:Pipe A: Multi-Teacher Forward SynthesisInput: A seed scenario (e.g., "Find a supplier for screws").Process: Route the request to multiple "Teacher" models (e.g., gpt-4o, qwen-max, claude-3-opus).Consensus/Selection: Generate $N$ trajectories.Output: The most comprehensive trajectory.Pipe B: Back-Translation (The "Reasoning" Inverter)Input: A set of Tool Definitions (Schema).Process:Chain Generator: Randomly or heuristically assemble a valid chain of tool calls (e.g., search_item $\rightarrow$ get_detail $\rightarrow$ add_to_cart).Simulator: Execute these tools to get real observations (or mock them if using GenTool).Back-Translator: Use an LLM to look at this chain of actions and ask: "What would a user ask to trigger this exact sequence?"Output: A complex User Prompt paired with the generated Trajectory.Pipe C: Reject Sampling (The Quality Gate)Validator: Re-runs every generated trajectory against a Sandbox or Mock Environment.Rules:Syntax Error $\rightarrow$ Reject.Hallucinated Tool Name $\rightarrow$ Reject.Hallucinated Argument (key not in schema) $\rightarrow$ Reject.Empty Observation Loop $\rightarrow$ Reject.3. Tech Stack & DependenciesLanguage: Python 3.10+LLM Interface: litellm (Crucial for easy swapping between Qwen, OpenAI, Anthropic).Data Validation: pydantic (Strict schema enforcement is non-negotiable).Retry Logic: tenacity.Dataset Format: datasets (HuggingFace) or jsonl.4. Detailed Data Structures (Pydantic)The AI assistant must implement these models first to establish the data contract.Pythonfrom pydantic import BaseModel, Field, conlist
from typing import List, Dict, Any, Optional, Literal

# --- Tool Definitions ---
class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: List[ToolParameter]

# --- Trajectory Components ---
class AgentStep(BaseModel):
    step_index: int
    role: Literal["assistant", "user", "tool"]
    content: str = Field(..., description="The thought process or tool output")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Structured tool calls if applicable")

class SynthesizedDataPair(BaseModel):
    id: str
    source_method: Literal["forward_teacher", "back_translation"]
    user_prompt: str
    trajectory: List[AgentStep]
    
    # Metadata for Reject Sampling
    execution_success: bool = Field(..., description="Did the tools actually run without error?")
    verifier_score: float = Field(..., description="Optional quality score from a judge")

# --- Configuration ---
class GenerationConfig(BaseModel):
    teacher_models: List[str] = ["gpt-4o", "qwen-max"]
    back_translation_model: str = "gpt-4o"
    max_steps: int = 15
    temperature: float = 0.7
5. Implementation Roadmap for CursorPhase 1: The Core Generator (Forward)Setup: Initialize poetry or pip project. Install litellm, pydantic, tenacity.Teacher Module: Create src/teachers/base_teacher.py. Implement a class that takes a list of ToolDefinition and a UserPrompt, and outputs a SynthesizedDataPair.Constraint: The Teacher MUST produce structured thoughts (CoT) before calling tools.Phase 2: The Reject Sampler (Execution Sandbox)Executor: Create src/execution/sandbox.py. This needs a DummyToolRegistry for testing.Validation Logic: Implement validate_trajectory(trajectory, tool_registry). It should check if every tool_call matches the schema of the tool_registry.Constraint: Use Pydantic's validate_call or similar logic to strictly check types.Phase 3: The Back-Translator (Reverse Engineering)Chain Synthesizer: Create src/generators/chain_builder.py. It should programmatically create valid sequences of tool calls (e.g., [Search -> Filter -> Sort]).Reverse Prompting: Create src/teachers/back_translator.py.Prompt Template: "You are an expert user emulator. I will show you a sequence of actions taken by an AI assistant. Your job is to write the specific, complex, and potentially messy user request that would have necessitated this exact sequence of actions."Phase 4: Pipeline IntegrationCreate main.py that runs the full loop:Generate Batch $\rightarrow$ Execute/Validate $\rightarrow$ Filter $\rightarrow$ Save to JSONL.6. Key System Prompts (For Reference)System Prompt for Back-Translation:PlaintextYou are an expert at reverse-engineering user intent. 
We have a sequence of API calls that an Agent performed to solve a task.
The sequence is:
{tool_execution_history}

Please generate a User Query that:
1. Is realistic and specific (e.g., includes specific constraints like price, location, color).
2. Is complex enough to justify ALL the steps in the history (not just the first one).
3. Ambiguity is allowed if the agent had to clarify, but generally, assume the user was goal-oriented.

Output format: Just the user query string.
System Prompt for Forward Teacher (Data Synthesis):PlaintextYou are an expert Data Generator for an E-commerce Agent.
Your goal is to generate a diverse, high-quality training trajectory.
I will give you a Scenario. You must act as the Agent.

Rules:
1. THINK before acting. Write a <thought> block explaining why you are choosing a tool.
2. Use tools precisely. 
3. If you encounter a problem, try to fix it (simulating resilience).
4. Explore edge cases (e.g., filtering by multiple complex criteria).