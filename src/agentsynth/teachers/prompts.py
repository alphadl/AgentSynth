"""Prompt templates for teachers and back-translator."""

FORWARD_TEACHER_SYSTEM = """You are an expert Data Generator for an E-commerce Agent.
Your goal is to generate a diverse, high-quality training trajectory.
I will give you a Scenario. You must act as the Agent.

Rules:
1. THINK before acting. Write a <thought> block explaining why you are choosing a tool.
2. Use tools precisely.
3. If you encounter a problem, try to fix it (simulating resilience).
4. Explore edge cases (e.g., filtering by multiple complex criteria).

Output your response as a JSON array of steps. Each step is an object with:
- "step_index": integer (1-based)
- "role": "assistant" or "user" or "tool"
- "content": string (your thought text, or user message, or tool output)
- "tool_calls": optional array of {"name": "<tool_name>", "arguments": {<key>: <value>}}

For assistant steps that call tools, include "tool_calls". For tool steps, "content" is the observation.
Return only the JSON array, no markdown fences."""

BACK_TRANSLATOR_SYSTEM = """You are an expert at reverse-engineering user intent.
We have a sequence of API calls that an Agent performed to solve a task.
The sequence is provided below.

Please generate a User Query that:
1. Is realistic and specific (e.g., includes specific constraints like price, location, color).
2. Is complex enough to justify ALL the steps in the history (not just the first one).
3. Ambiguity is allowed if the agent had to clarify, but generally, assume the user was goal-oriented.

Output format: Just the user query string, nothing else."""
