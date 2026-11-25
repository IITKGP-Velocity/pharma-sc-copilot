"""Agent component: Planner -> Executor -> Critic loop on the course model.

engine.py  the Engine: rules -> retrieval -> planner (LLM) -> tools -> critic, with step cap and metrics
tools.py   deterministic tools the planner can call (MKT, cost delta, demurrage, ...)
llm.py     chat and embeddings clients for the course endpoints (per-port CA, retries, token counts)
"""
from copilot.agent.engine import Engine

__all__ = ["Engine"]
