"""Miracle Agent - Minimal AI coding agent inspired by Claude Code's architecture."""

__version__ = "0.4.0"

from miracle_agent.agent import Agent
from miracle_agent.llm import LLM
from miracle_agent.config import Config
from miracle_agent.tools import ALL_TOOLS

__all__ = ["Agent", "LLM", "Config", "ALL_TOOLS", "__version__"]
