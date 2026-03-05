"""ACE LLM providers — client wrappers for LLM APIs.

- ``LiteLLMClient`` / ``LiteLLMConfig`` — LiteLLM integration (100+ providers)
- ``InstructorClient`` / ``wrap_with_instructor`` — Instructor structured outputs
- ``LangChainLiteLLMClient`` — LangChain + LiteLLM (optional)
- ``ClaudeCodeLLMClient`` / ``ClaudeCodeLLMConfig`` — Claude Code CLI (optional)
"""

from __future__ import annotations

from typing import Optional

from .config import ACEModelConfig, ModelConfig, load_config, save_config
from .instructor import InstructorClient, wrap_with_instructor
from .litellm import LiteLLMClient, LiteLLMConfig, LLMResponse
from .registry import ValidationResult, search_models, validate_connection

# Optional providers — soft imports -------------------------------------------

LangChainLiteLLMClient: Optional[type]
try:
    from .langchain import LangChainLiteLLMClient as _LC  # type: ignore[assignment]

    LangChainLiteLLMClient = _LC  # type: ignore[assignment]
except ImportError:
    LangChainLiteLLMClient = None  # type: ignore[assignment]

ClaudeCodeLLMClient: Optional[type]
ClaudeCodeLLMConfig: Optional[type]
CLAUDE_CODE_CLI_AVAILABLE: bool = False
try:
    from .claude_code import (
        CLAUDE_CODE_CLI_AVAILABLE as _CC_AVAILABLE,
        ClaudeCodeLLMClient as _CC,
        ClaudeCodeLLMConfig as _CCConfig,
    )

    ClaudeCodeLLMClient = _CC  # type: ignore[assignment]
    ClaudeCodeLLMConfig = _CCConfig  # type: ignore[assignment]
    CLAUDE_CODE_CLI_AVAILABLE = _CC_AVAILABLE
except ImportError:
    ClaudeCodeLLMClient = None  # type: ignore[assignment]
    ClaudeCodeLLMConfig = None  # type: ignore[assignment]

__all__ = [
    # Config
    "ModelConfig",
    "ACEModelConfig",
    "load_config",
    "save_config",
    # Registry
    "ValidationResult",
    "validate_connection",
    "search_models",
    # LiteLLM
    "LiteLLMClient",
    "LiteLLMConfig",
    "LLMResponse",
    # Instructor
    "InstructorClient",
    "wrap_with_instructor",
    # LangChain (optional)
    "LangChainLiteLLMClient",
    # Claude Code CLI (optional)
    "ClaudeCodeLLMClient",
    "ClaudeCodeLLMConfig",
    "CLAUDE_CODE_CLI_AVAILABLE",
]
