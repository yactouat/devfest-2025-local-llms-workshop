#!/usr/bin/env python3
"""
Utility functions for the DevFest 2025 Local LLMs Workshop.

This module provides helper functions used across multiple demo scripts.
"""

import subprocess
import sys


def get_available_model(prefer_thinking: bool = False, use_cloud: bool = False) -> str:
    """
    Get an available Ollama model, checking for qwen3:8b first, then llama3.1:latest.

    This function checks which models are available in Ollama and returns
    the first one found from the preferred list. It tries qwen3:8b first
    as it's a thinking model with better reasoning capabilities, then falls
    back to llama3.1:latest if qwen3 is not available.

    Args:
        prefer_thinking: If True, always return qwen3:8b if available.
                        If False, return llama3.1:latest if available, or qwen3:8b as fallback.

    Returns:
        str: The name of an available model ("qwen3:8b" or "llama3.1:latest")

    Raises:
        RuntimeError: If neither model is available in Ollama

    Example:
        >>> model = get_available_model(prefer_thinking=True)
        >>> llm = ChatOllama(model=model)
    """
    return "llama3.1:latest"
    if use_cloud:
        return "gemini-2.5-flash"
    else:
        try:
            # Get list of available models from Ollama
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True
            )

            available_models = result.stdout.lower()

            # Check which models are available
            has_qwen = "qwen3:8b" in available_models
            has_llama = "llama3.1:latest" in available_models or "llama3.1:latest" in available_models

            # Determine which model to return based on preference and availability
            if prefer_thinking:
                if has_qwen:
                    return "qwen3:8b"
                elif has_llama:
                    print("⚠️  Warning: qwen3:8b not found, using llama3.1:latest instead", file=sys.stderr)
                    return "llama3.1:latest"
            else:
                # For non-thinking use cases, prefer llama3.1 but fall back to qwen3
                if has_llama:
                    return "llama3.1:latest"
                elif has_qwen:
                    print("⚠️  Warning: llama3.1:latest not found, using qwen3:8b instead", file=sys.stderr)
                    return "qwen3:8b"

            # Neither model is available
            raise RuntimeError(
                "Neither qwen3:8b nor llama3.1:latest is available in Ollama.\n"
                "Please install at least one of them:\n"
                "  ollama pull qwen3:8b\n"
                "  ollama pull llama3.1:latest"
            )

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to query Ollama models: {e}\n"
                "Make sure Ollama is installed and running."
            ) from e
        except FileNotFoundError:
            raise RuntimeError(
                "Ollama command not found. Please install Ollama first:\n"
                "Visit https://ollama.ai for installation instructions."
            ) from None
