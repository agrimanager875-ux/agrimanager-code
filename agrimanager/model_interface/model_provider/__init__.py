"""
Unified model interface for multiple providers.

This package provides a unified interface for different LLM providers.
Each provider is imported lazily via model_factory to avoid loading
unnecessary dependencies.
"""
