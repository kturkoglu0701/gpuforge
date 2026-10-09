"""GPUForge — GPU-first resource orchestrator for AI IDEs and local LLM runtimes."""

from gpuforge.classify import ProcessInfo, classify_cmdline, classify_process

__version__ = "0.3.5"
__all__ = ["ProcessInfo", "classify_cmdline", "classify_process", "__version__"]
