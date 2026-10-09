"""Classify OS processes by command-line patterns (IDE, LLM, indexer, build)."""

from __future__ import annotations

import re
import shlex
from typing import Literal, Sequence

import psutil

from gpuforge.models import ProcessCategory, ProcessInfo

Category = Literal[
    "ai_ide_host",
    "ai_ide_extension",
    "llm_local_gpu",
    "llm_local_cpu",
    "llm_remote_client",
    "indexer",
    "build_tool",
    "unknown",
]

CATEGORIES: tuple[Category, ...] = (
    "ai_ide_host",
    "ai_ide_extension",
    "llm_local_gpu",
    "llm_local_cpu",
    "llm_remote_client",
    "indexer",
    "build_tool",
    "unknown",
)

_CATEGORY_TO_ENUM: dict[Category, ProcessCategory] = {
    "ai_ide_host": ProcessCategory.AI_IDE_HOST,
    "ai_ide_extension": ProcessCategory.AI_IDE_EXTENSION,
    "llm_local_gpu": ProcessCategory.LLM_LOCAL_GPU,
    "llm_local_cpu": ProcessCategory.LLM_LOCAL_CPU,
    "llm_remote_client": ProcessCategory.LLM_REMOTE_CLIENT,
    "indexer": ProcessCategory.INDEXER,
    "build_tool": ProcessCategory.BUILD_TOOL,
    "unknown": ProcessCategory.UNKNOWN,
}


def _norm_text(name: str, cmdline: Sequence[str]) -> str:
    joined = " ".join(cmdline) if cmdline else name
    return f"{name} {joined}".lower()


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


def _gpu_hints(text: str) -> bool:
    gpu_markers = (
        "--gpu-layers",
        "-ngl",
        "cuda",
        "cublas",
        "vulkan",
        "metal",
        "--device cuda",
        "--tensor-parallel",
        "nvidia",
    )
    if _has_any(text, gpu_markers):
        return True
    return bool(re.search(r"\b-ngl\s+[1-9]", text))


def _cpu_only_llm_hints(text: str) -> bool:
    return _has_any(
        text,
        (
            "--gpu-layers 0",
            "-ngl 0",
            "--n-gpu-layers 0",
            "cpu only",
            "--device cpu",
        ),
    )


def _is_ide_extension(text: str) -> bool:
    extension_markers = (
        "extension-host",
        "extensionhost",
        "--type=extension",
        "copilot-language-server",
        "github.copilot",
        "continue.continue",
        "language-server",
        "typescript-language-features",
    )
    return _has_any(text, extension_markers)


def _is_llama_binary(text: str, name: str) -> bool:
    base = name.lower()
    if _has_any(
        text,
        (
            "llama.cpp",
            "llama-cli",
            "llama-server",
            "llama_server",
        ),
    ):
        return True
    if base in ("llama-server", "llama-cli", "main", "server") and re.search(
        r"(?:^|\s)-m\s", text
    ):
        return True
    return False


def _is_build_tool(text: str) -> bool:
    build_markers = (
        "webpack",
        "vite build",
        "esbuild",
        "rollup",
        "tsc -b",
        "tsc --build",
        "cargo build",
        "cmake --build",
        "make -j",
        "ninja -c",
        "bazel build",
        "gradle build",
        "npm run build",
        "pnpm run build",
        "yarn run build",
    )
    return _has_any(text, build_markers)


def _is_remote_llm_client(text: str) -> bool:
    remote_markers = (
        "api.openai.com",
        "openai.azure.com",
        "anthropic.com",
        "generativelanguage.googleapis.com",
        "openrouter.ai",
        "together.xyz",
        "api.cohere.com",
        "huggingface.co/api",
        "chatgpt",
        "claude-cli",
    )
    return _has_any(text, remote_markers)


def _classify_rules(name: str, argv: Sequence[str]) -> tuple[Category, float]:
    text = _norm_text(name, argv)

    if _has_any(text, ("vllm", "vllm.entrypoints", "python -m vllm")):
        return "llm_local_gpu", 0.92

    if _has_any(text, ("tabby", "tabby serve", "tabby-agent")):
        cat = "llm_local_gpu" if _gpu_hints(text) or not _cpu_only_llm_hints(text) else "llm_local_cpu"
        conf = 0.85 if cat == "llm_local_gpu" else 0.8
        return cat, conf

    if _has_any(text, ("ollama", "ollama serve", "ollama runner")):
        if _cpu_only_llm_hints(text):
            return "llm_local_cpu", 0.88
        return "llm_local_gpu", 0.75

    if _is_llama_binary(text, name):
        if _cpu_only_llm_hints(text) or not _gpu_hints(text):
            return "llm_local_cpu", 0.82
        return "llm_local_gpu", 0.88

    if "copilot-language-server" in text:
        return "ai_ide_extension", 0.95

    if _has_any(text, ("continue", "continuedev", "@continuedev")):
        if _is_remote_llm_client(text):
            return "llm_remote_client", 0.78
        return "ai_ide_extension", 0.84

    if _is_ide_extension(text) and _has_any(
        text,
        ("cursor", "code", "codium", "windsurf", "zed", "jetbrains", "idea", "pycharm", "webstorm"),
    ):
        return "ai_ide_extension", 0.9

    if _has_any(text, ("ripgrep", " rg ", "/rg", "\\rg.exe", "rg --")) or re.search(
        r"(?:^|[\s/\\])rg(?:\s|$)", text
    ):
        return "indexer", 0.93

    if "pyright" in text or "pyright-langserver" in text:
        return "indexer", 0.9

    if "eslint" in text and _has_any(text, ("eslint", "eslint-server", "--stdin")):
        return "indexer", 0.88

    if "tsserver" in text or "typescript-language-server" in text:
        return "indexer", 0.9

    if _is_build_tool(text):
        return "build_tool", 0.85

    if _is_remote_llm_client(text):
        return "llm_remote_client", 0.8

    if _has_any(
        text,
        (
            "cursor",
            "/cursor",
            "\\cursor.exe",
            "cursor.app",
        ),
    ):
        return "ai_ide_host", 0.9

    if _has_any(
        text,
        (
            " visual studio code",
            "/code ",
            "\\code.exe",
            "electron .",
            "vscodium",
            "/codium",
            "\\codium.exe",
        ),
    ) or re.search(r"(?:^|[\s/\\])code(?:\s|$)", text):
        return "ai_ide_host", 0.88

    if _has_any(text, ("windsurf", "codeium windsurf", "/windsurf")):
        return "ai_ide_host", 0.88

    if _has_any(text, (" zed ", "/zed", "\\zed.exe", "zed-editor")):
        return "ai_ide_host", 0.88

    if re.search(
        r"(?:jetbrains|intellij|(?:^|[\s/\\])idea(?:\.sh|\s|$)|pycharm|webstorm|"
        r"goland|clion|rider|datagrip|phpstorm|rubymine|fleet)",
        text,
    ):
        return "ai_ide_host", 0.86

    return "unknown", 0.0


def classify_cmdline(
    name: str,
    cmdline: str | Sequence[str] | None = None,
    *,
    pid: int = 0,
) -> ProcessInfo:
    """
    Classify a process from its executable name and argv (or a shell-style cmdline string).

    Returns ProcessInfo with category and confidence in [0, 1].
    """
    if cmdline is None:
        argv: tuple[str, ...] = ()
    elif isinstance(cmdline, str):
        argv = tuple(shlex.split(cmdline)) if cmdline.strip() else ()
    else:
        argv = tuple(cmdline)

    cat, confidence = _classify_rules(name, argv)
    cmdline_str = " ".join(argv) if argv else name
    return ProcessInfo(
        pid=pid,
        name=name,
        cmdline=cmdline_str,
        category=_CATEGORY_TO_ENUM[cat],
        confidence=confidence,
    )


def classify_process(pid: int) -> ProcessInfo:
    """Classify a live process by PID using psutil."""
    proc = psutil.Process(pid)
    name = proc.name() or ""
    try:
        argv = tuple(proc.cmdline())
    except (psutil.AccessDenied, psutil.ZombieProcess):
        argv = ()
    info = classify_cmdline(name, argv, pid=pid)
    return info


def parse_cmdline_string(cmd: str) -> tuple[str, ...]:
    """Helper for tests: split a shell-style command string into argv."""
    return tuple(shlex.split(cmd))


def scan_processes(user_only: bool = True) -> list[ProcessInfo]:
    """Enumerate processes and return classified ProcessInfo rows."""
    uid = psutil.Process().uids().real if user_only else None
    out: list[ProcessInfo] = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if user_only and proc.uids().real != uid:
                continue
            pid = proc.pid
            name = proc.info.get("name") or proc.name() or ""
            try:
                argv = tuple(proc.cmdline())
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                argv = ()
            out.append(classify_cmdline(name, argv, pid=pid))
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return out
