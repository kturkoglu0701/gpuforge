"""Tests for gpuforge.classify."""

from __future__ import annotations

import os

from gpuforge.classify import CATEGORIES, ProcessInfo, classify_cmdline, classify_process, parse_cmdline_string
from gpuforge.models import ProcessCategory


def test_ollama_is_gpu_llm() -> None:
    info = classify_cmdline("ollama", "ollama serve")
    assert info.category == ProcessCategory.LLM_LOCAL_GPU


def test_cursor_host() -> None:
    info = classify_cmdline("cursor", "/usr/share/cursor/cursor")
    assert info.category == ProcessCategory.AI_IDE_HOST


def test_cursor_extension_host() -> None:
    info = classify_cmdline("cursor", "cursor --type=extensionHost")
    assert info.category == ProcessCategory.AI_IDE_EXTENSION


def test_vscode_and_codium() -> None:
    assert classify_cmdline("code", "/usr/share/code/code").category == ProcessCategory.AI_IDE_HOST
    assert classify_cmdline("codium", "/usr/bin/codium").category == ProcessCategory.AI_IDE_HOST


def test_windsurf_zed_jetbrains() -> None:
    assert classify_cmdline("windsurf", "windsurf").category == ProcessCategory.AI_IDE_HOST
    assert classify_cmdline("zed", "/usr/bin/zed").category == ProcessCategory.AI_IDE_HOST
    assert classify_cmdline("pycharm", "pycharm").category == ProcessCategory.AI_IDE_HOST


def test_copilot_and_continue() -> None:
    info = classify_cmdline("node", "node /path/copilot-language-server --stdio")
    assert info.category == ProcessCategory.AI_IDE_EXTENSION
    assert classify_cmdline("node", "node continue.continue/dist/continue").category == ProcessCategory.AI_IDE_EXTENSION


def test_local_llm_stack() -> None:
    assert (
        classify_cmdline("python3", "python3 -m vllm.entrypoints.openai.api_server").category
        == ProcessCategory.LLM_LOCAL_GPU
    )
    assert classify_cmdline("tabby", "tabby serve").category in (
        ProcessCategory.LLM_LOCAL_GPU,
        ProcessCategory.LLM_LOCAL_CPU,
    )
    assert (
        classify_cmdline("llama-server", "llama-server -m model.gguf --gpu-layers 32").category
        == ProcessCategory.LLM_LOCAL_GPU
    )
    assert classify_cmdline("main", "./main -m model.gguf -ngl 0").category == ProcessCategory.LLM_LOCAL_CPU
    assert (
        classify_cmdline("ollama", "ollama run llama3 --device cpu").category
        == ProcessCategory.LLM_LOCAL_CPU
    )


def test_tsserver_is_indexer() -> None:
    info = classify_cmdline("node", "/usr/lib/node_modules/typescript/lib/tsserver.js --stdio")
    assert info.category == ProcessCategory.INDEXER


def test_indexers() -> None:
    assert classify_cmdline("rg", "rg --files /home").category == ProcessCategory.INDEXER
    assert classify_cmdline("rg", ()).category == ProcessCategory.INDEXER
    assert classify_cmdline("node", "node pyright-langserver --stdio").category == ProcessCategory.INDEXER
    assert classify_cmdline("node", "node eslint --stdin").category == ProcessCategory.INDEXER


def test_build_and_remote() -> None:
    assert classify_cmdline("npm", "npm run build").category == ProcessCategory.BUILD_TOOL
    assert (
        classify_cmdline("curl", "curl https://api.openai.com/v1/chat/completions").category
        == ProcessCategory.LLM_REMOTE_CLIENT
    )


def test_unknown() -> None:
    info = classify_cmdline("sleep", "sleep 10")
    assert info.category == ProcessCategory.UNKNOWN
    assert info.confidence == 0.0


def test_categories_constant() -> None:
    assert set(CATEGORIES) == {c.value for c in ProcessCategory}


def test_classify_process_live() -> None:
    info = classify_process(os.getpid())
    assert isinstance(info, ProcessInfo)
    assert info.pid == os.getpid()
    assert info.category in ProcessCategory


def test_parse_cmdline_string_helper() -> None:
    assert parse_cmdline_string("node tsserver.js --stdio") == ("node", "tsserver.js", "--stdio")
