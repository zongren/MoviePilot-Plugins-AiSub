#!/usr/bin/env python3
"""Standalone CLI for the autosubv2 whole-file SRT translator.

Runs the exact same engine the MoviePilot plugin uses, without MoviePilot::

    export LLM_API_KEY=sk-...
    python3 tools/translate_srt.py "/path/to/movie.it.srt" \
        --api-type openai_chat --base-url https://api.deepseek.com \
        --model deepseek-flash --verify

The API key is only ever read from ``--api-key`` / the ``LLM_API_KEY``
environment variable and is never written anywhere.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "plugins" / "autosubv2"))

from translate.openai_translate import LLMError, create_provider  # noqa: E402
from translate.srt_engine import CHUNK_SIZE, read_srt_file, translate_srt_file  # noqa: E402


class ConsoleLogger:
    """Minimal logger compatible with the engine's ``info/warning/error`` API."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.started = time.time()

    def _emit(self, level: str, message: str) -> None:
        if self.quiet and level == "info":
            return
        elapsed = time.time() - self.started
        print(f"[{elapsed:8.1f}s] {level.upper():7s} {message}", file=sys.stderr, flush=True)

    def info(self, message: str) -> None:
        self._emit("info", message)

    def warning(self, message: str) -> None:
        self._emit("warning", message)

    warn = warning

    def error(self, message: str) -> None:
        self._emit("error", message)


def verify_timing(source: str, output: str) -> bool:
    """Check cue count, indexes and timestamp lines are identical."""
    src_doc = read_srt_file(source)
    out_doc = read_srt_file(output)
    src_sig = [(cue.index, cue.timestamp_line) for cue in src_doc.cues]
    out_sig = [(cue.index, cue.timestamp_line) for cue in out_doc.cues]
    if len(src_sig) != len(out_sig):
        print(f"校验失败：条目数不一致 源={len(src_sig)} 输出={len(out_sig)}", file=sys.stderr)
        return False
    mismatches = [
        (i, a, b) for i, (a, b) in enumerate(zip(src_sig, out_sig)) if a != b
    ]
    if mismatches:
        print(f"校验失败：{len(mismatches)} 条时间轴/序号不一致，前 5 条：{mismatches[:5]}", file=sys.stderr)
        return False
    print(f"校验通过：{len(src_sig)} 条字幕的序号与时间轴与源文件完全一致", file=sys.stderr)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Whole-file LLM SRT translation (autosubv2 engine, standalone).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="源 SRT 文件路径")
    parser.add_argument("-o", "--output", help="输出 SRT 路径（默认在源文件旁生成 .zh.机翻.srt）")
    parser.add_argument(
        "--api-type",
        choices=["openai_chat", "openai_responses", "anthropic_messages"],
        default=os.environ.get("LLM_API_TYPE", "openai_chat"),
        help="大模型接口类型",
    )
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-flash"))
    parser.add_argument("--effort", choices=["low", "high", "max"], default="high")
    parser.add_argument("--max-tokens", type=int, default=64000)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--timeout", type=int, default=900)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--zh-only", action="store_true", help="只输出中文译文")
    group.add_argument("--bilingual", action="store_true", help="输出译文+原文双语（默认）")
    parser.add_argument("--verify", action="store_true", help="翻译后校验时间轴与序号逐字节一致")
    parser.add_argument("--quiet", action="store_true", help="只输出警告和最终结果")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logger = ConsoleLogger(quiet=args.quiet)

    if not args.api_key:
        print("缺少 API Key：请设置 LLM_API_KEY 或传入 --api-key", file=sys.stderr)
        return 2
    if not os.path.isfile(args.input):
        print(f"输入文件不存在：{args.input}", file=sys.stderr)
        return 2

    source = os.path.abspath(args.input)
    output = args.output or f"{os.path.splitext(source)[0]}.zh.机翻.srt"

    try:
        provider = create_provider(
            args.api_type,
            args.base_url,
            args.api_key,
            args.model,
            reasoning_effort=args.effort,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    except (ValueError, LLMError) as exc:
        print(f"初始化大模型客户端失败：{exc}", file=sys.stderr)
        return 2

    logger.info(
        f"接口={args.api_type} base={args.base_url} model={args.model} "
        f"effort={args.effort} max_tokens={args.max_tokens} chunk_size={args.chunk_size}"
    )
    logger.info(f"源文件={source}")
    logger.info(f"输出文件={output}")

    started = time.time()
    try:
        result = translate_srt_file(
            source,
            output,
            provider,
            zh_only=args.zh_only,
            max_retries=args.max_retries,
            chunk_size=args.chunk_size,
            logger=logger,
        )
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130

    elapsed = time.time() - started
    print(
        "统计："
        f"总条目={result.total} 已翻译={result.translated} 未翻译={result.unrecovered} "
        f"请求={result.requests} 补译轮={result.repaired} 分块={result.chunked} "
        f"prompt_tokens={result.prompt_tokens} completion_tokens={result.completion_tokens} "
        f"耗时={elapsed:.1f}s"
    )
    if result.unrecovered:
        print(f"未翻译条目 id（前 20 个）：{result.unrecovered_ids[:20]}", file=sys.stderr)

    ok = True
    if args.verify:
        ok = verify_timing(source, output)
    return 0 if (ok and result.unrecovered == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
