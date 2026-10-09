"""Whole-file, id-keyed LLM subtitle translation engine.

Design goals
------------
* Timestamps and cue indexes are copied verbatim from the source SRT.  The
  model only ever sees ``[{"n": 1, "t": "..."}]`` and only ever returns
  ``n`` -> translated text, so it can never invent or shift a timing.
* One request covers the whole file (JSON array keyed by a stable sequential
  cue id).  Missing/blank ids are repaired with follow-up requests; if the
  file still is not complete (e.g. the response was truncated by the output
  token limit) the remaining cues are retried in fixed-size chunks.
* Pure standard library: no MoviePilot, ``srt`` or ``openai`` import, so the
  engine can be unit-tested and driven from a standalone CLI.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

#: Fallback chunk size (internal constant, deliberately not user facing).
CHUNK_SIZE = 300

#: Encoding fallback order for reading SRT files.
ENCODINGS: Tuple[str, ...] = ("utf-8-sig", "utf-8", "gbk", "latin-1")

TIMESTAMP_RE = re.compile(
    r"^\s*(?P<start>\d{1,4}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,4}:\d{2}:\d{2}[,.]\d{1,3})\s*(?P<settings>.*?)\s*$"
)

_ID_KEYS = ("n", "id", "index", "idx", "number", "no")
_TEXT_KEYS = ("t", "text", "translation", "translated", "zh", "译文", "翻译", "content")

SYSTEM_PROMPT = (
    "你是一名专业的字幕翻译。用户会给你一个 JSON 数组，数组中每个对象形如 "
    '{"n": 1, "t": "原文"}。请把每个对象的 t 字段翻译成自然、口语化的简体中文：\n'
    "1. 严格保持 n 字段不变，逐条一一对应，不要合并、拆分、增删或遗漏任何条目；\n"
    "2. 完整保留原文中的 <i></i>、<b></b>、[ ]、( )、♪、- 等标记与括号，保留人名、地名等专有名词；\n"
    "3. 若某条内容本身无需翻译（例如纯符号、拟声词、语气词），原样返回；\n"
    "4. 只输出 JSON 数组本身，不要输出 markdown 代码块、解释说明或任何多余文字。"
)


class SrtError(Exception):
    """Raised when an SRT file cannot be read."""


# ---------------------------------------------------------------------------
# SRT parsing / composing
# ---------------------------------------------------------------------------


@dataclass
class Cue:
    """One subtitle cue.  ``start``/``end``/``index``/``timestamp_line`` are
    kept exactly as they appeared in the source file."""

    id: int
    index: str
    start: str
    end: str
    settings: str
    text: str
    timestamp_line: str
    translation: Optional[str] = None


@dataclass
class RawBlock:
    """A block that is not a subtitle cue; re-emitted untouched."""

    text: str


@dataclass
class SrtDocument:
    items: List[Any] = field(default_factory=list)
    cues: List[Cue] = field(default_factory=list)
    trailing_newline: bool = True


def read_text_file(path: str) -> str:
    """Read a text file trying a series of encodings."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    last_error: Optional[Exception] = None
    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError) as exc:  # pragma: no cover - defensive
            last_error = exc
    # latin-1 never fails, so this is unreachable in practice.
    raise SrtError(f"无法解码字幕文件 {path}: {last_error}")


def parse_srt(text: str) -> SrtDocument:
    """Parse SRT text into a document of cues and passthrough blocks.

    Cue ids are reassigned as a stable sequential 1..N sequence so that a
    duplicated or missing index in the source file can never confuse the
    model <-> cue mapping.
    """
    if text.startswith("\ufeff"):
        text = text[1:]
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    doc = SrtDocument(trailing_newline=normalized.endswith("\n"))

    for block in re.split(r"\n[ \t]*\n", normalized):
        if not block.strip():
            continue
        # Consecutive blank lines leave leading/trailing newlines on a block.
        block = block.strip("\n")
        lines = block.split("\n")
        ts_index = None
        for i, line in enumerate(lines):
            if TIMESTAMP_RE.match(line):
                ts_index = i
                break
        # Not a cue, or an unexpected layout: keep it verbatim, never drop it.
        if ts_index is None or ts_index > 1:
            doc.items.append(RawBlock(block))
            continue

        match = TIMESTAMP_RE.match(lines[ts_index])
        index = lines[0] if ts_index == 1 else ""
        cue = Cue(
            id=0,
            index=index,
            start=match.group("start"),
            end=match.group("end"),
            settings=match.group("settings"),
            text="\n".join(lines[ts_index + 1:]),
            timestamp_line=lines[ts_index],
        )
        doc.cues.append(cue)
        doc.items.append(cue)

    for position, cue in enumerate(doc.cues, start=1):
        cue.id = position
    return doc


def compose_srt(doc: SrtDocument) -> str:
    """Serialize a document back to SRT text."""
    parts: List[str] = []
    for item in doc.items:
        if isinstance(item, Cue):
            lines: List[str] = []
            if item.index:
                lines.append(item.index)
            lines.append(item.timestamp_line)
            lines.extend(item.text.split("\n"))
            parts.append("\n".join(lines))
        else:
            parts.append(item.text)
    output = "\n\n".join(parts)
    if output and doc.trailing_newline:
        output += "\n"
    return output


def read_srt_file(path: str) -> SrtDocument:
    return parse_srt(read_text_file(path))


def write_srt_file(path: str, doc: SrtDocument) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(compose_srt(doc))


# ---------------------------------------------------------------------------
# Prompt building / response parsing
# ---------------------------------------------------------------------------


def build_translation_prompts(cues: Sequence[Cue]) -> Tuple[str, str]:
    """Return ``(system, user)`` for the whole-file id-keyed request."""
    payload = [{"n": cue.id, "t": cue.text} for cue in cues]
    return SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False)


def _coerce_id(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("+-").isdigit():
            return int(stripped)
    return None


def _first_key_value(obj: Dict[str, Any], keys: Sequence[str]) -> Any:
    lowered = {str(k).strip().lower(): v for k, v in obj.items()}
    for key in keys:
        if key in lowered:
            return lowered[key]
    return None


def _as_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _pairs_from_object(obj: Dict[str, Any]) -> List[Tuple[int, str]]:
    """Extract ``(id, text)`` pairs from one object, accepting key synonyms."""
    pairs: List[Tuple[int, str]] = []
    raw_id = _first_key_value(obj, _ID_KEYS)
    if raw_id is not None:
        cue_id = _coerce_id(raw_id)
        text = _as_text(_first_key_value(obj, _TEXT_KEYS))
        if cue_id is not None and text is not None:
            pairs.append((cue_id, text))
        return pairs
    # Mapping form: {"1": "...", "2": {...}}.
    for key, value in obj.items():
        cue_id = _coerce_id(key)
        if cue_id is None:
            continue
        if isinstance(value, dict):
            text = _as_text(_first_key_value(value, _TEXT_KEYS))
        else:
            text = _as_text(value)
        if text is not None:
            pairs.append((cue_id, text))
    return pairs


def _store(mapping: Dict[int, str], cue_id: int, text: Optional[str]) -> None:
    if text is None:
        return
    cleaned = text.strip()
    if not cleaned:
        return
    # Duplicated ids: the first non-blank translation wins.
    if mapping.get(cue_id):
        return
    mapping[cue_id] = cleaned


def _collect_pairs(value: Any, mapping: Dict[int, str]) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_pairs(item, mapping)
        return
    if isinstance(value, dict):
        pairs = _pairs_from_object(value)
        if pairs:
            for cue_id, text in pairs:
                _store(mapping, cue_id, text)
            return
        # Wrapper objects such as {"translations": [...]} / {"data": [...]}.
        for nested in value.values():
            if isinstance(nested, (list, dict)):
                _collect_pairs(nested, mapping)


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _try_loads(text: str) -> List[Any]:
    candidate = text.strip()
    if not candidate:
        return []
    for attempt in (candidate, _strip_trailing_commas(candidate)):
        try:
            return [json.loads(attempt)]
        except (ValueError, TypeError):
            continue
    return []


def _fenced_blocks(text: str) -> List[str]:
    blocks = re.findall(r"```[ \t]*(?:json|JSON)?[ \t]*\r?\n?(.*?)```", text, re.DOTALL)
    if not blocks:
        # Unclosed fence: take everything after the opening fence.
        marker = text.find("```")
        if marker != -1:
            tail = text[marker + 3:].lstrip()
            if tail.lower().startswith("json"):
                tail = tail[4:].lstrip()
            if tail:
                blocks.append(tail)
    return blocks


def _brace_matched_spans(text: str) -> List[str]:
    """Extract balanced ``[...]`` (then ``{...}``) spans, string-aware."""
    spans: List[str] = []
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    spans.append(text[start:i + 1])
                    break
    return spans


def _iter_json_lines(text: str) -> Iterable[str]:
    for line in text.splitlines():
        line = line.strip().rstrip(",").strip()
        if not line or line in ("[", "]", "{", "}"):
            continue
        yield line


def parse_translation_response(text: str) -> Dict[int, str]:
    """Robustly parse an LLM response into ``{cue_id: translation}``.

    Handles a bare JSON array, a fenced JSON array, JSON embedded in prose,
    NDJSON (one object per line), mapping objects and key synonyms.
    """
    mapping: Dict[int, str] = {}
    if not text:
        return mapping
    raw = text.strip()
    if not raw:
        return mapping

    candidates: List[str] = [raw]
    candidates.extend(_fenced_blocks(raw))
    candidates.extend(_brace_matched_spans(raw))
    for candidate in candidates:
        for value in _try_loads(candidate):
            _collect_pairs(value, mapping)

    # NDJSON / one-object-per-line, including partial overlap with the above.
    for line in _iter_json_lines(raw):
        for value in _try_loads(line):
            _collect_pairs(value, mapping)

    return mapping


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


@dataclass
class TranslationResult:
    total: int = 0
    translated: int = 0
    unrecovered: int = 0
    unrecovered_ids: List[int] = field(default_factory=list)
    requests: int = 0
    repaired: int = 0
    chunked: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0


def apply_translations(
    doc: SrtDocument, translations: Dict[int, str], zh_only: bool = False
) -> Tuple[int, List[int]]:
    """Write translations back onto the cues, strictly by id.

    Returns ``(applied, missing_ids)``.  Cues without a usable translation keep
    their original text, so nothing is ever lost.
    """
    applied = 0
    missing: List[int] = []
    for cue in doc.cues:
        original = cue.text
        translation = translations.get(cue.id)
        if translation and translation.strip():
            translation = translation.strip()
            cue.translation = translation
            if zh_only or not original.strip():
                cue.text = translation
            else:
                cue.text = f"{translation}\n{original}"
            applied += 1
        elif original.strip():
            missing.append(cue.id)
    return applied, missing


def _iter_chunks(items: Sequence[Any], size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), size):
        yield list(items[start:start + size])


class SubtitleTranslator:
    """Whole-file, id-keyed translation with repair and chunked fallback."""

    def __init__(
        self,
        provider: Any,
        *,
        zh_only: bool = False,
        max_retries: int = 3,
        chunk_size: int = CHUNK_SIZE,
        logger: Any = None,
        interrupt_check: Optional[Callable[[], None]] = None,
        response_parser: Callable[[str], Dict[int, str]] = parse_translation_response,
    ) -> None:
        self.provider = provider
        self.zh_only = zh_only
        self.max_retries = max(1, int(max_retries or 1))
        self.chunk_size = max(1, int(chunk_size or CHUNK_SIZE))
        self.logger = logger
        self.interrupt_check = interrupt_check
        self.response_parser = response_parser
        self.result = TranslationResult()

    # -- logging helpers ----------------------------------------------------
    def _log(self, level: str, message: str) -> None:
        if self.logger is None:
            return
        handler = getattr(self.logger, level, None)
        if handler is None:
            handler = getattr(self.logger, "info", None)
        if handler is not None:
            handler(message)

    def _check_interrupt(self) -> None:
        if self.interrupt_check is not None:
            self.interrupt_check()

    # -- request helpers ----------------------------------------------------
    def _account_usage(self) -> None:
        usage = getattr(self.provider, "last_usage", None)
        if usage is None:
            return
        self.result.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.result.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    def _request_into(self, cues: Sequence[Cue], translations: Dict[int, str]) -> int:
        """Run one request for ``cues`` and merge the result.  Never raises on
        request failure (the repair loop retries), but lets an interrupted
        task propagate."""
        self._check_interrupt()
        system, user = build_translation_prompts(cues)
        self.result.requests += 1
        try:
            text = self.provider.complete(system, user)
        except Exception as exc:  # noqa: BLE001 - retried by the caller
            self._log("warning", f"翻译请求失败（{len(cues)} 条）：{exc}")
            return 0
        self._account_usage()
        parsed = self.response_parser(text or "")
        known = {cue.id for cue in cues}
        added = 0
        for cue_id, value in parsed.items():
            if cue_id not in known:
                continue
            if translations.get(cue_id):
                continue
            if value and value.strip():
                translations[cue_id] = value.strip()
                added += 1
        return added

    def _translate_pending(self, pending: Sequence[Cue], translations: Dict[int, str]) -> List[Cue]:
        """Initial request plus up to ``max_retries`` repair rounds."""
        left = list(pending)
        for attempt in range(self.max_retries + 1):
            if not left:
                break
            if attempt:
                self.result.repaired += 1
                self._log("warning", f"仍有 {len(left)} 条未翻译，进行第 {attempt} 次补译")
            self._request_into(left, translations)
            left = [cue for cue in left if not translations.get(cue.id)]
        return left

    def translate_cues(self, cues: Sequence[Cue]) -> Dict[int, str]:
        translations: Dict[int, str] = {}
        pending = [cue for cue in cues if cue.text.strip()]
        if not pending:
            return translations
        self._log("info", f"提交整文件翻译请求：{len(pending)} 条字幕")
        left = self._translate_pending(pending, translations)
        if left:
            self.result.chunked += 1
            self._log(
                "warning",
                f"整文件翻译不完整，剩余 {len(left)} 条回退为分块翻译（每块 {self.chunk_size} 条）",
            )
            for chunk in _iter_chunks(left, self.chunk_size):
                self._translate_pending(chunk, translations)
        return translations

    def translate_document(self, doc: SrtDocument) -> TranslationResult:
        started = time.time()
        self.result = TranslationResult(total=len(doc.cues))
        try:
            translations = self.translate_cues(doc.cues)
            applied, missing = apply_translations(doc, translations, zh_only=self.zh_only)
            self.result.translated = applied
            self.result.unrecovered = len(missing)
            self.result.unrecovered_ids = missing
        finally:
            self.result.elapsed = time.time() - started
        self._log(
            "info",
            "翻译完成："
            f"总条目 {self.result.total}，已翻译 {self.result.translated}，"
            f"未翻译 {self.result.unrecovered}，请求 {self.result.requests} 次，"
            f"补译 {self.result.repaired} 轮，分块 {self.result.chunked} 次，"
            f"token: prompt {self.result.prompt_tokens} / completion {self.result.completion_tokens}，"
            f"耗时 {self.result.elapsed:.2f} 秒",
        )
        if self.result.unrecovered:
            self._log(
                "warning",
                f"有 {self.result.unrecovered} 条字幕未能翻译，已保留原文："
                f"{self.result.unrecovered_ids[:20]}",
            )
        return self.result

    def translate_file(self, source_path: str, dest_path: str) -> TranslationResult:
        doc = read_srt_file(source_path)
        result = self.translate_document(doc)
        write_srt_file(dest_path, doc)
        return result


def translate_srt_file(source_path: str, dest_path: str, provider: Any, **kwargs: Any) -> TranslationResult:
    """Convenience wrapper around :class:`SubtitleTranslator`."""
    return SubtitleTranslator(provider, **kwargs).translate_file(source_path, dest_path)


__all__ = [
    "CHUNK_SIZE",
    "ENCODINGS",
    "SYSTEM_PROMPT",
    "Cue",
    "RawBlock",
    "SrtDocument",
    "SrtError",
    "SubtitleTranslator",
    "TranslationResult",
    "apply_translations",
    "build_translation_prompts",
    "compose_srt",
    "parse_srt",
    "parse_translation_response",
    "read_srt_file",
    "read_text_file",
    "translate_srt_file",
    "write_srt_file",
]
