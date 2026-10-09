"""Chinese subtitle detection helpers.

Two independent signals are used throughout the plugin:

* **Language tags** – external subtitle file names (``Movie.chs.srt``) and
  embedded stream metadata (``tags.language`` / ``tags.title``) often carry a
  language tag (``zh``, ``zh-CN``, ``chi``, ``简`` ...).  These are cheap and
  authoritative when present.
* **Content sniffing** – a lot of subtitle tracks are untagged, so the actual
  text is inspected: a track counts as Chinese when it is dominated by Han
  ideographs and contains no Hiragana / Katakana / Hangul.

Everything here is standard library only and never raises out of the public
detection entry points: a broken candidate file or a failing extractor callback
must not abort a whole subtitle run.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .srt_engine import parse_srt, read_text_file

__all__ = [
    "CHINESE_LANG_TAGS",
    "normalize_lang_tag",
    "is_chinese_lang_tag",
    "is_chinese_cue",
    "is_chinese_text",
    "texts_look_chinese",
    "sniff_srt_file_chinese",
    "find_external_chinese_subtitle",
    "find_embedded_chinese_subtitle",
]


# ---------------------------------------------------------------------------
# Language tags
# ---------------------------------------------------------------------------

#: Normalized (lower case) language tags that identify Chinese subtitles.
CHINESE_LANG_TAGS: frozenset = frozenset(
    {
        "zh",
        "chi",
        "zho",
        "chs",
        "cht",
        "zh-cn",
        "zh-hans",
        "zh-hant",
        "zhong",
        "simp",
        "cn",
        "简体",
        "繁体",
        "中文",
        "机翻",
    }
)

#: Tokens that are Chinese even though they do not look like ISO codes.
_CHINESE_EXTRA_TOKENS: frozenset = frozenset({"简体", "繁体", "中文", "机翻"})

#: Tracks marked with one of these are treated as "no usable language tag".
_UNDETERMINED_LANG_TAGS: frozenset = frozenset({"", "und", "unknown", "undetermined"})

#: Subtitle containers we are willing to sniff as plain text/markup.
_SUBTITLE_EXTENSIONS: frozenset = frozenset({".srt", ".ass", ".ssa", ".sub", ".vtt"})

#: Subtitle codecs that carry bitmap images instead of text.
_IMAGE_SUBTITLE_CODECS: frozenset = frozenset(
    {"dvd_subtitle", "dvb_subtitle", "hdmv_pgs_subtitle"}
)


def normalize_lang_tag(tag: Any) -> str:
    """Normalize a language tag: strip, lower case, ``_`` -> ``-``.

    Non-string values (including ``None``) and empty strings become ``""``.
    """
    if not isinstance(tag, str):
        return ""
    return tag.strip().lower().replace("_", "-")


def is_chinese_lang_tag(tag: Any) -> bool:
    """Return ``True`` when *tag* denotes Chinese.

    Accepts ISO 639-1/2 codes (``zh``, ``chi``, ``zho``), common release tags
    (``chs``, ``cht``, ``cn``, ``simp``), any ``zh``-prefixed tag (``zh-tw``)
    and the Chinese script/word tokens.
    """
    normalized = normalize_lang_tag(tag)
    if not normalized:
        return False
    if normalized in CHINESE_LANG_TAGS:
        return True
    if normalized.startswith("zh"):
        return True
    # The set already holds these, but keep the check explicit and independent
    # of the constant's exact contents.
    return normalized in _CHINESE_EXTRA_TOKENS


# ---------------------------------------------------------------------------
# Content sniffing
# ---------------------------------------------------------------------------

_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_KANA_HANGUL_RE = re.compile(r"[\u3040-\u30ff\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_MARKUP_ANGLE_RE = re.compile(r"<[^>]*>")
_MARKUP_BRACE_RE = re.compile(r"\{[^}]*\}")


def _strip_markup(text: str) -> str:
    """Drop ``<...>`` (font/italic tags) and ``{...}`` (ASS override blocks)."""
    if not text:
        return ""
    return _MARKUP_BRACE_RE.sub("", _MARKUP_ANGLE_RE.sub("", text))


def is_chinese_cue(text: str, han_ratio: float = 0.5) -> bool:
    """Return ``True`` when one subtitle line looks Chinese.

    A line is Chinese when it contains at least one Han ideograph, contains no
    Kana/Hangul character at all, and its Han share of ``Han + Latin letters``
    reaches *han_ratio*.  Digits, punctuation and whitespace are ignored, so a
    mostly Chinese line with a couple of English words still counts.
    """
    if not text:
        return False
    stripped = _strip_markup(text)
    if not stripped:
        return False
    if _KANA_HANGUL_RE.search(stripped):
        return False
    han_count = len(_HAN_RE.findall(stripped))
    if han_count == 0:
        return False
    latin_count = len(_LATIN_RE.findall(stripped))
    total = han_count + latin_count
    if total <= 0:
        return False
    try:
        return (han_count / total) >= float(han_ratio)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return (han_count / total) >= 0.5


def _sample_evenly(items: Sequence[str], max_samples: int) -> List[str]:
    """Take up to *max_samples* items spread evenly across *items*."""
    total = len(items)
    if max_samples <= 0 or total <= 0:
        return []
    if total <= max_samples:
        return list(items)
    picked: List[str] = []
    for position in range(max_samples):
        index = (position * total) // max_samples
        candidate = items[index]
        if not picked or picked[-1] != candidate:
            picked.append(candidate)
    return picked


def texts_look_chinese(
    texts: Iterable[str],
    *,
    cue_ratio: float = 0.5,
    han_ratio: float = 0.5,
    max_samples: int = 200,
) -> bool:
    """Return ``True`` when at least *cue_ratio* of the sampled lines are Chinese.

    Empty strings are ignored, so both a cue list and one cue's multi-line text
    can be passed in.  Returns ``False`` when nothing non-empty is available.
    """
    try:
        materialized = list(texts)
    except TypeError:
        return False
    non_empty = [item for item in materialized if isinstance(item, str) and item.strip()]
    sampled = _sample_evenly(non_empty, max_samples)
    if not sampled:
        return False
    chinese = sum(1 for item in sampled if is_chinese_cue(item, han_ratio))
    try:
        return (chinese / len(sampled)) >= float(cue_ratio)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return (chinese / len(sampled)) >= 0.5


def is_chinese_text(text: str, cue_ratio: float = 0.5, han_ratio: float = 0.5) -> bool:
    """Return ``True`` when the raw text (one line per cue) looks Chinese."""
    if not text:
        return False
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return False
    return texts_look_chinese(lines, cue_ratio=cue_ratio, han_ratio=han_ratio)


def sniff_srt_file_chinese(path: str) -> bool:
    """Sniff whether the SRT file at *path* looks Chinese.

    Falls back to raw lines when the file yields no parseable cues, so even a
    malformed / non-SRT text file can be classified.
    """
    text = read_text_file(path)
    try:
        doc = parse_srt(text)
    except Exception:  # pragma: no cover - defensive, parse_srt is tolerant
        doc = None
    cue_texts = [cue.text for cue in getattr(doc, "cues", []) or []]
    if cue_texts:
        return texts_look_chinese(cue_texts)
    return is_chinese_text(text)


# ---------------------------------------------------------------------------
# External sidecar subtitles
# ---------------------------------------------------------------------------


def _split_video_path(video_file: str) -> Tuple[str, str, str]:
    directory, filename = os.path.split(video_file)
    stem, extension = os.path.splitext(filename)
    return directory, stem, extension


def _language_tokens(filename: str, stem: str) -> List[str]:
    """Return the normalized tags encoded in a sidecar filename.

    ``Movie.chs.zh-cn.srt`` -> ``["chs", "zh-cn"]``; ``Movie.srt`` -> ``[]``.
    """
    name_stem, _ = os.path.splitext(filename)
    if stem and name_stem.startswith(stem + "."):
        segment = name_stem[len(stem) + 1:]
    elif name_stem == stem:
        segment = ""
    else:
        segment = name_stem
    if not segment:
        return []
    return [normalize_lang_tag(part) for part in segment.split(".") if part]


def _sniff_plain_subtitle(path: str) -> bool:
    """Sniff a non-SRT subtitle (ASS/SSA/VTT/SUB) by stripping its markup.

    Header/style/timestamp lines are dropped; for ASS ``Dialogue:`` lines only
    the spoken text (after the last field separator) is kept, otherwise every
    dialogue line would be discarded and Chinese tracks would look empty.
    """
    text = read_text_file(path)
    lines: List[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("Dialogue:", "Format:", "Style:", "WEBVTT", "[")):
            if line.startswith("Dialogue:") and "," in line:
                spoken = line.rsplit(",", 1)[-1].strip()
                if spoken:
                    lines.append(spoken)
            continue
        if "-->" in line:
            continue
        lines.append(line)
    if not lines:
        return False
    return is_chinese_text("\n".join(lines))


def find_external_chinese_subtitle(video_file: str) -> Tuple[bool, Optional[str]]:
    """Find a Chinese sidecar subtitle for *video_file*.

    Returns ``(True, filename)`` with the bare file name (no directory) on the
    first hit, otherwise ``(False, None)``.  The video file itself does not need
    to exist; only the sidecar is inspected.
    """
    try:
        directory, stem, _extension = _split_video_path(video_file)
    except Exception:  # pragma: no cover - defensive
        return (False, None)

    try:
        entries = sorted(os.listdir(directory or "."))
    except OSError:
        return (False, None)

    prefix = (stem + ".") if stem else ""
    for filename in entries:
        try:
            if not filename.lower().endswith(tuple(_SUBTITLE_EXTENSIONS)):
                continue
            if prefix and not filename.startswith(prefix):
                continue
            path = os.path.join(directory, filename)

            for token in _language_tokens(filename, stem):
                if is_chinese_lang_tag(token):
                    return (True, filename)

            _name_stem, extension = os.path.splitext(filename)
            if extension.lower() == ".srt":
                if sniff_srt_file_chinese(path):
                    return (True, filename)
            elif _sniff_plain_subtitle(path):
                return (True, filename)
        except Exception:
            # A single unreadable candidate must never abort the search.
            continue
    return (False, None)


# ---------------------------------------------------------------------------
# Embedded subtitle streams
# ---------------------------------------------------------------------------


def _is_image_subtitle_stream(stream: dict) -> bool:
    if not isinstance(stream, dict):
        return True
    codec_name = stream.get("codec_name")
    if isinstance(codec_name, str) and codec_name.lower() in _IMAGE_SUBTITLE_CODECS:
        return True
    return "width" in stream


def _stream_tag(stream: dict, key: str) -> Optional[str]:
    tags = stream.get("tags")
    if not isinstance(tags, dict):
        return None
    value = tags.get(key)
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _title_has_han(title: Optional[str]) -> bool:
    if not title:
        return False
    return bool(_HAN_RE.search(title))


def find_embedded_chinese_subtitle(
    video_meta: dict,
    extractor: Any,
    *,
    max_sniff: int = 3,
) -> Tuple[bool, Optional[dict]]:
    """Find a Chinese embedded subtitle stream.

    *extractor* is called as ``extractor(index)`` with the ffmpeg stream index
    (``ffmpeg -map 0:s:N`` numbering over **all** subtitle streams) and must
    return a temporary extracted file path or ``None``.  Any temp file it
    returns is always deleted.

    Tagged streams win immediately (``source: "tag"``); untagged/undetermined
    streams are extracted and sniffed at most *max_sniff* times
    (``source: "content"``).  Never raises.
    """
    if not isinstance(video_meta, dict):
        return (False, None)

    streams = video_meta.get("streams") or []
    if not isinstance(streams, (list, tuple)):
        return (False, None)

    subtitle_index = 0
    sniffed = 0

    for stream in streams:
        if not isinstance(stream, dict):
            continue
        if stream.get("codec_type") != "subtitle":
            continue

        index = subtitle_index
        subtitle_index += 1

        if _is_image_subtitle_stream(stream):
            continue

        language = _stream_tag(stream, "language")
        title = _stream_tag(stream, "title")

        try:
            if is_chinese_lang_tag(language) or is_chinese_lang_tag(title):
                return (True, {"index": index, "language": language, "source": "tag"})

            normalized_lang = normalize_lang_tag(language)
            title_is_chinese = _title_has_han(title)
            if (
                normalized_lang
                and normalized_lang not in _UNDETERMINED_LANG_TAGS
                and not title_is_chinese
            ):
                # Explicitly foreign and not contradicting itself: trust the tag.
                continue

            if sniffed >= max_sniff:
                continue
            sniffed += 1

            temp_path = None
            try:
                temp_path = extractor(index)
                if temp_path and sniff_srt_file_chinese(temp_path):
                    return (
                        True,
                        {"index": index, "language": language, "source": "content"},
                    )
            except Exception:
                continue
            finally:
                if temp_path:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                    except Exception:  # pragma: no cover - defensive
                        pass
        except Exception:  # pragma: no cover - defensive
            continue

    return (False, None)
