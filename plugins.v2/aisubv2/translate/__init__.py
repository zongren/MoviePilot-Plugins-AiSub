"""Subtitle translation package for the AiSubV2 plugin.

Modules
-------
* :mod:`srt_engine` -- SRT parsing plus the whole-file, id-keyed translation
  engine (repair + chunked fallback).  Standard library only.
* :mod:`openai_translate` -- provider layer for OpenAI Chat Completions,
  OpenAI Responses and Anthropic Messages wire formats.
* :mod:`chinese` -- Chinese language tag / content detection used by the
  "skip files that already have Chinese subtitles" option.

The package intentionally has no imports at package level so that each module
can be loaded standalone (e.g. by ``tools/translate_srt.py``) without pulling
in MoviePilot.
"""

__all__ = ["srt_engine", "openai_translate", "chinese"]
