"""Tests for translate.chinese (stdlib unittest only)."""

import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # plugins/autosubv2

from translate.chinese import (  # noqa: E402
    CHINESE_LANG_TAGS,
    find_embedded_chinese_subtitle,
    find_external_chinese_subtitle,
    is_chinese_cue,
    is_chinese_lang_tag,
    is_chinese_text,
    normalize_lang_tag,
    sniff_srt_file_chinese,
    texts_look_chinese,
)


CHINESE_SRT = (
    "1\n"
    "00:00:01,000 --> 00:00:03,000\n"
    "這是一個測試字幕\n"
    "\n"
    "2\n"
    "00:00:04,000 --> 00:00:06,000\n"
    "我们去看电影吧\n"
)

JAPANESE_SRT = (
    "1\n"
    "00:00:01,000 --> 00:00:03,000\n"
    "こんにちは、元気ですか\n"
    "\n"
    "2\n"
    "00:00:04,000 --> 00:00:06,000\n"
    "今日はいい天気ですね\n"
)


class TempDirTestCase(unittest.TestCase):
    """Base class providing a temporary directory cleaned up after each test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name)

    def write(self, name, text, encoding="utf-8"):
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=encoding)
        return str(path)

    def make_extractor(self, mapping):
        """Return (extractor, calls) writing the given temp files on demand."""
        calls = []

        def extractor(index):
            calls.append(index)
            payload = mapping.get(index)
            if payload is None:
                return None
            fd, temp_path = tempfile.mkstemp(suffix=".srt")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            return temp_path

        return extractor, calls


class LangTagTests(TempDirTestCase):
    def test_normalize_lang_tag(self):
        self.assertEqual(normalize_lang_tag("  ZH_CN "), "zh-cn")
        self.assertEqual(normalize_lang_tag("Chi"), "chi")
        self.assertEqual(normalize_lang_tag(None), "")
        self.assertEqual(normalize_lang_tag(123), "")
        self.assertEqual(normalize_lang_tag(""), "")

    def test_is_chinese_lang_tag_positive(self):
        positives = [
            "zh",
            "chi",
            "zho",
            "chs",
            "cht",
            "zh-cn",
            "zh_cn",
            "zh-hans",
            "ZH-Hant",
            "zh-tw",
            "zhong",
            "simp",
            "cn",
            "简体",
            "繁体",
            "中文",
            "机翻",
        ]
        for tag in positives:
            with self.subTest(tag=tag):
                self.assertTrue(is_chinese_lang_tag(tag))

    def test_is_chinese_lang_tag_negative(self):
        for tag in ["en", "ja", "und", None, "", "  ", "jpn", "eng", "kr"]:
            with self.subTest(tag=tag):
                self.assertFalse(is_chinese_lang_tag(tag))

    def test_every_declared_tag_is_chinese(self):
        self.assertTrue(CHINESE_LANG_TAGS)
        for tag in CHINESE_LANG_TAGS:
            with self.subTest(tag=tag):
                self.assertTrue(is_chinese_lang_tag(tag))


class CueDetectionTests(TempDirTestCase):
    def test_is_chinese_cue_basics(self):
        self.assertTrue(is_chinese_cue("這是一個測試字幕"))
        self.assertTrue(is_chinese_cue("我们去看电影吧"))
        self.assertFalse(is_chinese_cue("こんにちは、元気ですか"))
        self.assertFalse(is_chinese_cue("안녕하세요"))
        self.assertFalse(is_chinese_cue("Hello world"))
        self.assertFalse(is_chinese_cue(""))

    def test_is_chinese_cue_ignores_markup_and_ratios(self):
        self.assertTrue(is_chinese_cue("<i>中文</i>字幕"))
        self.assertTrue(is_chinese_cue("{\\an8}中文字幕"))
        # Mostly Chinese with a couple of English words still counts.
        self.assertTrue(is_chinese_cue("这是 OK 的一个测试"))
        # Latin-dominated line does not.
        self.assertFalse(is_chinese_cue("this is basically English 的"))
        # Han ratio threshold is honoured.
        self.assertTrue(is_chinese_cue("中A", han_ratio=0.5))
        self.assertFalse(is_chinese_cue("中A", han_ratio=0.6))

    def test_texts_look_chinese(self):
        self.assertTrue(texts_look_chinese(["中文字幕", "", "更多中文"]))
        self.assertFalse(texts_look_chinese(["こんにちは", "元気ですか"]))
        self.assertFalse(texts_look_chinese([]))
        self.assertFalse(texts_look_chinese(["", "   "]))
        # Majority vote across cues.
        self.assertTrue(texts_look_chinese(["中文字幕", "こんにちは"]))
        self.assertFalse(texts_look_chinese(["中文字幕", "こんにちは", "おはよう", "こんばんは"]))
        # max_samples limits the number of inspected lines (evenly sampled,
        # so with one sample the first line decides).
        self.assertFalse(texts_look_chinese(["こんにちは", "中文", "中文", "中文"], max_samples=1))
        self.assertTrue(texts_look_chinese(["中文", "日本語です"], max_samples=1))

    def test_is_chinese_text(self):
        self.assertTrue(is_chinese_text("這是一個測試字幕\n我们去看电影吧"))
        self.assertFalse(is_chinese_text("こんにちは\n元気ですか"))
        self.assertFalse(is_chinese_text(""))
        self.assertFalse(is_chinese_text("\n\n"))

    def test_sniff_srt_file_chinese(self):
        self.assertTrue(sniff_srt_file_chinese(self.write("cn.srt", CHINESE_SRT)))
        self.assertFalse(sniff_srt_file_chinese(self.write("jp.srt", JAPANESE_SRT)))
        # No cues -> raw line fallback.
        self.assertTrue(sniff_srt_file_chinese(self.write("raw.txt", "这不是标准的SRT\n")))
        self.assertFalse(sniff_srt_file_chinese(self.write("raw_jp.txt", "こんにちは\n")))


class EmbeddedStreamTests(TempDirTestCase):
    def test_tagged_chinese_stream_wins_without_extraction(self):
        def extractor(index):  # pragma: no cover - must never run
            raise AssertionError("extractor must not be called for a tagged stream")

        meta = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "zh"}},
            ]
        }
        found, info = find_embedded_chinese_subtitle(meta, extractor)
        self.assertTrue(found)
        self.assertEqual(info["index"], 0)
        self.assertEqual(info["source"], "tag")
        self.assertEqual(info["language"], "zh")

    def test_untagged_japanese_stream_not_chinese(self):
        extractor, calls = self.make_extractor({0: JAPANESE_SRT})
        meta = {"streams": [{"codec_type": "subtitle", "codec_name": "subrip", "tags": {}}]}
        found, info = find_embedded_chinese_subtitle(meta, extractor)
        self.assertFalse(found)
        self.assertIsNone(info)
        self.assertEqual(calls, [0])

    def test_untagged_traditional_chinese_track_detected(self):
        extractor, calls = self.make_extractor({0: CHINESE_SRT})
        meta = {"streams": [{"codec_type": "subtitle", "codec_name": "subrip", "tags": {}}]}
        found, info = find_embedded_chinese_subtitle(meta, extractor)
        self.assertTrue(found)
        self.assertEqual(info["index"], 0)
        self.assertEqual(info["source"], "content")
        self.assertEqual(calls, [0])

    def test_temp_file_is_deleted_after_sniff(self):
        extractor, calls = self.make_extractor({0: JAPANESE_SRT})
        meta = {"streams": [{"codec_type": "subtitle", "codec_name": "subrip", "tags": {}}]}

        created = []
        original = tempfile.mkstemp

        def tracking_mkstemp(*args, **kwargs):
            fd, path = original(*args, **kwargs)
            created.append(path)
            return fd, path

        tempfile.mkstemp = tracking_mkstemp
        try:
            found, _info = find_embedded_chinese_subtitle(meta, extractor)
        finally:
            tempfile.mkstemp = original

        self.assertFalse(found)
        self.assertEqual(calls, [0])
        self.assertEqual(len(created), 1)
        self.assertFalse(os.path.exists(created[0]))
        self.assertTrue(created[0].startswith(tempfile.gettempdir()))

    def test_image_based_stream_is_skipped(self):
        meta = {
            "streams": [
                {
                    "codec_type": "subtitle",
                    "codec_name": "hdmv_pgs_subtitle",
                    "width": 1920,
                    "tags": {"language": "zh"},
                },
                {"codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "zh"}},
            ]
        }
        found, info = find_embedded_chinese_subtitle(meta, lambda index: None)
        self.assertTrue(found)
        self.assertEqual(info["index"], 1)
        self.assertEqual(info["source"], "tag")

    def test_index_counts_all_subtitle_streams(self):
        calls = []

        def extractor(index):
            calls.append(index)
            return None

        meta = {
            "streams": [
                {"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle", "tags": {}},
                {"codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "und"}},
            ]
        }
        found, info = find_embedded_chinese_subtitle(meta, extractor)
        self.assertFalse(found)
        self.assertIsNone(info)
        self.assertEqual(calls, [1])

    def test_foreign_tag_is_not_extracted(self):
        def extractor(index):  # pragma: no cover - must never run
            raise AssertionError("explicit foreign tag must not be extracted")

        meta = {
            "streams": [
                {"codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "jpn"}}
            ]
        }
        found, info = find_embedded_chinese_subtitle(meta, extractor)
        self.assertFalse(found)
        self.assertIsNone(info)

    def test_max_sniff_limits_extractions(self):
        extractor, calls = self.make_extractor({})
        meta = {
            "streams": [
                {"codec_type": "subtitle", "codec_name": "subrip", "tags": {}},
                {"codec_type": "subtitle", "codec_name": "subrip", "tags": {}},
                {"codec_type": "subtitle", "codec_name": "subrip", "tags": {}},
                {"codec_type": "subtitle", "codec_name": "subrip", "tags": {}},
            ]
        }
        found, info = find_embedded_chinese_subtitle(meta, extractor, max_sniff=2)
        self.assertFalse(found)
        self.assertIsNone(info)
        self.assertEqual(calls, [0, 1])

    def test_extractor_errors_are_swallowed(self):
        def extractor(index):
            raise RuntimeError("boom")

        meta = {"streams": [{"codec_type": "subtitle", "codec_name": "subrip", "tags": {}}]}
        found, info = find_embedded_chinese_subtitle(meta, extractor)
        self.assertFalse(found)
        self.assertIsNone(info)

    def test_title_han_overrides_foreign_tag(self):
        extractor, calls = self.make_extractor({0: CHINESE_SRT})
        meta = {
            "streams": [
                {
                    "codec_type": "subtitle",
                    "codec_name": "subrip",
                    "tags": {"language": "en", "title": "中文字幕"},
                }
            ]
        }
        found, info = find_embedded_chinese_subtitle(meta, extractor)
        self.assertTrue(found)
        self.assertEqual(info["index"], 0)
        self.assertEqual(info["source"], "content")
        self.assertEqual(calls, [0])

    def test_missing_and_malformed_meta(self):
        self.assertEqual(find_embedded_chinese_subtitle({}, lambda i: None), (False, None))
        self.assertEqual(
            find_embedded_chinese_subtitle({"streams": None}, lambda i: None), (False, None)
        )


class ExternalSubtitleTests(TempDirTestCase):
    def test_chs_suffix_detected(self):
        video = str(self.dir / "Movie.2024.1080p.mkv")
        self.write("Movie.2024.1080p.chs.srt", "")
        found, filename = find_external_chinese_subtitle(video)
        self.assertTrue(found)
        self.assertEqual(filename, "Movie.2024.1080p.chs.srt")

    def test_chinese_tag_without_content_detected(self):
        video = str(self.dir / "Movie.mkv")
        self.write("Movie.zh-cn.srt", "")
        found, filename = find_external_chinese_subtitle(video)
        self.assertTrue(found)
        self.assertEqual(filename, "Movie.zh-cn.srt")

    def test_untagged_chinese_srt_detected_by_content(self):
        video = str(self.dir / "Movie.mkv")
        self.write("Movie.srt", CHINESE_SRT)
        found, filename = find_external_chinese_subtitle(video)
        self.assertTrue(found)
        self.assertEqual(filename, "Movie.srt")

    def test_japanese_srt_not_detected(self):
        video = str(self.dir / "Movie.mkv")
        self.write("Movie.ja.srt", JAPANESE_SRT)
        found, filename = find_external_chinese_subtitle(video)
        self.assertFalse(found)
        self.assertIsNone(filename)

    def test_no_matching_subtitle(self):
        video = str(self.dir / "Movie.mkv")
        self.write("Other.srt", CHINESE_SRT)
        found, filename = find_external_chinese_subtitle(video)
        self.assertFalse(found)
        self.assertIsNone(filename)

    def test_missing_directory_never_raises(self):
        found, filename = find_external_chinese_subtitle(str(self.dir / "nope" / "Movie.mkv"))
        self.assertFalse(found)
        self.assertIsNone(filename)

    def test_ass_with_chinese_dialogue(self):
        video = str(self.dir / "Movie.mkv")
        self.write(
            "Movie.ass",
            "[Script Info]\nTitle: demo\n\n[Events]\n"
            "Format: Layer, Start, End, Style, Text\n"
            "Dialogue: 0,0:00:01.00,0:00:03.00,Default,中文字幕\n",
        )
        found, filename = find_external_chinese_subtitle(video)
        self.assertTrue(found)
        self.assertEqual(filename, "Movie.ass")

    def test_untagged_japanese_ass_not_detected(self):
        video = str(self.dir / "Movie.mkv")
        self.write(
            "Movie.ass",
            "[Script Info]\nTitle: demo\n\n[Events]\n"
            "Format: Layer, Start, End, Style, Text\n"
            "Dialogue: 0,0:00:01.00,0:00:03.00,Default,こんにちは\n",
        )
        found, filename = find_external_chinese_subtitle(video)
        self.assertFalse(found)
        self.assertIsNone(filename)


if __name__ == "__main__":
    unittest.main()
