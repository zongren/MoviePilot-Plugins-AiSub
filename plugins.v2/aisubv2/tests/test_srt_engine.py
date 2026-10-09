"""Offline unit tests for the whole-file SRT translation engine.

Run with::

    python3 -m unittest discover -s plugins.v2/aisubv2/tests -p 'test_srt_engine.py' -v
"""

import json
import os
import sys
import pathlib
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # plugins.v2/aisubv2

from translate.srt_engine import (  # noqa: E402
    CHUNK_SIZE,
    Cue,
    SubtitleTranslator,
    apply_translations,
    build_translation_prompts,
    compose_srt,
    parse_srt,
    parse_translation_response,
    read_srt_file,
    write_srt_file,
)

SAMPLE = """\
1
00:00:01,000 --> 00:00:02,000
Hello world

2
00:00:03,000 --> 00:00:04,000
<i>Second</i>
line two

3
00:00:05,000 --> 00:00:06,000
[uomo]
"""


class FakeProvider:
    """Provider stub whose ``complete`` is driven by a callback over cues."""

    def __init__(self, responder):
        self.responder = responder
        self.calls = []
        self.last_usage = None

    def complete(self, system, user):
        cues = json.loads(user)
        self.calls.append(cues)
        return self.responder(cues)


def full_json(cues, prefix="译"):
    return json.dumps([{"n": c["n"], "t": prefix + c["t"]} for c in cues], ensure_ascii=False)


def timestamp_lines(text):
    return [line for line in text.split("\n") if "-->" in line]


def cue_signature(doc):
    return [(cue.index, cue.timestamp_line) for cue in doc.cues]


class ParseComposeTests(unittest.TestCase):
    def test_round_trip_is_byte_identical(self):
        doc = parse_srt(SAMPLE)
        self.assertEqual(len(doc.cues), 3)
        self.assertEqual(compose_srt(doc), SAMPLE)

    def test_indexes_and_timestamps_untouched(self):
        doc = parse_srt(SAMPLE)
        self.assertEqual([c.index for c in doc.cues], ["1", "2", "3"])
        self.assertEqual(doc.cues[0].start, "00:00:01,000")
        self.assertEqual(doc.cues[0].end, "00:00:02,000")
        self.assertEqual(doc.cues[0].timestamp_line, "00:00:01,000 --> 00:00:02,000")
        self.assertEqual(doc.cues[1].text, "<i>Second</i>\nline two")

    def test_sequential_ids_even_when_source_indexes_are_odd(self):
        text = "7\n00:00:01,000 --> 00:00:02,000\nA\n\n7\n00:00:03,000 --> 00:00:04,000\nB\n"
        doc = parse_srt(text)
        self.assertEqual([c.id for c in doc.cues], [1, 2])
        self.assertEqual([c.index for c in doc.cues], ["7", "7"])

    def test_settings_suffix_is_preserved(self):
        text = "1\n00:00:01,000 --> 00:00:02,000 X1:100 X2:200\nA\n"
        doc = parse_srt(text)
        self.assertEqual(doc.cues[0].settings, "X1:100 X2:200")
        self.assertEqual(compose_srt(doc), text)

    def test_non_cue_blocks_are_kept_verbatim(self):
        text = "Random header\n\n1\n00:00:01,000 --> 00:00:02,000\nA\n"
        doc = parse_srt(text)
        self.assertEqual(compose_srt(doc), text)

    def test_crlf_is_normalized_without_losing_timestamps(self):
        text = "1\r\n00:00:01,000 --> 00:00:02,000\r\nA\r\n"
        doc = parse_srt(text)
        self.assertEqual(doc.cues[0].timestamp_line, "00:00:01,000 --> 00:00:02,000")
        self.assertEqual(compose_srt(doc), "1\n00:00:01,000 --> 00:00:02,000\nA\n")

    def test_index_less_cue(self):
        text = "00:00:01,000 --> 00:00:02,000\nA\n"
        doc = parse_srt(text)
        self.assertEqual(len(doc.cues), 1)
        self.assertEqual(doc.cues[0].index, "")
        self.assertEqual(compose_srt(doc), text)


class EncodingTests(unittest.TestCase):
    def test_gbk_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.srt")
            content = "1\n00:00:01,000 --> 00:00:02,000\n你好世界\n"
            with open(path, "wb") as fh:
                fh.write(content.encode("gbk"))
            doc = read_srt_file(path)
            self.assertEqual(doc.cues[0].text, "你好世界")

    def test_write_and_read_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.srt")
            dst = os.path.join(tmp, "out.srt")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write(SAMPLE)
            doc = read_srt_file(src)
            write_srt_file(dst, doc)
            with open(dst, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), SAMPLE)


class ResponseParsingTests(unittest.TestCase):
    def test_bare_json_array(self):
        mapping = parse_translation_response('[{"n":1,"t":"一"},{"n":2,"t":"二"}]')
        self.assertEqual(mapping, {1: "一", 2: "二"})

    def test_fenced_json(self):
        text = 'Here you go:\n```json\n[{"n":1,"t":"一"},{"n":2,"t":"二"}]\n```\n'
        self.assertEqual(parse_translation_response(text), {1: "一", 2: "二"})

    def test_json_embedded_in_prose(self):
        text = 'Sure! [{"n": 4, "t": "四"}, {"n": 5, "t": "五"}] hope that helps'
        self.assertEqual(parse_translation_response(text), {4: "四", 5: "五"})

    def test_ndjson(self):
        text = '{"n":1,"t":"一"}\n{"n":2,"t":"二"}\n{"n":3,"t":"三"}\n'
        self.assertEqual(parse_translation_response(text), {1: "一", 2: "二", 3: "三"})

    def test_key_synonyms(self):
        self.assertEqual(parse_translation_response('[{"id":1,"text":"一"}]'), {1: "一"})
        self.assertEqual(parse_translation_response('[{"index":2,"translation":"二"}]'), {2: "二"})
        self.assertEqual(parse_translation_response('[{"n":"3","t":"三"}]'), {3: "三"})

    def test_mapping_object_form(self):
        self.assertEqual(parse_translation_response('{"1":"一","2":"二"}'), {1: "一", 2: "二"})

    def test_wrapper_object_form(self):
        text = '{"translations":[{"n":1,"t":"一"}]}'
        self.assertEqual(parse_translation_response(text), {1: "一"})

    def test_duplicate_ids_first_non_blank_wins(self):
        mapping = parse_translation_response('[{"n":1,"t":"一"},{"n":1,"t":"壹"}]')
        self.assertEqual(mapping, {1: "一"})
        mapping = parse_translation_response('[{"n":1,"t":""},{"n":1,"t":"壹"}]')
        self.assertEqual(mapping, {1: "壹"})

    def test_blank_translations_are_treated_as_missing(self):
        self.assertEqual(parse_translation_response('[{"n":1,"t":"   "}]'), {})

    def test_empty_and_garbage_input(self):
        self.assertEqual(parse_translation_response(""), {})
        self.assertEqual(parse_translation_response("no json here"), {})

    def test_truncated_array_recovers_complete_objects_only(self):
        mapping = parse_translation_response('[{"n":1,"t":"一"},{"n":2,"t":')
        self.assertEqual(mapping, {1: "一"})


class PromptTests(unittest.TestCase):
    def test_prompt_payload_is_id_keyed_json(self):
        doc = parse_srt(SAMPLE)
        system, user = build_translation_prompts(doc.cues)
        self.assertIn("只输出 JSON 数组", system)
        payload = json.loads(user)
        self.assertEqual(payload[0], {"n": 1, "t": "Hello world"})
        self.assertEqual(len(payload), 3)


class EngineTests(unittest.TestCase):
    def _doc(self, text=SAMPLE):
        return parse_srt(text)

    def test_whole_file_translation_preserves_timestamps(self):
        provider = FakeProvider(lambda cues: full_json(cues))
        doc = self._doc()
        result = SubtitleTranslator(provider).translate_document(doc)
        self.assertEqual(result.total, 3)
        self.assertEqual(result.translated, 3)
        self.assertEqual(result.unrecovered, 0)
        self.assertEqual(result.requests, 1)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(provider.calls[0]), 3)
        output = compose_srt(doc)
        self.assertEqual(timestamp_lines(output), timestamp_lines(SAMPLE))
        self.assertEqual(cue_signature(doc), cue_signature(parse_srt(SAMPLE)))
        self.assertIn("译Hello world\nHello world", output)

    def test_zh_only_output(self):
        provider = FakeProvider(lambda cues: full_json(cues))
        doc = self._doc()
        SubtitleTranslator(provider, zh_only=True).translate_document(doc)
        self.assertEqual(doc.cues[0].text, "译Hello world")
        self.assertNotIn("Hello world\nHello world", compose_srt(doc))

    def test_missing_ids_are_repaired(self):
        state = {"first": True}

        def responder(cues):
            ids = [c["n"] for c in cues]
            if state["first"] and len(cues) > 1:
                state["first"] = False
                ids = ids[:-1]  # drop the last cue
            return json.dumps([{"n": n, "t": "译"} for n in ids], ensure_ascii=False)

        provider = FakeProvider(responder)
        doc = self._doc()
        result = SubtitleTranslator(provider, max_retries=3).translate_document(doc)
        self.assertEqual(result.unrecovered, 0)
        self.assertEqual(result.repaired, 1)
        self.assertEqual(result.requests, 2)

    def test_unrecoverable_cues_keep_original_text(self):
        provider = FakeProvider(lambda cues: "[]")
        doc = self._doc()
        result = SubtitleTranslator(provider, max_retries=1, chunk_size=2).translate_document(doc)
        self.assertEqual(result.unrecovered, 3)
        self.assertEqual(compose_srt(doc), SAMPLE)
        self.assertTrue(result.unrecovered_ids == [1, 2, 3])

    def test_chunked_fallback_on_truncation(self):
        def responder(cues):
            if len(cues) > 1:
                return '[{"n":%d,"t":"译"}]' % cues[0]["n"]  # simulating truncation
            return full_json(cues)

        provider = FakeProvider(responder)
        doc = self._doc()
        result = SubtitleTranslator(provider, max_retries=1, chunk_size=1).translate_document(doc)
        self.assertEqual(result.unrecovered, 0)
        self.assertEqual(result.chunked, 1)
        self.assertEqual(result.translated, 3)
        self.assertEqual(timestamp_lines(compose_srt(doc)), timestamp_lines(SAMPLE))

    def test_request_failure_does_not_raise(self):
        def boom(cues):
            raise RuntimeError("network down")

        doc = self._doc()
        result = SubtitleTranslator(FakeProvider(boom), max_retries=1, chunk_size=2).translate_document(doc)
        self.assertEqual(result.unrecovered, 3)
        self.assertEqual(compose_srt(doc), SAMPLE)

    def test_interrupt_propagates(self):
        class Interrupt(Exception):
            pass

        def check():
            raise Interrupt()

        doc = self._doc()
        with self.assertRaises(Interrupt):
            SubtitleTranslator(FakeProvider(lambda c: full_json(c)), interrupt_check=check).translate_document(doc)

    def test_default_chunk_size_is_300(self):
        self.assertEqual(CHUNK_SIZE, 300)

    def test_blank_cues_are_not_requested(self):
        text = "1\n00:00:01,000 --> 00:00:02,000\n\n\n2\n00:00:03,000 --> 00:00:04,000\nHi\n"
        doc = parse_srt(text)
        provider = FakeProvider(lambda cues: full_json(cues))
        result = SubtitleTranslator(provider).translate_document(doc)
        self.assertEqual(len(provider.calls[0]), 1)
        self.assertEqual(result.translated, 1)
        self.assertEqual(result.unrecovered, 0)


class ApplyTests(unittest.TestCase):
    def test_apply_strictly_by_id(self):
        doc = parse_srt(SAMPLE)
        apply_translations(doc, {2: "第二"}, zh_only=False)
        self.assertEqual(doc.cues[0].text, "Hello world")  # untouched
        self.assertEqual(doc.cues[1].text, "第二\n<i>Second</i>\nline two")
        self.assertEqual(doc.cues[2].text, "[uomo]")

    def test_apply_ignores_unknown_ids(self):
        doc = parse_srt(SAMPLE)
        applied, missing = apply_translations(doc, {99: "x"}, zh_only=False)
        self.assertEqual(applied, 0)
        self.assertEqual(missing, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
