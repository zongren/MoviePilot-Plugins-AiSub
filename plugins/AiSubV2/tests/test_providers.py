import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # plugins/AiSubV2
from translate.openai_translate import (
    AnthropicMessagesProvider,
    BaseLLMProvider,
    LLMError,
    LLMUsage,
    OpenAIChatProvider,
    OpenAIResponsesProvider,
    create_provider,
)

import unittest
from unittest.mock import MagicMock, patch

import requests


CHAT_PAYLOAD = {
    "id": "chatcmpl-1",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "reasoning_content": "内部的思考过程，不应出现在译文里",
                "content": "你好，世界",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
}

RESPONSES_PAYLOAD = {
    "id": "resp-1",
    "output": [
        {
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "内部推理，应被跳过"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "第一段"},
                {"type": "output_text", "text": "第二段"},
            ],
        },
    ],
    "output_text": "回退文本不应被使用",
    "usage": {"input_tokens": 13, "output_tokens": 5, "total_tokens": 18},
}

ANTHROPIC_PAYLOAD = {
    "id": "msg-1",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "thinking", "thinking": "内部思考，应被跳过", "signature": "sig"},
        {"type": "text", "text": "你好"},
        {"type": "text", "text": "，世界"},
    ],
    "usage": {"input_tokens": 8, "output_tokens": 4},
}


def make_response(payload, status_code=200, text=""):
    return MagicMock(status_code=status_code, text=text, json=lambda: payload)


class ProviderTestCase(unittest.TestCase):
    def assert_call(self, mock_post, url, headers, body, timeout=900, proxies=None):
        self.assertEqual(mock_post.call_count, 1, "每次 complete 只允许一次 HTTP 请求")
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], url)
        self.assertEqual(kwargs["headers"], headers)
        self.assertEqual(kwargs["json"], body)
        self.assertEqual(kwargs["timeout"], timeout)
        self.assertEqual(kwargs["proxies"], proxies)

    # -- openai_chat -------------------------------------------------------
    def test_chat_urls(self):
        payload = dict(CHAT_PAYLOAD)
        cases = [
            ("https://api.deepseek.com", "https://api.deepseek.com/v1/chat/completions"),
            (
                "https://api.deepseek.com/anthropic",
                "https://api.deepseek.com/anthropic/v1/chat/completions",
            ),
            ("https://api.deepseek.com/v1", "https://api.deepseek.com/v1/chat/completions"),
            ("https://api.deepseek.com/v1/", "https://api.deepseek.com/v1/chat/completions"),
        ]
        for base, expected in cases:
            with self.subTest(base=base):
                provider = OpenAIChatProvider(base, "sk-test", "deepseek-chat")
                with patch("requests.post") as post:
                    post.return_value = make_response(payload)
                    provider.complete("SYS", "USER")
                    self.assert_call(
                        post,
                        expected,
                        {
                            "Authorization": "Bearer sk-test",
                            "Content-Type": "application/json",
                        },
                        {
                            "model": "deepseek-chat",
                            "messages": [
                                {"role": "system", "content": "SYS"},
                                {"role": "user", "content": "USER"},
                            ],
                            "temperature": 0.2,
                            "effort": "high",
                            "max_tokens": 64000,
                        },
                    )

    def test_chat_body_with_options_and_proxy(self):
        provider = OpenAIChatProvider(
            "https://api.deepseek.com",
            "sk-abc",
            "deepseek-reasoner",
            temperature=0.7,
            reasoning_effort="medium",
            max_tokens=1234,
            timeout=30,
            proxy={"https": "http://127.0.0.1:7890"},
        )
        with patch("requests.post") as post:
            post.return_value = make_response(dict(CHAT_PAYLOAD))
            provider.complete("系统", "用户")
            self.assert_call(
                post,
                "https://api.deepseek.com/v1/chat/completions",
                {"Authorization": "Bearer sk-abc", "Content-Type": "application/json"},
                {
                    "model": "deepseek-reasoner",
                    "messages": [
                        {"role": "system", "content": "系统"},
                        {"role": "user", "content": "用户"},
                    ],
                    "temperature": 0.7,
                    "effort": "medium",
                    "max_tokens": 1234,
                },
                timeout=30,
                proxies={"https": "http://127.0.0.1:7890"},
            )

    def test_chat_extracts_content_and_skips_reasoning(self):
        provider = OpenAIChatProvider("https://api.deepseek.com", "sk-test", "m")
        with patch("requests.post") as post:
            post.return_value = make_response(dict(CHAT_PAYLOAD))
            text = provider.complete("SYS", "USER")
        self.assertEqual(text, "你好，世界")
        self.assertNotIn("思考", text)

    def test_chat_empty_content_is_empty_string(self):
        provider = OpenAIChatProvider("https://api.deepseek.com", "sk-test", "m")
        payload = {"choices": [{"message": {"role": "assistant", "content": ""}}]}
        with patch("requests.post") as post:
            post.return_value = make_response(payload)
            self.assertEqual(provider.complete("SYS", "USER"), "")

    def test_chat_usage(self):
        provider = OpenAIChatProvider("https://api.deepseek.com", "sk-test", "m")
        with patch("requests.post") as post:
            post.return_value = make_response(dict(CHAT_PAYLOAD))
            provider.complete("SYS", "USER")
        self.assertEqual(
            provider.last_usage, LLMUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18)
        )
        self.assertIsInstance(provider.last_usage, LLMUsage)
        self.assertEqual(provider.last_response["id"], "chatcmpl-1")

    # -- openai_responses --------------------------------------------------
    def test_responses_urls(self):
        payload = dict(RESPONSES_PAYLOAD)
        cases = [
            ("https://api.deepseek.com", "https://api.deepseek.com/v1/responses"),
            (
                "https://api.deepseek.com/anthropic",
                "https://api.deepseek.com/anthropic/v1/responses",
            ),
            ("https://api.deepseek.com/v1", "https://api.deepseek.com/v1/responses"),
        ]
        for base, expected in cases:
            with self.subTest(base=base):
                provider = OpenAIResponsesProvider(base, "sk-test", "gpt-5")
                with patch("requests.post") as post:
                    post.return_value = make_response(payload)
                    provider.complete("SYS", "USER")
                    self.assert_call(
                        post,
                        expected,
                        {
                            "Authorization": "Bearer sk-test",
                            "Content-Type": "application/json",
                        },
                        {
                            "model": "gpt-5",
                            "instructions": "SYS",
                            "input": "USER",
                            "reasoning": {"effort": "high"},
                            "max_output_tokens": 64000,
                        },
                    )

    def test_responses_extracts_output_text_and_skips_reasoning(self):
        provider = OpenAIResponsesProvider("https://api.deepseek.com", "sk-test", "gpt-5")
        with patch("requests.post") as post:
            post.return_value = make_response(dict(RESPONSES_PAYLOAD))
            text = provider.complete("SYS", "USER")
        self.assertEqual(text, "第一段第二段")
        self.assertNotIn("内部推理", text)
        self.assertNotIn("回退文本", text)

    def test_responses_falls_back_to_output_text(self):
        provider = OpenAIResponsesProvider("https://api.deepseek.com", "sk-test", "gpt-5")
        payload = {"output": [], "output_text": "回退正文"}
        with patch("requests.post") as post:
            post.return_value = make_response(payload)
            self.assertEqual(provider.complete("SYS", "USER"), "回退正文")

    def test_responses_usage(self):
        provider = OpenAIResponsesProvider("https://api.deepseek.com", "sk-test", "gpt-5")
        with patch("requests.post") as post:
            post.return_value = make_response(dict(RESPONSES_PAYLOAD))
            provider.complete("SYS", "USER")
        self.assertEqual(
            provider.last_usage, LLMUsage(prompt_tokens=13, completion_tokens=5, total_tokens=18)
        )
        self.assertEqual(provider.last_response["id"], "resp-1")

    # -- anthropic_messages ------------------------------------------------
    def test_anthropic_urls(self):
        payload = dict(ANTHROPIC_PAYLOAD)
        cases = [
            ("https://api.deepseek.com", "https://api.deepseek.com/v1/messages"),
            (
                "https://api.deepseek.com/anthropic",
                "https://api.deepseek.com/anthropic/v1/messages",
            ),
            ("https://api.deepseek.com/v1", "https://api.deepseek.com/v1/messages"),
        ]
        for base, expected in cases:
            with self.subTest(base=base):
                provider = AnthropicMessagesProvider(base, "sk-ant", "claude-sonnet-4")
                with patch("requests.post") as post:
                    post.return_value = make_response(payload)
                    provider.complete("SYS", "USER")
                    self.assert_call(
                        post,
                        expected,
                        {
                            "x-api-key": "sk-ant",
                            "anthropic-version": "2023-06-01",
                            "Content-Type": "application/json",
                        },
                        {
                            "model": "claude-sonnet-4",
                            "max_tokens": 64000,
                            "system": "SYS",
                            "messages": [{"role": "user", "content": "USER"}],
                            "effort": "high",
                        },
                    )

    def test_anthropic_extracts_text_and_skips_thinking(self):
        provider = AnthropicMessagesProvider("https://api.deepseek.com", "sk-ant", "claude")
        with patch("requests.post") as post:
            post.return_value = make_response(dict(ANTHROPIC_PAYLOAD))
            text = provider.complete("SYS", "USER")
        self.assertEqual(text, "你好，世界")
        self.assertNotIn("思考", text)

    def test_anthropic_usage_computes_total(self):
        provider = AnthropicMessagesProvider("https://api.deepseek.com", "sk-ant", "claude")
        with patch("requests.post") as post:
            post.return_value = make_response(dict(ANTHROPIC_PAYLOAD))
            provider.complete("SYS", "USER")
        self.assertEqual(
            provider.last_usage, LLMUsage(prompt_tokens=8, completion_tokens=4, total_tokens=12)
        )
        self.assertEqual(provider.last_response["id"], "msg-1")

    # -- errors ------------------------------------------------------------
    def test_http_400_raises_llm_error(self):
        provider = OpenAIChatProvider("https://api.deepseek.com", "sk-test", "m")
        with patch("requests.post") as post:
            post.return_value = make_response(None, status_code=400, text="bad request")
            with self.assertRaises(LLMError) as ctx:
                provider.complete("SYS", "USER")
        message = str(ctx.exception)
        self.assertIn("400", message)
        self.assertIn("bad request", message)
        self.assertIsNone(provider.last_usage)
        self.assertIsNone(provider.last_response)

    def test_request_exception_raises_llm_error(self):
        provider = OpenAIChatProvider("https://api.deepseek.com", "sk-test", "m")
        with patch("requests.post") as post:
            post.side_effect = requests.RequestException("connection reset")
            with self.assertRaises(LLMError) as ctx:
                provider.complete("SYS", "USER")
        self.assertIn("connection reset", str(ctx.exception))

    def test_invalid_json_raises_llm_error(self):
        provider = OpenAIChatProvider("https://api.deepseek.com", "sk-test", "m")
        bad = MagicMock(status_code=200, text="<html>oops</html>")
        bad.json = MagicMock(side_effect=ValueError("no json"))
        with patch("requests.post") as post:
            post.return_value = bad
            with self.assertRaises(LLMError) as ctx:
                provider.complete("SYS", "USER")
        self.assertIn("oops", str(ctx.exception))

    def test_no_retry_on_error(self):
        provider = OpenAIChatProvider("https://api.deepseek.com", "sk-test", "m")
        with patch("requests.post") as post:
            post.return_value = make_response(None, status_code=500, text="boom")
            with self.assertRaises(LLMError):
                provider.complete("SYS", "USER")
            self.assertEqual(post.call_count, 1)

    # -- factory -----------------------------------------------------------
    def test_create_provider_mapping(self):
        mapping = {
            "openai_chat": OpenAIChatProvider,
            "openai_responses": OpenAIResponsesProvider,
            "anthropic_messages": AnthropicMessagesProvider,
        }
        for api_type, cls in mapping.items():
            with self.subTest(api_type=api_type):
                provider = create_provider(
                    api_type, "https://api.deepseek.com", "sk-test", "m", temperature=0.5
                )
                self.assertIsInstance(provider, cls)
                self.assertIsInstance(provider, BaseLLMProvider)
                self.assertEqual(provider.api_type, api_type)
                self.assertEqual(provider.temperature, 0.5)

    def test_create_provider_unknown_type(self):
        with self.assertRaises(ValueError) as ctx:
            create_provider("nope", "https://api.deepseek.com", "sk-test", "m")
        message = str(ctx.exception)
        for valid in ("openai_chat", "openai_responses", "anthropic_messages"):
            self.assertIn(valid, message)

    # -- validation --------------------------------------------------------
    def test_empty_config_value_errors(self):
        cases = [
            ({"base_url": "  ", "api_key": "sk", "model": "m"}, "llm_base_url"),
            ({"base_url": "https://x", "api_key": "", "model": "m"}, "llm_api_key"),
            ({"base_url": "https://x", "api_key": "sk", "model": "\t"}, "llm_model"),
        ]
        for kwargs, field in cases:
            with self.subTest(field=field):
                with self.assertRaises(ValueError) as ctx:
                    OpenAIChatProvider(**kwargs)
                message = str(ctx.exception)
                self.assertIn(field, message)
                self.assertIn("不能为空", message)

    def test_initial_state(self):
        provider = OpenAIChatProvider("https://api.deepseek.com", "sk-test", "m")
        self.assertEqual(provider.base_url, "https://api.deepseek.com")
        self.assertEqual(provider.api_key, "sk-test")
        self.assertEqual(provider.model, "m")
        self.assertEqual(provider.temperature, 0.2)
        self.assertEqual(provider.reasoning_effort, "high")
        self.assertEqual(provider.max_tokens, 64000)
        self.assertEqual(provider.timeout, 900)
        self.assertIsNone(provider.proxy)
        self.assertIsNone(provider.last_usage)
        self.assertIsNone(provider.last_response)
        self.assertEqual(BaseLLMProvider.api_type, "base")


if __name__ == "__main__":
    unittest.main()
