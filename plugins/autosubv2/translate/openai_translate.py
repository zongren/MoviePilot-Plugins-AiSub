"""LLM provider 适配层，只依赖标准库与 ``requests``。

本模块把三种常见的 LLM HTTP 线格式（wire format）统一成同一个
``BaseLLMProvider.complete(system, user) -> str`` 接口，供字幕翻译引擎使用：

- ``openai_chat``（``/v1/chat/completions``）：
  OpenAI 兼容的 Chat Completions。system/user 作为两条 ``messages`` 发送，
  正文取自 ``choices[0].message.content``。
- ``openai_responses``（``/v1/responses``）：
  OpenAI Responses API。``instructions`` 承载 system，``input`` 承载 user，
  ``reasoning.effort`` 表示推理强度，正文由 ``output[].content[]`` 中
  ``type == "output_text"`` 的片段拼接而成（``output_text`` 作为回退）。
- ``anthropic_messages``（``/v1/messages``）：
  Anthropic Messages API。system 是顶层 ``system`` 字段，user 是唯一一条
  ``messages``，正文由 ``content[]`` 中 ``type == "text"`` 的片段拼接而成。

每次 ``complete`` 只发起一次 HTTP 请求（不做内部重试），失败时抛出
``LLMError``；重试由上层翻译引擎负责。调用成功后 ``last_usage`` 保存本次
token 用量，``last_response`` 保存解析后的响应体。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

__all__ = [
    "LLMError",
    "LLMUsage",
    "BaseLLMProvider",
    "OpenAIChatProvider",
    "OpenAIResponsesProvider",
    "AnthropicMessagesProvider",
    "create_provider",
]


class LLMError(Exception):
    """LLM 调用失败（HTTP 错误、网络异常或响应无法解析）。"""


@dataclass
class LLMUsage:
    """一次 LLM 调用的 token 用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class BaseLLMProvider:
    """所有 provider 的公共基类。"""

    api_type: str = "base"
    path: str = ""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.2,
        reasoning_effort: str = "high",
        max_tokens: int = 64000,
        timeout: int = 900,
        proxy: Optional[dict] = None,
    ) -> None:
        base_url = (base_url or "").strip()
        api_key = (api_key or "").strip()
        model = (model or "").strip()
        if not base_url:
            raise ValueError("llm_base_url 不能为空")
        if not api_key:
            raise ValueError("llm_api_key 不能为空")
        if not model:
            raise ValueError("llm_model 不能为空")

        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.proxy = proxy
        self.last_usage: Optional[LLMUsage] = None
        self.last_response: Optional[dict] = None

    # -- HTTP helpers -------------------------------------------------------
    def _endpoint(self, path: str) -> str:
        """拼接 base_url 与 path，避免出现 ``/v1/v1`` 重复。"""
        base = self.base_url.rstrip("/")
        if base.endswith("/v1") and path.startswith("/v1/"):
            path = path[len("/v1"):]
        return base + path

    @staticmethod
    def _body_text(response: Any) -> str:
        text = getattr(response, "text", "") or ""
        if not isinstance(text, str):
            text = str(text)
        return text

    def _post(self, path: str, headers: Dict[str, str], body: Dict[str, Any]) -> dict:
        """发起唯一一次 POST，返回解析后的 JSON（dict）。"""
        url = self._endpoint(path)
        self.last_usage = None
        self.last_response = None

        try:
            response = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=self.timeout,
                proxies=self.proxy or None,
            )
        except requests.RequestException as exc:
            raise LLMError(f"请求 {url} 失败：{exc}") from exc

        status = getattr(response, "status_code", None)
        if status is not None and status >= 400:
            raise LLMError(f"HTTP {status}：{self._body_text(response)[:500]}")

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(f"响应不是合法 JSON：{self._body_text(response)[:500]}") from exc

        if not isinstance(data, dict):
            raise LLMError("响应 JSON 结构异常：顶层不是对象")

        self.last_response = data
        self._parse_usage(data)
        return data

    def _parse_usage(self, data: dict) -> None:
        raw = data.get("usage")
        if not isinstance(raw, dict):
            return
        prompt = raw.get("prompt_tokens", raw.get("input_tokens", 0)) or 0
        completion = raw.get("completion_tokens", raw.get("output_tokens", 0)) or 0
        total = raw.get("total_tokens")
        try:
            prompt = int(prompt)
            completion = int(completion)
            total = int(total) if total is not None else prompt + completion
        except (TypeError, ValueError):
            return
        self.last_usage = LLMUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )

    # -- public API ---------------------------------------------------------
    def complete(self, system: str, user: str) -> str:  # pragma: no cover - base
        raise NotImplementedError(f"{type(self).__name__} 未实现 complete()")


class OpenAIChatProvider(BaseLLMProvider):
    """OpenAI 兼容的 Chat Completions。"""

    api_type = "openai_chat"
    path = "/v1/chat/completions"

    def complete(self, system: str, user: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
        }
        data = self._post(self.path, headers, body)

        content = ""
        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                text = message.get("content")
                if isinstance(text, str):
                    content = text
        return content


class OpenAIResponsesProvider(BaseLLMProvider):
    """OpenAI Responses API。"""

    api_type = "openai_responses"
    path = "/v1/responses"

    def complete(self, system: str, user: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "instructions": system,
            "input": user,
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_tokens,
        }
        data = self._post(self.path, headers, body)

        parts = []
        output = data.get("output")
        if isinstance(output, list):
            for entry in output:
                if not isinstance(entry, dict):
                    continue
                content = entry.get("content")
                if not isinstance(content, list):
                    continue
                for sub in content:
                    if not isinstance(sub, dict):
                        continue
                    if sub.get("type") != "output_text":
                        continue
                    text = sub.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        result = "".join(parts)
        if not result:
            fallback = data.get("output_text")
            if isinstance(fallback, str):
                result = fallback
        return result


class AnthropicMessagesProvider(BaseLLMProvider):
    """Anthropic Messages API。"""

    api_type = "anthropic_messages"
    path = "/v1/messages"

    def complete(self, system: str, user: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "effort": self.reasoning_effort,
        }
        data = self._post(self.path, headers, body)

        parts = []
        content = data.get("content")
        if isinstance(content, list):
            for entry in content:
                if not isinstance(entry, dict):
                    continue
                if entry.get("type") != "text":
                    continue
                text = entry.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)


_PROVIDERS = {
    OpenAIChatProvider.api_type: OpenAIChatProvider,
    OpenAIResponsesProvider.api_type: OpenAIResponsesProvider,
    AnthropicMessagesProvider.api_type: AnthropicMessagesProvider,
}


def create_provider(
    api_type: str,
    base_url: str,
    api_key: str,
    model: str,
    **kwargs: Any,
) -> BaseLLMProvider:
    """根据 ``api_type`` 创建对应的 provider。"""
    key = (api_type or "").strip().lower()
    provider_cls = _PROVIDERS.get(key)
    if provider_cls is None:
        valid = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"未知的 api_type：{api_type!r}，可选值：{valid}")
    return provider_cls(base_url, api_key, model, **kwargs)
