"""Standalone configuration tests for the AiSubV2 plugin.

The plugin module (``plugins.v2/aisubv2/__init__.py``) imports MoviePilot and a
few optional third-party packages.  None of them are installed in this
environment, so we install lightweight stubs into :data:`sys.modules` *before*
importing the plugin and then exercise the v2.3 configuration surface.

The MoviePilot ``get_config`` stub intentionally raises ``AssertionError`` for
the ``"ChatGPT"`` key: the plugin no longer depends on the central ChatGPT
configuration and asking for it must fail loudly.
"""

import pathlib
import shutil
import sys
import tempfile
import types
import unittest


REPO = pathlib.Path(__file__).resolve().parents[3]
# The source directory is ``plugins.v2/``, whose name is not a valid Python
# identifier, so the plugin package is imported under its runtime name
# (``aisubv2``) -- the same name MoviePilot uses as ``app.plugins.aisubv2``.
PLUGIN_ROOT = REPO / "plugins.v2"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))


# ---------------------------------------------------------------------------
# Observation sinks used by the stubs.
# ---------------------------------------------------------------------------
LOG_RECORDS = []
GET_CONFIG_KEYS = []
POSTED_MESSAGES = []


def _new_module(name, **attrs):
    """Register a fresh :class:`types.ModuleType` under *name*."""
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    sys.modules[name] = module
    return module


def _install_moviepilot_stubs():
    """Install minimal ``app.*`` (MoviePilot) stubs."""

    # -- package skeleton -------------------------------------------------
    app_pkg = _new_module("app")
    core_pkg = _new_module("app.core")
    schemas_pkg = _new_module("app.schemas")
    utils_pkg = _new_module("app.utils")
    app_pkg.core = core_pkg
    app_pkg.schemas = schemas_pkg
    app_pkg.utils = utils_pkg

    # -- app.core.config --------------------------------------------------
    settings = types.SimpleNamespace(
        RMT_MEDIAEXT={".mp4", ".mkv"},
        PROXY={},
    )
    core_pkg.config = _new_module("app.core.config", settings=settings)

    # -- app.core.context -------------------------------------------------
    class MediaInfo:
        pass

    core_pkg.context = _new_module("app.core.context", MediaInfo=MediaInfo)

    # -- app.core.event ---------------------------------------------------
    class Event:
        def __init__(self, event_type=None, event_data=None):
            self.event_type = event_type
            self.event_data = event_data

    class _EventManager:
        def __init__(self):
            self.registrations = []

        def register(self, *args, **kwargs):
            self.registrations.append((args, kwargs))

            def _decorator(func):
                return func

            return _decorator

    core_pkg.event = _new_module(
        "app.core.event", eventmanager=_EventManager(), Event=Event
    )

    # -- app.schemas ------------------------------------------------------
    class TransferInfo:
        pass

    schemas_pkg.TransferInfo = TransferInfo

    class NotificationType:
        Plugin = "Plugin"

    class EventType:
        TransferComplete = "TransferComplete"

    schemas_pkg.types = _new_module(
        "app.schemas.types",
        NotificationType=NotificationType,
        EventType=EventType,
    )

    # -- app.log ----------------------------------------------------------
    class _StubLogger:
        @staticmethod
        def _append(level, message):
            LOG_RECORDS.append((level, str(message)))

        def info(self, message="", *args, **kwargs):
            self._append("info", message)

        def warn(self, message="", *args, **kwargs):
            self._append("warn", message)

        def warning(self, message="", *args, **kwargs):
            self._append("warning", message)

        def error(self, message="", *args, **kwargs):
            self._append("error", message)

        def debug(self, message="", *args, **kwargs):
            self._append("debug", message)

    app_pkg.log = _new_module("app.log", logger=_StubLogger())

    # -- app.utils.system -------------------------------------------------
    class SystemUtils:
        @staticmethod
        def copy(src, dst):
            shutil.copyfile(str(src), str(dst))
            return dst

    utils_pkg.system = _new_module("app.utils.system", SystemUtils=SystemUtils)

    # -- app.plugins ------------------------------------------------------
    class _PluginBase:
        def __init__(self, *args, **kwargs):
            self._stub_data = {}
            self._stub_data_path = None
            self._stub_updated_config = None

        def get_data(self, key, default=None):
            if key in self._stub_data:
                return self._stub_data[key]
            return {} if default is None else default

        def save_data(self, key, value):
            self._stub_data[key] = value

        def update_config(self, cfg):
            self._stub_updated_config = cfg

        def get_data_path(self):
            if self._stub_data_path is None:
                self._stub_data_path = pathlib.Path(
                    tempfile.mkdtemp(prefix="AiSubV2-stub-")
                )
            return self._stub_data_path

        def post_message(self, **kwargs):
            POSTED_MESSAGES.append(kwargs)

        def get_config(self, key):
            GET_CONFIG_KEYS.append(key)
            if key == "ChatGPT":
                raise AssertionError(
                    "插件不应再读取 MoviePilot 的 ChatGPT 配置（v2.3 已改为独立大模型配置）"
                )
            return None

    app_pkg.plugins = _new_module("app.plugins", _PluginBase=_PluginBase)


def _install_third_party_stubs():
    """Install stubs for the optional dependencies the plugin imports."""
    if "iso639" not in sys.modules:

        class NonExistentLanguageError(ValueError):
            pass

        _new_module(
            "iso639",
            find=lambda name: name or None,
            to_iso639_1=lambda name: name,
            NonExistentLanguageError=NonExistentLanguageError,
        )

    if "psutil" not in sys.modules:
        _new_module("psutil", cpu_count=lambda logical=True: 4)

    if "srt" not in sys.modules:
        _new_module(
            "srt",
            Subtitle=object,
            parse=lambda *args, **kwargs: [],
            compose=lambda *args, **kwargs: "",
        )

    if "lxml" not in sys.modules:

        class _FakeHtmlElement:
            def xpath(self, expression):
                return ""

        lxml_module = _new_module("lxml")
        lxml_module.etree = _new_module(
            "lxml.etree", HTML=lambda *args, **kwargs: _FakeHtmlElement()
        )


_install_moviepilot_stubs()
_install_third_party_stubs()

import aisubv2 as plugin_module  # noqa: E402  (import after stubbing)


PLUGIN_API_MODELS = [
    "llm_api_type",
    "llm_base_url",
    "llm_api_key",
    "llm_model",
    "llm_reasoning_effort",
    "llm_max_tokens",
    "max_retries",
    "zh_only",
    "ignore_if_chinese_exists",
]

REMOVED_CONFIG_KEYS = ["enable_batch", "batch_size", "context_window", "enable_merge"]


def _iter_model_props(node):
    """Yield every ``props`` dict that declares a ``model`` key."""
    if isinstance(node, dict):
        props = node.get("props")
        if isinstance(props, dict) and "model" in props:
            yield props
        for value in node.values():
            yield from _iter_model_props(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_model_props(item)


class AiSubV2PluginConfigTest(unittest.TestCase):
    def setUp(self):
        LOG_RECORDS.clear()
        GET_CONFIG_KEYS.clear()
        POSTED_MESSAGES.clear()
        self.plugin = plugin_module.AiSubV2()

    # -- 1. version -------------------------------------------------------
    def test_plugin_version_is_2_3(self):
        self.assertEqual(plugin_module.AiSubV2.plugin_version, "2.3")

    # -- 2. defaults + removed keys --------------------------------------
    def test_get_form_defaults_and_removed_keys(self):
        form = self.plugin.get_form()
        self.assertIsInstance(form, tuple)
        self.assertEqual(len(form), 2)
        schema, defaults = form

        expected_defaults = {
            "llm_api_type": "openai_chat",
            "llm_base_url": "",
            "llm_api_key": "",
            "llm_model": "",
            "llm_reasoning_effort": "high",
            "llm_max_tokens": 64000,
            "max_retries": 3,
            "zh_only": False,
            "ignore_if_chinese_exists": True,
        }
        for key, value in expected_defaults.items():
            self.assertIn(key, defaults, f"默认配置缺少 {key}")
            self.assertEqual(defaults[key], value, f"{key} 默认值不符")
            self.assertIs(
                type(defaults[key]), type(value), f"{key} 默认值类型不符"
            )

        for key in REMOVED_CONFIG_KEYS:
            self.assertNotIn(key, defaults, f"已废弃的配置项不应出现在默认值中：{key}")

        schema_models = {props["model"] for props in _iter_model_props(schema)}
        for key in REMOVED_CONFIG_KEYS:
            self.assertNotIn(key, schema_models, f"已废弃的配置项不应出现在表单中：{key}")

    # -- 3. schema exposes the LLM surface -------------------------------
    def test_llm_schema_exposes_required_models(self):
        schema, _defaults = self.plugin.get_form()
        model_props = {}
        for props in _iter_model_props(schema):
            model_props[props["model"]] = props

        for model in PLUGIN_API_MODELS:
            self.assertIn(model, model_props, f"配置表单缺少字段 {model}")

        self.assertEqual(
            model_props["llm_api_key"].get("type"),
            "password",
            "llm_api_key 应为 password 类型",
        )

    # -- 4. openai_chat wiring + no ChatGPT dependency --------------------
    def test_translate_zh_wires_openai_chat_provider(self):
        self.plugin.init_plugin(
            {
                "enabled": False,
                "enable_asr": False,
                "translate_zh": True,
                "llm_base_url": "https://api.deepseek.com",
                "llm_api_key": "sk-test",
                "llm_model": "deepseek-flash",
            }
        )

        self.assertIsNotNone(self.plugin._llm_provider)
        self.assertEqual(
            type(self.plugin._llm_provider).__name__, "OpenAIChatProvider"
        )
        # The plugin must not consult the legacy central ChatGPT config.
        self.assertNotIn("ChatGPT", GET_CONFIG_KEYS)
        # ...and the guard stub really would have failed loudly.
        with self.assertRaises(AssertionError):
            self.plugin.get_config("ChatGPT")

    # -- 5. other provider types ------------------------------------------
    def test_translate_zh_wires_alternative_providers(self):
        cases = {
            "anthropic_messages": "AnthropicMessagesProvider",
            "openai_responses": "OpenAIResponsesProvider",
        }
        for api_type, class_name in cases.items():
            with self.subTest(api_type=api_type):
                plugin = plugin_module.AiSubV2()
                plugin.init_plugin(
                    {
                        "enabled": False,
                        "enable_asr": False,
                        "translate_zh": True,
                        "llm_api_type": api_type,
                        "llm_base_url": "https://api.example.com",
                        "llm_api_key": "sk-test",
                        "llm_model": "some-model",
                    }
                )
                self.assertIsNotNone(plugin._llm_provider)
                self.assertEqual(
                    type(plugin._llm_provider).__name__, class_name
                )
                self.assertNotIn("ChatGPT", GET_CONFIG_KEYS)

    # -- 6. blank credentials degrade gracefully --------------------------
    def test_blank_llm_api_key_keeps_provider_none(self):
        self.plugin.init_plugin(
            {
                "enabled": False,
                "enable_asr": False,
                "translate_zh": True,
                "llm_base_url": "https://api.deepseek.com",
                "llm_api_key": "",
                "llm_model": "deepseek-flash",
            }
        )

        self.assertIsNone(
            self.plugin._llm_provider,
            "缺少 llm_api_key 时不应创建 provider",
        )
        errors = [message for level, message in LOG_RECORDS if level == "error"]
        self.assertTrue(
            any("llm" in message or "大模型" in message for message in errors),
            f"应记录配置错误日志，实际日志：{LOG_RECORDS}",
        )

    # -- 7. zh_only / ignore_if_chinese_exists ----------------------------
    def test_zh_only_and_ignore_defaults_when_omitted(self):
        self.plugin.init_plugin(
            {"enabled": False, "enable_asr": False, "translate_zh": False}
        )
        self.assertIs(self.plugin._zh_only, False)
        self.assertIs(self.plugin._ignore_if_chinese_exists, True)

    def test_zh_only_and_ignore_explicit_values(self):
        self.plugin.init_plugin(
            {
                "enabled": False,
                "enable_asr": False,
                "translate_zh": False,
                "zh_only": True,
                "ignore_if_chinese_exists": False,
            }
        )
        self.assertIs(self.plugin._zh_only, True)
        self.assertIs(self.plugin._ignore_if_chinese_exists, False)

    # -- 8. regression: incomplete config must still stop the service ------
    def test_incomplete_llm_config_still_stops_service(self):
        stopped = []
        self.plugin.stop_service = lambda: stopped.append(True)
        self.plugin.init_plugin(
            {
                "enabled": False,
                "enable_asr": False,
                "translate_zh": True,
                "llm_base_url": "https://api.deepseek.com",
                "llm_api_key": "",
                "llm_model": "deepseek-flash",
            }
        )
        self.assertTrue(stopped, "配置不完整时也必须执行停止流程，避免残留消费线程")
        self.assertIsNone(self.plugin._llm_provider)

    # -- 9. regression: stale provider must be cleared on re-init ---------
    def test_llm_provider_is_reset_on_reinit(self):
        valid = {
            "enabled": False,
            "enable_asr": False,
            "translate_zh": True,
            "llm_base_url": "https://api.deepseek.com",
            "llm_api_key": "sk-test",
            "llm_model": "deepseek-flash",
        }
        self.plugin.init_plugin(dict(valid))
        self.assertIsNotNone(self.plugin._llm_provider)

        self.plugin.init_plugin({"enabled": False, "enable_asr": False, "translate_zh": False})
        self.assertIsNone(self.plugin._llm_provider, "关闭翻译后应清空旧客户端")

        self.plugin.init_plugin(dict(valid))
        self.assertIsNotNone(self.plugin._llm_provider)
        bad = dict(valid)
        bad["llm_api_key"] = ""
        self.plugin.init_plugin(bad)
        self.assertIsNone(self.plugin._llm_provider, "配置失效后不应保留旧客户端")


if __name__ == "__main__":
    unittest.main()
