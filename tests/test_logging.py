"""日志格式化器测试：断言「该带的信息真的带上了」。

这两个格式化器是排障入口，出错时不会有人来测它们，所以只能靠这里。
"""

import json
import logging
import unittest

from core.logging import JsonFormatter, TextFormatter, request_id_var, setup_logging


def _record(msg="access", **extra):
    rec = logging.LogRecord(
        name="api.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


class TestTextFormatter(unittest.TestCase):
    def setUp(self):
        self.fmt = TextFormatter()
        self._token = request_id_var.set("abcdef1234567890")

    def tearDown(self):
        request_id_var.reset(self._token)

    def test_request_id_prefix_is_shortened(self):
        """request_id 只显示前 8 位：完整 32 位会把行撑爆，日志反而难读。"""
        out = self.fmt.format(_record())
        self.assertIn("[abcdef12]", out)

    def test_extra_fields_are_rendered(self):
        """访问日志必须带 method / path / status / duration_ms。

        曾经漏过一次：只打印 msg，日志里全是 "access"，等于白记。
        """
        out = self.fmt.format(
            _record(method="POST", path="/api/chat", status=200, duration_ms=12.5)
        )
        for frag in ("method=POST", "path=/api/chat", "status=200", "duration_ms=12.5"):
            self.assertIn(frag, out)

    def test_no_extras_has_no_trailing_garbage(self):
        out = self.fmt.format(_record())
        self.assertNotIn("  =", out)
        self.assertTrue(out.endswith("access"))

    def test_uvicorn_color_message_is_filtered(self):
        """uvicorn 会给日志塞 color_message 模板（带 %d 占位符），不该出现在行尾。"""
        out = self.fmt.format(_record(color_message="Uvicorn running on %s://%s:%d"))
        self.assertNotIn("color_message", out)
        self.assertNotIn("%d", out)


class TestJsonFormatter(unittest.TestCase):
    def setUp(self):
        self.fmt = JsonFormatter()
        self._token = request_id_var.set("deadbeef")

    def tearDown(self):
        request_id_var.reset(self._token)

    def test_output_is_single_line_json(self):
        """一行一条，采集端才能按行切；多行 JSON 会把日志管道搞乱。"""
        out = self.fmt.format(_record(status=500))
        self.assertNotIn("\n", out)
        data = json.loads(out)
        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["msg"], "access")
        self.assertEqual(data["status"], 500)
        self.assertEqual(data["request_id"], "deadbeef")

    def test_non_ascii_is_not_escaped(self):
        """ensure_ascii=False：中文问题句要能直接看，不要变成 \\uXXXX。"""
        out = self.fmt.format(_record(msg="检索失败"))
        self.assertIn("检索失败", out)
        self.assertNotIn("\\u", out)

    def test_exception_is_included(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            rec = _record()
            rec.exc_info = sys.exc_info()
        data = json.loads(self.fmt.format(rec))
        self.assertIn("ValueError: boom", data["exc"])


class TestSetupLogging(unittest.TestCase):
    def test_idempotent(self):
        """重复初始化不能叠加 handler，否则每条日志打印两遍。"""
        setup_logging("development")
        n1 = len(logging.getLogger().handlers)
        setup_logging("development")
        n2 = len(logging.getLogger().handlers)
        self.assertEqual(n1, 1)
        self.assertEqual(n2, 1)

    def test_production_uses_json(self):
        setup_logging("production")
        handler = logging.getLogger().handlers[0]
        self.assertIsInstance(handler.formatter, JsonFormatter)
        setup_logging("test")  # 还原，避免影响其他用例
        self.assertIsInstance(logging.getLogger().handlers[0].formatter, TextFormatter)


if __name__ == "__main__":
    unittest.main()
