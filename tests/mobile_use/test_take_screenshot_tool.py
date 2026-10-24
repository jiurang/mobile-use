"""截图工具（`take_screenshot`）的单测。

它是什么
--------
**UI 元素表与 OCR 的补充**：模型自己喊一声要看图，框架把当屏那张图挂在
**工具结果**上（多模态内容块）。寻址仍然走元素表的 ``[i]``。

为什么这些断言值得钉
--------------------
1. **图必须在工具结果里，而且是真的图片块** —— 挂错地方（变成一段字符串）
   模型就看不见图，工具白跑。
2. **息屏时不能给黑图** —— 黑图是 500 token 的纯浪费，还要让模型对着黑屏
   猜（真机教训：30 分钟息屏让任务从 30 步涨到 51 步）。
3. **旧图必须被顶掉** —— 旧图对应旧屏，留着会误导模型；每张图约 500 input
   token，而局部历史每轮整段重发 ⇒ 攒图就是每轮烧钱。
   ⚠️ 顶掉的方式是**替换内容**，不是删消息：删 ToolMessage 会留下孤儿
   tool_call，OpenAI 兼容端点直接 400。
4. **多模态内容不能被当成纯文本处理** —— 以前那句 ``str(message.content)``
   会把整张图变成一坨 base64 字面量：Executor 那边图没了（模型看不到），
   Reporter 那边汇报正文多出几十 KB 噪声。
"""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO

from langchain_core.messages import AIMessage, ToolMessage
from PIL import Image

from minitap.mobile_use.agents.executor.prompt import truncate_local_history
from minitap.mobile_use.agents.reporter.reporter import render_execution_log
from minitap.mobile_use.tools.index import EXECUTOR_WRAPPERS_TOOLS
from minitap.mobile_use.utils.message_content import content_to_text, has_image_block


def _make_png(w: int = 1220, h: int = 2712) -> str:
    """一张假手机屏（PNG base64）。"""
    img = Image.new("RGB", (w, h), (120, 180, 240))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _jpeg_dims(b64: str) -> tuple[int, int]:
    img = Image.open(BytesIO(base64.b64decode(b64)))
    return img.width, img.height


class _FakeState:
    """只提供 `asanitize_update` 的假 state。"""

    def __init__(self, executor_messages=None):
        self.executor_messages = executor_messages or []

    async def asanitize_update(self, *, ctx, update, agent):
        return update


def _run_tool(monkeypatch, *, power="on", image=None, error=None, state=None):
    """跑一次工具，返回 (Command, 假控制器)。"""
    from minitap.mobile_use.tools.mobile import take_screenshot as ts

    calls: list[str] = []

    class _FakeController:
        def __init__(self, ctx):
            calls.append("controller")

        async def get_screen_power_state(self) -> str:
            calls.append("power")
            return power

        async def take_screenshot(self) -> str:
            calls.append("capture")
            if error is not None:
                raise error
            return image if image is not None else _make_png()

    monkeypatch.setattr(ts, "UnifiedMobileController", _FakeController)

    tool = ts.get_take_screenshot_tool(None)  # type: ignore[arg-type]
    out = asyncio.run(
        tool.coroutine(  # type: ignore[misc]
            agent_thought="元素表看不清这个图标",
            tool_call_id="call_1",
            state=state or _FakeState(),
        )
    )
    return out, calls


def _tool_message(out) -> ToolMessage:
    messages = out.update["executor_messages"]
    assert messages, "工具结果没写进 executor_messages"
    return messages[-1]


class TestToolRegistration:
    def test_tool_is_in_the_executor_toolset(self):
        """注册进去，直连派发（dispatch_decisions 按名字查表）才认得。"""
        names = {w.tool_fn_getter.__name__ for w in EXECUTOR_WRAPPERS_TOOLS}
        assert "get_take_screenshot_tool" in names

    def test_tool_has_no_required_args_besides_thought(self):
        """除 agent_thought 外没有必填参数 —— 模型只要写 action 就能派发。"""
        from minitap.mobile_use.tools.mobile import take_screenshot as ts

        tool = ts.get_take_screenshot_tool(None)  # type: ignore[arg-type]
        fields = getattr(tool.args_schema, "model_fields", {})
        required = {
            name
            for name, f in fields.items()
            if f.is_required() and name not in {"tool_call_id", "state", "agent_thought"}
        }
        assert required == set(), f"不该有模型必须自己填的参数: {required}"

    def test_tool_does_not_change_the_screen(self):
        """截图不动屏幕 ⇒ 不该进「动过屏幕的工具」名单。

        进了的话，每截一次图 contextor 都要多等一轮沉降，纯粹变慢。
        """
        from minitap.mobile_use.agents.executor.tool_node import DEVICE_CHANGING_TOOLS

        assert "take_screenshot" not in DEVICE_CHANGING_TOOLS


class TestSuccess:
    def test_returns_text_plus_an_image_block(self, monkeypatch):
        out, _ = _run_tool(monkeypatch)
        message = _tool_message(out)

        assert message.status == "success"
        assert isinstance(message.content, list), "多模态结果的 content 必须是块列表"
        types = [block.get("type") for block in message.content]
        assert types == ["text", "image_url"], f"块顺序/类型不对: {types}"

    def test_the_image_is_a_jpeg_data_uri(self, monkeypatch):
        out, _ = _run_tool(monkeypatch)
        message = _tool_message(out)

        url = message.content[1]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,"), url[:40]

    def test_the_picture_is_downscaled_to_the_tile_budget(self, monkeypatch):
        """1220x2712 的真机屏必须被压到 1x2 tile（≈610x1356）—— 一张图约 500 token。"""
        out, _ = _run_tool(monkeypatch)
        message = _tool_message(out)

        b64 = message.content[1]["image_url"]["url"].split(",", 1)[1]
        width, height = _jpeg_dims(b64)
        assert width <= 640, f"宽 {width} 没压到 tile 预算内"
        assert height <= 1536, f"高 {height} 没压到 tile 预算内"
        # 等比例：不能只缩一边
        assert abs(width / height - 1220 / 2712) < 0.02

    def test_the_text_says_it_is_a_supplement(self, monkeypatch):
        """图是补充 —— 寻址仍然走元素表。这句话是给模型的契约。"""
        out, _ = _run_tool(monkeypatch)
        message = _tool_message(out)

        text = content_to_text(message.content)
        assert "SUPPLEMENT" in text
        assert "element table" in text

    def test_the_thought_line_stays_short(self, monkeypatch):
        """写进 agents_thoughts 的那句不能把整段说明搬过去。

        agents_thoughts 是**全员共享**的记忆，cortex 每轮都要读它 ——
        把图的长说明塞进去会撑爆 cortex 的 prompt。
        """
        out, _ = _run_tool(monkeypatch)
        thoughts = out.update["agents_thoughts"]
        assert len(thoughts) == 2
        assert len(thoughts[1]) < 200, f"agents_thoughts 那句太长: {thoughts[1]!r}"


class TestScreenOff:
    def test_reports_the_screen_is_off_instead_of_sending_a_black_picture(self, monkeypatch):
        """息屏时截图是全黑的 —— 发过去是纯浪费，还要误导模型。"""
        out, calls = _run_tool(monkeypatch, power="off")
        message = _tool_message(out)

        assert message.status == "error"
        assert "wake_screen" in content_to_text(message.content)
        assert not has_image_block(message.content), "息屏时不该发图片"
        assert "capture" not in calls, "息屏时不该白抓一次屏"


class TestCaptureFailure:
    def test_failure_is_reported_with_the_reason(self, monkeypatch):
        out, _ = _run_tool(monkeypatch, error=RuntimeError("adb died"))
        message = _tool_message(out)

        assert message.status == "error"
        text = content_to_text(message.content)
        assert "adb died" in text, f"失败原因要原样回灌给模型: {text!r}"
        assert not has_image_block(message.content)


class TestSupersedesOlderScreenshots:
    def test_an_older_screenshot_is_replaced_by_a_note(self, monkeypatch):
        """旧图对应旧屏 —— 留着会误导模型，每轮还要重发一次。"""
        old = ToolMessage(
            tool_call_id="call_0",
            content=[
                {"type": "text", "text": "Screenshot of the current screen"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
            ],
            id="msg_old",
        )
        state = _FakeState(executor_messages=[old])

        out, _ = _run_tool(monkeypatch, state=state)
        messages = out.update["executor_messages"]

        replaced = [m for m in messages if getattr(m, "id", None) == "msg_old"]
        assert len(replaced) == 1, "旧截图消息必须还在（删了会留下孤儿 tool_call → 400）"
        assert not has_image_block(replaced[0].content), "旧图必须被顶掉"
        assert "superseded" in content_to_text(replaced[0].content).lower()

    def test_the_new_screenshot_is_still_there(self, monkeypatch):
        old = ToolMessage(
            tool_call_id="call_0",
            content=[
                {"type": "text", "text": "old"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
            ],
            id="msg_old",
        )
        state = _FakeState(executor_messages=[old])

        out, _ = _run_tool(monkeypatch, state=state)
        messages = out.update["executor_messages"]

        images = [m for m in messages if has_image_block(getattr(m, "content", None))]
        assert len(images) == 1, "只应留最新一张图"


class TestMultimodalContentIsNotFlattened:
    """多模态内容被当成纯文本处理的那三个受害点。"""

    @staticmethod
    def _screenshot_message() -> ToolMessage:
        # 文字要**足够长**才会触发截断（门槛 120 token ≈ 480 字符），
        # 否则这条测试测的是「没超预算时原样返回」，不是截断路径。
        return ToolMessage(
            tool_call_id="call_1",
            content=[
                {"type": "text", "text": "Screenshot of the current screen " + "x" * 2000},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
            ],
        )

    def test_truncation_keeps_the_image(self):
        """截断只截文字 —— 图被换成字符串的话，模型就看不见图了。"""
        message = self._screenshot_message()
        result = truncate_local_history([message])

        assert len(result) == 1
        content = result[0].content
        assert isinstance(content, list), "多模态 content 不能被换成字符串"
        assert has_image_block(content), "图片块必须原样保留"
        assert len(content_to_text(content)) < 2000, "文字部分应该被截短"

    def test_execution_log_does_not_dump_base64(self):
        """汇报正文里不能出现 base64 —— 既污染报告又白烧 reporter 的 token。"""
        log = render_execution_log([AIMessage(content="thinking"), self._screenshot_message()])

        assert "base64" not in log
        assert "QUJD" not in log
        assert "Screenshot of the current screen" in log


class TestTheImageReachesTheProvider:
    """最后一环：图要真的**发出去**，而不是在某个中间层被压成字符串。

    链路：工具结果 → 局部历史 → ``convert_to_openai_messages`` → 请求体。
    中间任何一层把 content 压成字符串，模型就永远看不见图。
    """

    def test_the_history_converts_to_an_image_url_block(self):
        from langchain_core.messages import convert_to_openai_messages

        message = ToolMessage(
            tool_call_id="call_1",
            content=[
                {"type": "text", "text": "Screenshot of the current screen"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
            ],
        )
        converted = convert_to_openai_messages([message])

        assert len(converted) == 1
        payload = converted[0]
        assert payload["role"] == "tool"
        assert payload["tool_call_id"] == "call_1"
        blocks = payload["content"]
        assert isinstance(blocks, list), "多模态 content 必须是块列表"
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "image_url"
        assert blocks[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_the_truncated_history_still_converts_with_the_image(self):
        """截断之后（发给 LLM 的那份）图还在 —— 这是模型唯一能看到图的地方。"""
        from langchain_core.messages import convert_to_openai_messages

        message = ToolMessage(
            tool_call_id="call_1",
            content=[
                {"type": "text", "text": "Screenshot " + "x" * 2000},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
            ],
        )
        payload = convert_to_openai_messages(truncate_local_history([message]))[0]

        blocks = payload["content"]
        assert blocks[1]["type"] == "image_url", "截断把图弄丢了"
        assert blocks[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
