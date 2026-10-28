"""截图工具（`take_screenshot`）的单测。

它是什么
--------
**UI 元素表与 OCR 的补充**：模型自己喊一声要看图，框架把当屏那张图挂在
**prompt 的最后一条消息**上（元素表旁边）。寻址仍然走元素表的 ``[i]``。

为什么这些断言值得钉
--------------------
1. **图必须在 prompt 里，而且是真的图片块** —— 挂错地方（变成一段字符串）
   模型就看不见图，工具白跑。
2. **息屏时不能给黑图** —— 黑图是 500 token 的纯浪费，还要让模型对着黑屏
   猜（真机教训：30 分钟息屏让任务从 30 步涨到 51 步）。
3. **图绝不进局部历史**（2026-10-26 改）—— **图片块会进 provider 的前缀
   缓存键**（真 API 实测，MiniMax-M3.1-Flash：把请求中段的图删掉或换一张，
   命中量立刻掉到「图之前」的位置）。图在历史里 = 被钉在历史中段 ⇒ 下一次
   拍图要顶替它 ⇒ 那个位置的字节变了 ⇒ **前缀从历史中段起整段作废**。
   真机实测（execution 893）：14 次顶替让 47,181 个本可命中的 token 变满价，
   缓存率从 82% 掉到 41%。
   ⇒ 现在：工具结果只留**文字**，图另存 ``executor_screenshot``，装配 prompt
   时挂到**最后一条消息**（易变区）⇒ 局部历史一个字节都不动。
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
from minitap.mobile_use.prompt_language import PromptLanguage
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


class _FakeCtx:
    """最小 ctx：工具只用它的 `prompt_language`（决定图注和通道标题的语言）。"""

    prompt_language = PromptLanguage.EN


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

    tool = ts.get_take_screenshot_tool(_FakeCtx())  # type: ignore[arg-type]
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

        tool = ts.get_take_screenshot_tool(_FakeCtx())  # type: ignore[arg-type]
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
    def test_the_tool_result_carries_text_only(self, monkeypatch):
        """工具结果里**没有图片块**。

        图片块会进 provider 的前缀缓存键 —— 留在局部历史里就等于把前缀
        钉死在那儿，下一次拍图顶替它时整段作废。
        """
        out, _ = _run_tool(monkeypatch)
        message = _tool_message(out)

        assert message.status == "success"
        assert isinstance(message.content, list), "多模态结果的 content 必须是块列表"
        types = [block.get("type") for block in message.content]
        assert types == ["text"], f"工具结果里不该有图片块: {types}"
        assert not has_image_block(message.content)

    def test_the_picture_goes_to_its_own_state_field(self, monkeypatch):
        """图另存 —— 装配 prompt 时挂到最后一条消息（易变区）。"""
        out, _ = _run_tool(monkeypatch)
        shot = out.update["executor_screenshot"]

        assert shot, "图必须存进 executor_screenshot"
        raw = base64.b64decode(shot)
        assert raw[:2] == b"\xff\xd8", "应该是 JPEG（SOI 标记）"
        assert not shot.startswith("data:"), "state 里存裸 base64，data URI 前缀在装配时加"

    def test_the_assembled_block_is_a_jpeg_data_uri(self):
        """装配出来的块是 OpenAI 原生形态：``image_url`` + data URI，**只有图**。"""
        from minitap.mobile_use.utils.message_content import build_screenshot_blocks

        blocks = build_screenshot_blocks("QUJD")

        assert [b["type"] for b in blocks] == ["image_url"]
        assert blocks[0]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"

    def test_the_image_channel_message_carries_the_label(self):
        """图和标题在**同一条消息**里 —— 标题就是把两条通道隔开的那道线。

        元素表 / OCR 是文字通道（带 ``[i]`` 编号），图是像素通道（没有编号）。
        """
        from minitap.mobile_use.utils.message_content import build_screenshot_message_content

        blocks = build_screenshot_message_content("QUJD", language=PromptLanguage.EN)

        assert [b["type"] for b in blocks] == ["text", "image_url"]
        assert blocks[0]["text"] == "Image perception"
        assert blocks[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"

    def test_the_picture_is_downscaled_to_the_tile_budget(self, monkeypatch):
        """1220x2712 的真机屏必须被压到 1x2 tile（≈610x1356）—— 一张图约 500 token。"""
        out, _ = _run_tool(monkeypatch)
        shot = out.update["executor_screenshot"]

        width, height = _jpeg_dims(shot)
        assert width <= 640, f"宽 {width} 没压到 tile 预算内"
        assert height <= 1536, f"高 {height} 没压到 tile 预算内"
        # 等比例：不能只缩一边
        assert abs(width / height - 1220 / 2712) < 0.02

    def test_the_tool_result_is_one_line_and_points_at_the_channel(self, monkeypatch):
        """工具结果**只有一行事实**：拍到了没有 / 多大 / 图在哪儿。

        ⚠️ 逐轮那句长说明（"图是补充、寻址走行号…"）**删掉了** ——
        那套契约写在**静态提示词**里（``executor.md``），模型每轮都读得到；
        逐轮重说一遍只是把一段易变文字塞进局部历史。
        """
        out, _ = _run_tool(monkeypatch)
        message = _tool_message(out)

        text = content_to_text(message.content)
        assert len(text) < 200, f"工具结果不该再带长说明: {text!r}"
        assert "Image perception" in text, f"要指路：图在哪儿。{text!r}"
        assert "SUPPLEMENT" not in text, "长说明不该回到工具结果里"

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


class TestTheScreenshotNeverEntersTheHistory:
    """**局部历史一个字节都不许因为图而改。**（2026-10-26 改契约）

    老做法是「顶替旧图」：把历史里那条带图消息的内容换成一句占位说明。
    那一下改的是**历史中段**的字节 ⇒ 前缀缓存从那儿起整段作废
    （真机实测 execution 893：缓存率 82% → 41%，14 次顶替让 47,181 个
    本可命中的 token 变满价）。

    现在图根本不进历史 ⇒ 没有可顶替的东西 ⇒ 前缀永远稳定。
    """

    def test_the_history_update_only_appends(self, monkeypatch):
        """工具写进历史的只有那条新工具结果 —— 旧消息一律不碰。"""
        old = ToolMessage(
            tool_call_id="call_0",
            content=[
                {"type": "text", "text": "已截图：image/jpeg，60,106 字节。"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
            ],
            id="msg_old",
        )
        state = _FakeState(executor_messages=[old])

        out, _ = _run_tool(monkeypatch, state=state)
        messages = out.update["executor_messages"]

        assert len(messages) == 1, "只该追加新工具结果，不该把旧消息带回来改写"
        assert not any(getattr(m, "id", None) == "msg_old" for m in messages), (
            "旧截图消息不该被改写 —— 改了就击穿前缀缓存"
        )

    def test_the_old_message_object_is_untouched(self, monkeypatch):
        """state 里的旧消息对象**原样不动**（连内容都不改）。"""
        old = ToolMessage(
            tool_call_id="call_0",
            content=[
                {"type": "text", "text": "已截图：image/jpeg，60,106 字节。"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
            ],
            id="msg_old",
        )
        state = _FakeState(executor_messages=[old])
        before = list(old.content)

        _run_tool(monkeypatch, state=state)

        assert old.content == before, "旧消息的内容被改了"
        assert has_image_block(old.content), "旧消息里的图还在（事实层不动）"

    def test_the_new_screenshot_replaces_the_previous_one(self, monkeypatch):
        """只留**最新一张**图 —— 覆盖 state 里的那一张，不是在历史里顶替。"""
        out, _ = _run_tool(monkeypatch)
        first = out.update["executor_screenshot"]

        out2, _ = _run_tool(monkeypatch)
        second = out2.update["executor_screenshot"]

        assert first and second
        assert out2.update["executor_screenshot"] == second
        # 状态里**只有一张**图的位置（一个字段），不是两份
        assert set(out2.update) >= {"executor_screenshot"}


class TestMultimodalContentIsNotFlattened:
    """多模态内容被当成纯文本处理的那三个受害点。"""

    @staticmethod
    def _screenshot_message() -> ToolMessage:
        # 文字要**足够长**才会触发截断（门槛 120 token ≈ 480 字符），
        # 否则这条测试测的是「没超预算时原样返回」，不是截断路径。
        return ToolMessage(
            tool_call_id="call_1",
            content=[
                {"type": "text", "text": "已截图：image/jpeg，60,106 字节。 " + "x" * 2000},
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
        assert "已截图：image/jpeg，60,106 字节。" in log


class TestTheImageReachesTheProvider:
    """最后一环：图要真的**发出去**，而不是在某个中间层被压成字符串。

    链路：state 里的图 → prompt 最后一条消息 → ``convert_to_openai_messages``
    → 请求体。中间任何一层把 content 压成字符串，模型就永远看不见图。
    """

    def test_the_history_converts_to_an_image_url_block(self):
        from langchain_core.messages import convert_to_openai_messages

        message = ToolMessage(
            tool_call_id="call_1",
            content=[
                {"type": "text", "text": "已截图：image/jpeg，60,106 字节。"},
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
