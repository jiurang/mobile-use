"""Executor 的图片通道：**Executor 自己决定**，Manager 控制开关。

三个开关（Manager 侧 `AgentConfig.perception` → `MobileUseContext.perception`）：

* ``ui_hierarchy`` —— UI 元素表。关掉 ⇒ 不建表 ⇒ 走 legacy OCR 寻址。
* ``ocr`` —— OCR 文本。关掉 ⇒ **不跑 OCR**，也不给 OCR 块。
* ``screenshot`` —— 图片。``always`` / ``on_demand``（Executor 自己举手）/ ``off``。

这个文件锁住**执行层**的行为（提示词那一半在 ``test_screenshot_on_demand.py``）：

1. 通道开着 + Executor 举过手 ⇒ 这一轮挂图；没举手 ⇒ 不挂。
2. ``always`` ⇒ 每轮都挂；``off`` ⇒ 永远不挂（**举手也不挂**）。
3. 挂的图和元素表是**同一屏**（用 contextor 抓的那一帧，不重抓）。
4. 图块排在 prompt **最后**（它是最不可缓存的一块）。
5. Executor 举的手要**跨轮**传递（state 字段）—— 它决定的是下一轮。
6. 通道全关时**不阻断**，但要把「你看不见屏幕」如实告诉模型。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "minitap"))

import pytest  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from pytest import MonkeyPatch  # noqa: E402

from minitap.mobile_use.agents.executor.executor import ExecutorNode  # noqa: E402
from minitap.mobile_use.agents.executor.prompt import build_executor_messages  # noqa: E402
from minitap.mobile_use.agents.executor.types import ExecutorOutput  # noqa: E402
from minitap.mobile_use.config import LLM, LLMConfig, LLMConfigUtils, LLMWithFallback  # noqa: E402
from minitap.mobile_use.context import (  # noqa: E402
    DeviceContext,
    DevicePlatform,
    MobileUseContext,
    PerceptionConfig,
)
from minitap.mobile_use.graph.state import State  # noqa: E402

# 一张 1x1 的 PNG（base64）—— 只用来让压缩函数有真图可处理。
_TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="  # noqa: E501


def _ctx(perception: PerceptionConfig | None = None) -> MobileUseContext:
    def llm() -> LLMWithFallback:
        return LLMWithFallback(
            provider="openai", model="gpt-5", fallback=LLM(provider="openai", model="o4-mini")
        )

    return MobileUseContext(
        trace_id="test",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id="test",
            device_width=1220,
            device_height=2712,
        ),
        llm_config=LLMConfig(
            planner=llm(),
            orchestrator=llm(),
            contextor=llm(),
            cortex=llm(),
            executor=llm(),
            utils=LLMConfigUtils(outputter=llm(), hopper=llm()),
        ),
        perception=perception or PerceptionConfig(),
    )


def _state(**overrides) -> State:
    base = {
        "messages": [],
        "initial_goal": "read the wifi name",
        "subgoal_plan": [],
        "latest_ui_hierarchy": None,
        "latest_ocr_results": None,
        "latest_screen_elements": None,
        "latest_screenshot": None,
        "focused_app_info": None,
        "device_date": None,
        "structured_decisions": None,
        "complete_subgoals_by_ids": [],
        "agents_thoughts": [],
        "remaining_steps": 400,
        "executor_messages": [],
        "cortex_last_thought": None,
        "scratchpad": {},
    }
    base.update(overrides)
    return State(**base)


def _state_with_screenshot(**overrides) -> State:
    base = {"latest_screenshot": _TINY_PNG_B64, "executor_goal": "read the wifi name"}
    base.update(overrides)
    return _state(**base)


class TestScreenshotChannel:
    """通道三态 × Executor 举不举手。"""

    @pytest.mark.asyncio
    async def test_on_demand_needs_the_previous_round_to_have_asked(self):
        """默认档：Executor **上一轮**举过手，这一轮才有图。"""
        node = ExecutorNode(_ctx(PerceptionConfig(screenshot="on_demand")))

        assert await node._screenshot_message(state=_state_with_screenshot()) is None

        msg = await node._screenshot_message(
            state=_state_with_screenshot(executor_wants_screenshot=True)
        )
        assert msg is not None

    @pytest.mark.asyncio
    async def test_always_attaches_every_round(self):
        """``always``：不用举手，每轮都有图。"""
        node = ExecutorNode(_ctx(PerceptionConfig(screenshot="always")))
        msg = await node._screenshot_message(state=_state_with_screenshot())
        assert msg is not None

    @pytest.mark.asyncio
    async def test_off_never_attaches_even_when_asked(self):
        """``off``：通道关了，Executor 举手也不给。

        提示词里会写明「没有图」（见 test_screenshot_on_demand.py），
        所以模型不会一直要；但**执行层**必须独立守住这条 ——
        提示词和执行层各判各的，早晚会有一边不同步。
        """
        node = ExecutorNode(_ctx(PerceptionConfig(screenshot="off")))
        assert (
            await node._screenshot_message(
                state=_state_with_screenshot(executor_wants_screenshot=True)
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_no_capture_in_state_means_no_image(self):
        """手里没图就别硬凑（contextor 没抓到、或者这一轮没重抓）。"""
        node = ExecutorNode(_ctx(PerceptionConfig(screenshot="always")))
        assert await node._screenshot_message(state=_state()) is None

    @pytest.mark.asyncio
    async def test_image_message_carries_a_caption(self):
        """图必须带一句说明 —— 否则模型会猜它对应哪一屏。

        on_demand 档下，模型举手时看到的是**上一轮**的屏；它的动作执行后
        屏可能已经变了。不说明清楚，模型会把图和元素表对错号。
        """
        node = ExecutorNode(_ctx(PerceptionConfig(screenshot="on_demand")))
        msg = await node._screenshot_message(
            state=_state_with_screenshot(executor_wants_screenshot=True)
        )
        assert msg is not None
        content = msg.content
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"]
        assert content[1]["type"] == "image_url"


class TestPromptAssembly:
    """图块的位置 + OCR 开关。"""

    def _messages(self, *, perception: PerceptionConfig, screenshot=None, ocr_results=None):
        return build_executor_messages(
            ctx=_ctx(perception),
            goal="read the wifi name",
            screen_elements=None,
            failures=[],
            local_messages=[],
            device_state_lines=[],
            ocr_results=ocr_results,
            screenshot_message=screenshot,
        )

    def test_image_block_is_last(self):
        """图必须排在**最后** —— 它是 prompt 里最不可缓存的一块。

        放在任何每轮变的内容之前，它之后的全部字节每轮都要满价重算
        （前缀缓存只认连续前缀）。
        """
        shot = HumanMessage(content=[{"type": "image_url", "image_url": {"url": "data:x"}}])
        messages = self._messages(perception=PerceptionConfig(screenshot="always"), screenshot=shot)
        assert messages[-1] is shot

    def test_ocr_block_is_absent_when_the_channel_is_off(self):
        """OCR 关掉 ⇒ 不给 OCR 块（不是「关了还偷偷给」）。"""
        from minitap.mobile_use.services.ocr.types import OcrText
        from minitap.mobile_use.utils.ui_hierarchy import ElementBounds

        ocr = [
            OcrText(
                text="hello",
                score=0.9,
                bounds=ElementBounds(x=0, y=0, width=10, height=10),
            )
        ]
        on = self._messages(perception=PerceptionConfig(ocr=True), ocr_results=ocr)
        off = self._messages(perception=PerceptionConfig(ocr=False), ocr_results=ocr)
        assert any("hello" in str(m.content) for m in on)
        assert not any("hello" in str(m.content) for m in off)


class TestDecisionCrossesRounds:
    """Executor 举的手要跨轮传递（state 字段）。"""

    def test_state_field_exists_and_defaults_to_false(self):
        assert "executor_wants_screenshot" in State.model_fields
        assert State.model_fields["executor_wants_screenshot"].default is False

    @pytest.mark.asyncio
    async def test_the_hand_is_written_into_state(self, monkeypatch: MonkeyPatch):
        """模型说「我要图」⇒ 这一轮的 state 更新里必须带上这个决定。

        它决定的是**下一轮**的 prompt（Executor 节点每轮是新调用，
        节点实例上的字段活不过一轮 ⇒ 只能走 state）。
        """

        async def _decide(self, *, state, goal, screenshot_message=None):
            return ExecutorOutput(
                actions=[{"action": "tap", "target": {"index": 1}}],
                agent_thought="tap the row",
                need_screenshot=True,
            )

        monkeypatch.setattr(ExecutorNode, "_decide", _decide)

        node = ExecutorNode(_ctx(PerceptionConfig(screenshot="on_demand")))
        update = await node(_state_with_screenshot())
        assert update["executor_wants_screenshot"] is True

    @pytest.mark.asyncio
    async def test_not_raising_the_hand_clears_the_flag(self, monkeypatch: MonkeyPatch):
        """模型不再举手 ⇒ 下一轮就没有图（每轮被模型的新回答覆盖）。"""

        async def _decide(self, *, state, goal, screenshot_message=None):
            return ExecutorOutput(
                actions=[{"action": "tap", "target": {"index": 1}}],
                agent_thought="tap the row",
                need_screenshot=False,
            )

        monkeypatch.setattr(ExecutorNode, "_decide", _decide)

        node = ExecutorNode(_ctx(PerceptionConfig(screenshot="on_demand")))
        update = await node(_state_with_screenshot(executor_wants_screenshot=True))
        assert update["executor_wants_screenshot"] is False


class TestHoldTheActionUntilTheModelSeesThePicture:
    """举手要图 ⇒ **这一轮的动作不执行**，下一轮带图重来。

    需求：「决定权在 Executor 手中 —— UI+OCR 不够就让模型看图」。
    它此刻只有元素表 + OCR，给出的动作是**猜的**；猜错的那一下可能是
    「确认支付」。而循环本来就要继续下一轮 ⇒ 不额外烧 LLM 调用。
    """

    @staticmethod
    def _raising_hand():
        async def _decide(self, *, state, goal, screenshot_message=None):
            return ExecutorOutput(
                actions=[{"action": "tap", "target": {"index": 1}}],
                agent_thought="I cannot tell them apart",
                need_screenshot=True,
            )

        return _decide

    @staticmethod
    def _has_action(update: dict) -> bool:
        """这次更新里有没有**可执行**的动作（带 tool_calls 的 AIMessage）。"""
        return any(getattr(m, "tool_calls", None) for m in update.get("executor_messages", []))

    @pytest.mark.asyncio
    async def test_the_action_is_held_back(self, monkeypatch: MonkeyPatch):
        """举手 ⇒ 动作扣下 ⇒ 回灌一条 HumanMessage（graph 走 retry 边重来）。"""
        monkeypatch.setattr(ExecutorNode, "_decide", self._raising_hand())

        node = ExecutorNode(_ctx(PerceptionConfig(screenshot="on_demand")))
        update = await node(_state_with_screenshot())

        assert not self._has_action(update), "举了手就不该执行猜出来的动作"
        messages = update["executor_messages"]
        assert isinstance(messages[-1], HumanMessage)
        assert "screenshot" in str(messages[-1].content)
        assert update["executor_wants_screenshot"] is True

    @pytest.mark.asyncio
    async def test_it_acts_when_the_picture_did_not_help(self, monkeypatch: MonkeyPatch):
        """已经带过图了（上一轮举过手）⇒ 它还要图就是「图没帮上忙」⇒ 让它动手。

        不这么判就是无限循环：举手 → 等图 → 举手 → 等图 …
        """
        monkeypatch.setattr(ExecutorNode, "_decide", self._raising_hand())

        node = ExecutorNode(_ctx(PerceptionConfig(screenshot="on_demand")))
        update = await node(
            _state_with_screenshot(executor_wants_screenshot=True)  # 上一轮已经给过图
        )
        assert self._has_action(update), "看过图还举手，就该让它动手，不能一直等"

    @pytest.mark.asyncio
    async def test_it_does_not_hold_when_the_channel_is_off(self, monkeypatch: MonkeyPatch):
        """通道关着 ⇒ 等不到图 ⇒ 绝不能扣动作（否则整个子目标卡死）。"""
        monkeypatch.setattr(ExecutorNode, "_decide", self._raising_hand())

        node = ExecutorNode(_ctx(PerceptionConfig(screenshot="off")))
        update = await node(_state_with_screenshot())
        assert self._has_action(update)

    @pytest.mark.asyncio
    async def test_it_does_not_hold_in_always_mode(self, monkeypatch: MonkeyPatch):
        """always 档：图每轮都在，它没什么要等的。"""
        monkeypatch.setattr(ExecutorNode, "_decide", self._raising_hand())

        node = ExecutorNode(_ctx(PerceptionConfig(screenshot="always")))
        update = await node(_state_with_screenshot())
        assert self._has_action(update)

    @pytest.mark.asyncio
    async def test_it_does_not_hold_without_a_capture(self, monkeypatch: MonkeyPatch):
        """手里没有图 ⇒ 等不到 ⇒ 不能扣动作。"""
        monkeypatch.setattr(ExecutorNode, "_decide", self._raising_hand())

        node = ExecutorNode(_ctx(PerceptionConfig(screenshot="on_demand")))
        update = await node(_state(executor_goal="read the wifi name"))
        assert self._has_action(update)

    def test_the_graph_routes_the_held_action_back_to_the_executor(self):
        """扣下动作后，graph 必须**回 executor 重来**（不能进 reporter 收尾）。

        靠的是「最后一条消息是 HumanMessage」这条判据（和 dispatcher 拒了
        那条路同一个形状）。屏幕没变过 ⇒ 走 retry 边**不重新抓屏** ⇒
        元素表和图天然同屏。
        """
        from minitap.mobile_use.graph.graph import post_executor_gate

        state = _state(
            executor_messages=[
                AIMessage(content="I cannot tell them apart"),
                HumanMessage(content="Your previous action was NOT executed: screenshot..."),
            ]
        )
        assert post_executor_gate(state) == "retry"


class TestDegradeInsteadOfBlock:
    """通道全关 ⇒ 不阻断，但要把事实告诉模型。"""

    def test_all_channels_off_tells_the_model_it_is_blind(self):
        """三条通道全关 ⇒ 如实告诉模型「你没有屏幕描述」，但**不阻断**。"""
        node = ExecutorNode(_ctx(PerceptionConfig(ui_hierarchy=False, ocr=False, screenshot="off")))
        lines = node._perception_state_lines()
        assert lines
        assert "no description of the screen" in lines[0]

    def test_the_blind_notice_has_no_image_words(self):
        """看不到图的模型，**连这句告知里都不能提图**。"""
        node = ExecutorNode(_ctx(PerceptionConfig(ui_hierarchy=False, ocr=False, screenshot="off")))
        text = " ".join(node._perception_state_lines()).lower()
        for word in ("screenshot", "picture", "image"):
            assert word not in text, f"不该出现「{word}」"

    def test_only_the_picture_channel_left_mentions_asking_for_it(self):
        node = ExecutorNode(
            _ctx(PerceptionConfig(ui_hierarchy=False, ocr=False, screenshot="on_demand"))
        )
        lines = node._perception_state_lines()
        assert lines
        assert "need_screenshot" in lines[0]

    def test_text_channels_on_says_nothing(self):
        """还有一条文本通道就不用啰嗦 —— 设备状态行也是每轮变的内容。"""
        node = ExecutorNode(_ctx(PerceptionConfig()))
        assert node._perception_state_lines() == []
