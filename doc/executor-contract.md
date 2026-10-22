# Executor 自主执行 —— 实现契约（v2）

> 这是**所有并行实现任务的唯一接口依据**。每个子代理只改自己名下的文件，
> 不许改别人名下的文件。所有签名以本文档为准，**不要自己发明**。

**方案全文**：`doc/executor-autonomy-plan.md`（含 4 处审计修正后的最终形态）
**仓库**：`E:\MyCoding\mobile-use`
**Python 环境**：`uv run python ...`（项目根目录执行）

---

## 零、已经定好的东西（主控已完成，**不要重复实现**）

### 0.1 `minitap/mobile_use/agents/executor/types.py`（已建好，可直接用）

```python
class ExecutorOutput(BaseModel):
    actions: list[dict] = []          # 每个 {"action": "<工具名>", ...参数}
    agent_thought: str = ""           # ≤1 句，为什么是这个动作
    done: bool = False                # 目标达成
    done_reason: str | None = None
    need_screenshot: bool = False     # 元素表+OCR 不够，要图

class ExecutorReport(BaseModel):
    outcome: Literal["completed", "failed", "blocked"]
    summary: str = ""                 # ← 只有这个字段由 LLM 写
    evidence: str | None = None
    actions_taken: list[str] = []
    failed_targets: list[FailedTarget] = []
    blocked_reason: str | None = None

def determine_outcome(*, done: bool, failures: list[dict] | None = None,
                      step_limit_reached: bool = False) -> Literal[...]
    # done → completed；步数超限 → blocked；有失败 → failed；其余 → failed
```

### 0.2 `minitap/mobile_use/graph/state.py`（已加好字段）

| 字段 | 类型 | reducer | 谁写 | 含义 |
|---|---|---|---|---|
| `executor_goal` | `str \| None` = None | take_last | Cortex | 本子目标的目标陈述 |
| `executor_done` | `bool` = False | take_last | Executor | 本轮是否认为达成 |
| `executor_steps` | `int` = 0 | take_last | Executor | 本子目标已执行步数 |
| `executor_failures` | `list[dict]` = [] | take_last | 工具层 | **append-only** 失败事实 |
| `executor_report` | `ExecutorReport \| None` = None | take_last | Reporter | 上次汇报 |
| `executor_messages` | `list[AnyMessage]` | add_messages | Executor + 工具 | **本子目标内的局部历史** |

> ⚠️ `executor_messages` 的 key **不变**（`EXECUTOR_MESSAGES_KEY = "executor_messages"`），
> 工具层写它的代码**不用改**。只是清空时机变了：
> **Cortex 下发新子目标时清空**（`RemoveMessage(id=REMOVE_ALL_MESSAGES)`），
> 不再是「每轮都清空」。

### 0.3 新的 graph 拓扑（**W8 实现，其他人不改 graph.py**）

```
START → planner → orchestrator(初始化) → convergence ──continue──> cortex
                                                               │
                                                    cortex 下发目标│
                                                               ↓
                        ┌───────────── 主循环（graph 级环）──────────────┐
                        │  contextor → executor → executor_tools        │
                        │      ↑                        │                │
                        │      └────────────────────────┘                │
                        └────────────────────────────────────────────────┘
                                       │ executor 判定 done
                                       ↓
                              reporter → {cortex, orchestrator(看门狗)}
```

**关键**：`contextor` 在环里，每轮抓屏并**重建屏幕元素表**（见 §1.2），
所以工具读到的 `state.latest_screen_elements` **永远是当屏的**（索引契约成立）。

---

## 一、文件所有权（**严格不越界**）

| 子代理 | 名下文件（**只有这些**） |
|---|---|
| **W1 Prompt** | `minitap/mobile_use/agents/executor/executor.md`（重写）<br>`minitap/mobile_use/agents/reporter/reporter.md`（新建）<br>`scripts/check_prompt_templates.py`<br>`tests/test_prompt_templates.py` |
| **W2 配置** | `minitap/mobile_use/config.py`<br>`llm-config.defaults.jsonc`<br>`llm-config.override.jsonc`<br>`minitap/mobile_use/sdk/utils.py` |
| **W3 观测** | `minitap/mobile_use/services/telemetry.py`<br>`minitap/mobile_use/tui/events.py` |
| **W4 感知** | `minitap/mobile_use/agents/contextor/contextor.py` |
| **W5 Executor** | `minitap/mobile_use/agents/executor/executor.py`（重写）<br>`minitap/mobile_use/agents/executor/prompt.py`（新建） |
| **W6 Cortex** | `minitap/mobile_use/agents/cortex/cortex.py`<br>`minitap/mobile_use/agents/cortex/types.py` |
| **W7 工具层** | `minitap/mobile_use/utils/failed_targets.py`<br>`minitap/mobile_use/tools/mobile/tap.py`<br>`minitap/mobile_use/tools/mobile/long_press_on.py`<br>`minitap/mobile_use/tools/mobile/focus_and_input_text.py`<br>`minitap/mobile_use/tools/mobile/focus_and_clear_text.py` |
| **W8 graph** | `minitap/mobile_use/graph/graph.py` |
| **W9 测试** | `tests/` 下新建的测试文件（只新建，不改别人已有的） |

**冲突判定**：任何两个子代理的文件集合不得相交。上表已保证。

---

## 二、各任务的详细要求

### W1 — Prompt 层

#### W1.1 重写 `minitap/mobile_use/agents/executor/executor.md`

它是 **Jinja2 模板**（和 `cortex.md` 一样），变量**只有两个**：

| 变量 | 类型 | 渲染成 |
|---|---|---|
| `platform` | `str` | `"android"` / `"ios"` |
| `executor_tools_list` | `str` | 逗号分隔的**已排序**工具名（`format_tools_list` 的输出） |

内容必须覆盖（**顺序即缓存友好度，见 §三**）：

1. **身份**：Executor 有**一个**目标（会被渲染进来），自己是眼睛+手。
2. **目标陈述占位**：`Your goal: {{ goal }}` —— ⚠️ `goal` **不是**模板变量，
   是**每轮动态注入**的（见 §2 W5.1 的 prompt 组装顺序）。
   → **实现方式**：`executor.md` 里写 `Your goal: {{ goal }}`，
   W5 在渲染时传 `goal=<state.executor_goal>`。
   **所以模板变量实际是三个：`platform` / `executor_tools_list` / `goal`。**
3. **Your Loop**：看元素表 → 决定一个动作 → 重复。
4. **Rules**（照抄 `cortex.md` 里**必须保留**的语义）：
   - 一次只做一个动作
   - **不要重复失败的目标**（失败清单会单独注入）
   - 每次都信**当前的**元素表，不要信记忆里的旧屏
   - 目标达成就 `done: true`
   - 走不动就**如实说**，不要编成功
   - **子目标锚点检查**：「这一步还在完成本次目标吗？偏离了就停下来说明」
5. **元素索引说明**：照搬 `cortex.md:46-88`（`## Element targeting`）的核心语义
   —— `[i]` 是 1-based 行号，**直接用**，不要抄 id 和坐标。
   （这节 574 tok，**不能删**——它是索引契约的说明书。）
6. **Tools 文本工具表**：从 `cortex.md:90-120` 搬过来（`## Tools` 节），
   包括 `{{ executor_tools_list }}` 那行。
7. **感知通道**：`need_screenshot` 什么时候置 true；屏幕状态行（息屏提示）怎么理解。
8. **Output**：字段表 —— `actions` / `agent_thought` / `done` / `done_reason` / `need_screenshot`。

> ⚠️ **不要**在 executor.md 里出现 `bind_tools` / JSON schema 相关的任何话。
> **不要**写「所有工具的完整参数定义」——那正是要避免的 4,008 token。

#### W1.2 新建 `minitap/mobile_use/agents/reporter/reporter.md`

- 模板变量：**无**（完全静态）
- 内容：Reporter 只做压缩，**不判断**（`outcome` 由代码给）；
  只写 `summary`（1~2 句）和挑 `evidence`；事实字段由代码填。
- 明确写「不要复述每一步」「不要编造日志里没有的东西」。

#### W1.3 `scripts/check_prompt_templates.py`

该脚本有**自覆盖守卫**（`TEMPLATE_VARS` 登记 + 磁盘扫描比对）。
必须把新文件登记进去：

```python
"agents/executor/executor.md": {...},   # 变量: platform, executor_tools_list, goal
"agents/reporter/reporter.md": {...},   # 无变量
```

同时**保留** `agents/executor/executor.md` 原有的登记项（它在列表里，只是变量表变了）。

#### W1.4 `tests/test_prompt_templates.py`

同步更新期望值。**不许**为了让测试变绿而删断言。

---

### W2 — 配置层

#### W2.1 `config.py`

```python
AgentNode = Literal[
    "planner", "orchestrator", "contextor", "cortex", "executor",
    "reporter",          # 新增
]

class LLMConfig(BaseModel):
    ...
    reporter: LLMWithFallback | None = None    # ← 可选！为 None 时回落 executor
```

- **`get_agent()` 里加回落**：`reporter` 为 None 时返回 `executor` 的配置。
- `validate_providers()`：`reporter` 有值才校验（None 跳过）。
- `__str__()`：加一行显示 reporter（None 时显示「回落 executor」）。
- **不要**把 `reporter` 做成必填 —— 实测有 22 处 `LLMConfig(...)` 构造点。

#### W2.2 `llm-config.defaults.jsonc`

**5 个 preset** 都要处理：`default` / `anthropic` / `minimax` / `deepseek` / `recommended`。
`reporter` **可以不写**（走回落），但**要能写**。

建议值（按方案 §2.1.1 的模型分层）：

| 角色 | 模型 | 理由 |
|---|---|---|
| executor | 快模型（现状 `gpt-5-nano` 级别） | 每轮 1 次，延迟叠加 |
| reporter | 快模型（回落 executor 即可） | 只写摘要 + 挑证据 |
| cortex | 强模型（现状 `gpt-5`） | 真正的「一次定生死」 |
| orchestrator | **建议升到强模型** | 它现在是看门狗，判「计划还走得通吗」是高层判断 |

#### W2.3 `llm-config.override.jsonc`

本机实际生效的那份。加 `reporter`（可省略）。

#### W2.4 `sdk/utils.py`

`ALL_ROLE_NAMES`（约 :29-34）加 `"reporter"`。
检查 `llm_config_for_role_overrides` / `llm_config_for_single_model`
是否需要同步加 reporter 分支。

---

### W3 — 可观测性

#### W3.1 `services/telemetry.py`

新增两个角色的 usage 归因 + 三个新指标：

| 事件 | 内容 |
|---|---|
| `capture_executor_step` | `task_id` / `step_index` / `actions_count` / `done` / `input_tokens` / `output_tokens` |
| `capture_reporter_report` | `task_id` / `outcome` / `summary_len` / `has_evidence` / `failed_count` |
| `capture_subgoal_progress` | `task_id` / `subgoal_id` / `steps` / `outcome` / `retry_count` |

另外：
- `capture_cortex_decision` 的 `has_decisions` / `has_goals_completion`
  在新架构下**恒为 False**（字段已删）→ **要么改造这个事件，要么标注它已废弃**。
  不许留下「看着有数其实是常数」的假指标。
- `capture_executor_dispatch` 的 `dispatched` 在新架构下**恒为 True**
  （Executor 现在**总是**走确定性直连）→ 同上。

> **不要**把 `services/usage_bridge.py` 造出来 —— 审计发现它**被引用但不存在**。
>
> ✅ **2026-10-21 实测更新：这个「洞」是假的结论。** 那句「图里的 usage 由
> usage_bridge 上报」从来不成立（该文件 git 全历史都没有），但**埋点本身是通的**：
> 真机 graph 实测（`probe_out/executor_e2e.json`）executor 21 次调用 /
> 120,772 in / 65,672 cached / 18,114 out 全是真实非零值。捕获槽装的是同一个
> list 对象，跨图可见；当初「每行都是 0」的真因是客户端没开 `stream_usage=True`。
> ⇒ 结论不变（**不要造这个文件**），但理由从「补断路」变成「别造多余的东西」。

#### W3.2 `tui/events.py`

- `AGENT_LABELS`（约 :41-49）加 `"executor"` / `"reporter"`。
  现在的硬编码映射 + `if node not in AGENT_LABELS: return`（约 :228-229）
  会让新节点**在 TUI 里完全不出现**。
- `ACTION_NODE = "executor_tools"`（约 :52）—— 工具执行仍然经过
  `executor_tools` 节点，**这个不用改**（W8 会保留 `executor_tools` 节点）。
- 加一个 `reporter` 的显示标签（绿色的汇报节点，用户要能看见）。

---

### W4 — 感知搬迁（Contextor）

**只改 `minitap/mobile_use/agents/contextor/contextor.py`。**

#### W4.1 在 Contextor 里建屏幕元素表

现在元素表在 `cortex.py:336-341` 构建（`build_screen_elements` + `render_screen_elements`）。
**搬到这里**：contextor 拿到 `device_data.elements`（层级树）后，
用它 + OCR 结果 + `focused_app_info` 建表，写进 state。

参考现有实现（`cortex.py:319-341`）：

```python
from minitap.mobile_use.utils.screen_elements import build_screen_elements
from minitap.mobile_use.utils.ui_hierarchy_view import normalize_hierarchy

tree_rows = None
if self.ctx.perception.ui_hierarchy and device_data.elements:
    tree_rows = normalize_hierarchy(device_data.elements)

screen_elements = build_screen_elements(
    tree_rows=tree_rows,
    ocr_results=(ocr_results if self.ctx.perception.ocr else None),
    foreground_package=current_app_package,
)
```

**兼容 iOS / 云设备**：`tree_rows is None` 时（层级树形状不认识），
元素表建不出来 → `screen_elements` 留空列表，工具侧走
`latest_ocr_results` 的老路径（`tools/utils.py:384-398` 已有这个分支，**不用改**）。

#### W4.2 `latest_ui_hierarchy` 的清理责任

现在 cortex 在返回时把 `latest_ui_hierarchy` 置 None（`cortex.py:421`），
所以工具侧读它**永远是空的**（`tools/utils.py:178,197` 有注释说明）。

**保持这个行为不变**：contextor 建完表后，在同一个 update 里
把 `latest_ui_hierarchy` 置 None。**否则工具侧的 id/文字查找分支会突然复活**，
那是行为变更，不在本次范围内。

#### W4.3 写进 state 的字段

```python
update={
    "latest_screen_elements": screen_elements,   # 新增
    "latest_ui_hierarchy": None,                 # 建完表就清空（保持工具侧行为）
    "latest_ocr_results": ocr_results,
    "latest_screenshot": device_data.base64,
    "focused_app_info": current_app_package,
    "screen_size": (device_data.width, device_data.height),
    "device_date": device_date,
    "screen_state": screen_state,
    "agents_thoughts": [agent_outcome],
}
```

> ⚠️ `latest_screen_elements` 和 `latest_ocr_results` **不能被清空**
> —— 它们是**索引契约**：工具要把 `Target.index` 还原成当屏那一行的坐标。
> 下一个 contextor 会整体覆盖它们。

---

### W5 — Executor 节点

**改 `minitap/mobile_use/agents/executor/executor.py`（重写），新建
`minitap/mobile_use/agents/executor/prompt.py`。**

#### W5.1 `prompt.py` — prompt 组装

**消息顺序 = 缓存友好度**（方案 §4.1.1，这是本次改造的核心机制之一）。

**最终顺序（以这个为准）：**

```
[1] SystemMessage   executor.md 渲染结果（静态，含工具表）
[2] HumanMessage    目标陈述（子目标内恒定）
[3] HumanMessage    失败清单（executor_failures 渲染，只增不减 → 前缀稳定）
[4] HumanMessage    设备状态（息屏 / 前台 app / 日期）
[5] ...             局部历史（executor_messages 里的 AIMessage/ToolMessage）
[6] HumanMessage    屏幕元素表（每轮变）        ← **必须在最后**
[7] HumanMessage    截图（只在 need_screenshot 时追加）
```

> 🔴 **[6] 必须放在 [5] 之后**（即元素表在局部历史**后面**）。
> 元素表每轮变，放中间会把它之后的全部历史每轮重算 → 前缀缓存命中率**递减**。
> 正确布局实测 79.9%（递增），错误布局 53.8%（递减，低于现状 74.6%）。
>
> 换句话说：**每轮变的东西一律放末尾。**

需要的函数签名：

```python
def build_executor_messages(
    *,
    ctx: MobileUseContext,
    goal: str,
    screen_elements: list[ScreenElement] | None,
    failures: list[dict],
    local_messages: list[AnyMessage],
    device_state_lines: list[str],
    screenshot_message: AnyMessage | None = None,
) -> list[BaseMessage]:
    ...
```

#### W5.2 `executor.py` — 节点逻辑

```
ExecutorNode.__call__(state):
  1. goal = state.executor_goal
     没有 goal → 直接返回一个 AIMessage（无 tool_calls）+ executor_done=False
        （正常情况下不该发生；graph 会保证有 goal 才进 executor）
  2. 组装 messages（见 W5.1）
  3. llm = get_llm(ctx, name="executor")
     用 _ToolCallStructuredRunnable(llm, ExecutorOutput) 拿结构化输出
     （**不要** bind_tools 那 16 个设备工具！详见 services/llm.py:529）
  4. response: ExecutorOutput
  5. if response.need_screenshot and 没截过图:
        抓图 → 追加截图消息 → **同一轮再问一次**（照搬 cortex.py:392-398 的做法）
  6. 动作转换：
        decisions_json = json.dumps(response.actions, ensure_ascii=False)
        tool_calls = dispatch_decisions(decisions_json, default_thought=response.agent_thought, tools=tools)
  7. if tool_calls is None:
        → **把 dispatcher 的拒绝理由回灌给 Executor**（作为下一轮的 ToolMessage），
          **不要**退回 LLM 重来（那会死循环，见方案 §3.1.1 坑 2）
        → 返回 AIMessage 无 tool_calls + executor_done=False
     else:
        → 返回 AIMessage(content="", tool_calls=tool_calls)
  8. 写 state：
        executor_messages: [AIMessage(...)]        # 追加自己的回合
        executor_done: response.done
        executor_steps: state.executor_steps + 1
```

**步数上限**：`MAX_STEPS_PER_SUBGOAL`（建议常量放 `constants.py`？**不行**，`constants.py`
归 W7。**放在 `executor.py` 里**，或 `agents/executor/types.py`——后者已被主控建好，
**追加常量到 types.py 属于越界**（主控的文件）。→ **放在 `executor.py` 里**。

超限时：写 `executor_done = True` + 一个说明，让 Reporter 判 `blocked`。

> `dispatch_decisions` 的拒绝理由怎么拿到：它 `logger.info` 打出来了但**没返回**。
> 你需要**改 `dispatcher.py` 让它把拒绝理由返回出来**。
> ⚠️ **`dispatcher.py` 归主控**——**不要改**。改用这个办法：
> 自己在 `executor.py` 里先做一次**轻量校验**（检查 action 名在不在 tools 里、
> 必填参数齐不齐），不通过就自己生成拒绝理由回灌。
> 或者：调用 `dispatch_decisions`，若返回 None 就用**通用理由**回灌
> （"the previous action was rejected; it must be a valid tool name with all
> required arguments — see the Tools table"）。
> **后者更简单，推荐。**

#### W5.3 局部历史的读写

- **读**：`state.executor_messages`（含 Executor 自己的 AIMessage + 工具的 ToolMessage）
- **写**：在 update 里追加自己的 AIMessage

**不要**在 Executor 侧清空 `executor_messages` —— **清空责任在 Cortex 下发新目标时**
（W6 负责）。

---

### W6 — Cortex 改造

**改 `minitap/mobile_use/agents/cortex/cortex.py` 和 `agents/cortex/types.py`。**

#### W6.1 新的 `CortexOutput`（`cortex/types.py`）

```python
class CortexOutput(BaseModel):
    next_subgoal: str | None = None        # 下一个子目标的目标陈述（自然语言）
    subgoal_completed: bool = False        # 上一个子目标是否达成
    completion_reason: str | None = None   # 为什么
    # 注意：**不要** needs_replan（replan 权只在 Orchestrator，方案 §6.5.1）
    # 注意：**不要** need_screenshot（Cortex 不看屏了，感知通道下沉到 Executor）
```

#### W6.2 Cortex 的 prompt 瘦身

**删掉**（这些都下沉给 Executor 了）：
- 元素表渲染规则、`## Element targeting`、工具参数表、perception 三通道说明
- 截图相关的全部内容
- `decisions_reason` / `goals_completion_reason` 两个字段
- 感知块（元素表 / OCR / 截图）—— **Cortex 不再看屏**

**保留**：
- 计划日志（`subgoal_plan_log`）
- 子目标状态（`render_subgoal_status`）
- Executor 的汇报（新增块）
- 知识库接口（先留占位，不实现）

#### W6.3 消息块顺序（缓存友好）

```
[1] SystemMessage   cortex.md（静态）
[2] HumanMessage    设备信息 + 目标（任务内恒定）
[3] HumanMessage    thoughts 历史（只追加）
[4] HumanMessage    计划日志（只追加）
[5] HumanMessage    子目标状态（固定槽位原地替换）
[6] HumanMessage    失败目标清单
[7] HumanMessage    Executor 汇报（每子目标变）   ← 新增
[8] HumanMessage    Executor 的失败事实（跨子目标）  ← 新增
```

汇报块是「每轮变」，必须排在 thoughts 之后。

#### W6.4 Cortex 的新职责

```
收到 Executor 汇报（state.executor_report）：
  - outcome == completed → subgoal_completed=True，写 completion_reason
  - outcome == failed/blocked → subgoal_completed=False
      → 决定「换招重发」还是「标记该子目标失败」
      → **默认换招重发**：写一个新的 next_subgoal（换个思路）
      → 换招时**清空 executor_messages** + 重置 executor_steps
  - 下发下一个子目标 → next_subgoal=<自然语言>

**不**产出 needs_replan（replan 权在 Orchestrator）。
```

**清空局部历史**：下发新目标时
`EXECUTOR_MESSAGES_KEY: [RemoveMessage(id=REMOVE_ALL_MESSAGES)]`
+ `executor_steps: 0`。

> ⚠️ `RemoveMessage` 需要 `from langchain_core.messages import RemoveMessage`
> 和 `from langgraph.graph.message import REMOVE_ALL_MESSAGES`。

#### W6.5 删除旧的感知代码

- `cortex.py:319-343` 的元素表构建 → **删掉**（搬到 W4 的 contextor）
- `cortex.py:387-398` 的 need_screenshot 二问 → **删掉**（下沉到 Executor）
- `cortex.py:296-313` 的设备状态块 → **删掉**（Executor 自己看）
- `cortex.py:453-477` `get_executor_agent_feedback` → **重写**成读 `state.executor_report`

#### W6.6 保留什么

- `structured_decisions` 字段**废弃**（不再由 Cortex 产出）。
  **但 state 里先留着**（W8 会清理）—— 不要删 state 字段。
- `cortex_last_thought` 同样先留着。

---

### W7 — 工具层

#### W7.1 `utils/failed_targets.py` — 加 append-only 辅助

```python
def append_failure(streak: list[dict] | None, *, key: str, index: int | None,
                   resource_id: str | None, label: str | None,
                   reason: str) -> list[dict]:
    """append-only 的失败事实。**不**清空、**不**换键替换。"""
    items = list(streak or [])
    items.append({"key": key, "index": index, "resource_id": resource_id,
                  "label": label or "", "reason": reason})
    return items
```

**不要**改 `record_failure` / `breaker_reason` / `render_failures` 的现有语义
（熔断器靠它们，`tests/mobile_use/test_failed_targets.py` 守着）。

#### W7.2 四个工具文件

每个工具在**失败分支**里，除了写 `failed_targets`（熔断账本，语义不变），
**额外**写 `executor_failures`（append-only 事实账本）：

```python
# 失败分支
failed_targets = record_target_failure(state=state, target=target, reason=...)
executor_failures = append_failure(
    state.executor_failures,
    key=target_key(target),
    index=target.index,
    resource_id=target.resource_id,
    label=target.text,
    reason=failure_details[:200],
)

# 成功分支
failed_targets: list[FailedTarget] = []          # 熔断账本：成功即清（不变）
executor_failures = state.executor_failures      # 事实账本：**不清**
```

update 里加一项 `"executor_failures": executor_failures`。

> ⚠️ **成功时 `executor_failures` 不清空** —— 这正是它和 `failed_targets` 的区别。
> 事实要跨子目标保留。

四个文件：
- `tools/mobile/tap.py`（成功分支在 :227 附近）
- `tools/mobile/long_press_on.py`
- `tools/mobile/focus_and_input_text.py`
- `tools/mobile/focus_and_clear_text.py`

---

### W8 — Graph 接入

**只改 `minitap/mobile_use/graph/graph.py`。**

#### W8.1 节点

- 保留：`planner` / `orchestrator` / `contextor` / `cortex` / `executor_tools` / `summarizer` / `convergence`
- 新增：`reporter`（`agents/reporter/reporter.py` —— **W8 自己建这个文件**，
  它不在别人的所有权列表里；`agents/reporter/` 目录也要 W8 建）
- `executor` 节点**保留同名**（`ExecutorNode`），但语义变成「决策一步」

> ⚠️ `agents/reporter/reporter.py` 不在任何人的名下 → **归 W8**。
> W1 只写 `agents/reporter/reporter.md`（prompt），W8 写 `agents/reporter/reporter.py`（节点）。

#### W8.2 边

```
START → planner → orchestrator → convergence
convergence --continue--> cortex
convergence --replan--> planner
convergence --end--> END

cortex --(下发目标)--> contextor
contextor --> executor
executor --(有动作)--> executor_tools
executor --(done)--> reporter
executor_tools --> contextor            # 回到环里重新抓屏
reporter --> cortex                     # Cortex 判断
reporter --> orchestrator               # 看门狗（每子目标无条件）
```

条件边用**函数**实现（和现在的 `post_cortex_gate` 同款）：

```python
def post_executor_gate(state) -> Literal["invoke_tools", "report"]:
    if state.executor_done:
        return "report"
    # 有没有待执行的动作？—— 从 executor_messages 最后一条 AIMessage 的 tool_calls 判断
    ...
    return "report"   # 兜底：没有动作 → 去 reporter 收尾
```

`reporter → orchestrator` 这条**无条件**边是**审计修正的关键**
（方案 §6.5.1）——没有它，计划级失败就没人处理。

#### W8.3 Reporter 节点

`agents/reporter/reporter.py`：

```python
class ReporterNode:
    async def __call__(self, state: State):
        # 1. outcome = determine_outcome(done=state.executor_done,
        #                               failures=state.executor_failures,
        #                               step_limit_reached=...)
        # 2. evidence / actions_taken / failed_targets —— 代码从 state 里搬
        # 3. summary —— LLM 写（用 reporter.md，只写 summary + 挑 evidence）
        #    失败时的 summary 也要写（模型要说清楚卡在哪）
        # 4. 返回 update: {"executor_report": report}
```

LLM 调用：`get_llm(ctx, name="reporter")` + `_ToolCallStructuredRunnable(llm, 某个 schema)`。
schema 要**只有** `summary` 和 `evidence` 两个字段：

```python
class ReporterSummary(BaseModel):
    summary: str
    evidence: str | None = None
```

（这个 schema 可以放在 `agents/reporter/reporter.py` 里，W8 自己定。）

#### W8.4 `summarizer`

新拓扑里**保留** `summarizer`（裁剪 `state.messages`），位置在
`executor_tools` 之后。**或者**直接去掉——如果去掉，`MAX_MESSAGES_IN_HISTORY`
的裁剪就没了，`state.messages` 会涨。**建议保留**。

#### W8.5 清理

- `post_cortex_gate` 重写：现在 Cortex 的输出是 `next_subgoal`，
  触发条件是「有 next_subgoal」或「有汇报要处理」。
- `structured_decisions` / `cortex_last_thought` 在 state 里**先留着**
  （W6 不删 state 字段），但 graph 里不再用。

---

### W9 — 测试

**只在 `tests/` 下新建文件**，不改别人已有的测试（除了 W1 名下的 `test_prompt_templates.py`）。

新建 `tests/mobile_use/test_executor_autonomy.py`，覆盖：

| # | 测试 | 断言 |
|---|---|---|
| T1 | Executor 请求体 | **不含** 16 个设备工具的 schema（断言 `_get_request_payload` 的 `tools` 字段） |
| T2 | 局部历史生命周期 | Cortex 下发新目标时 `executor_messages` 被清空 |
| T3 | 汇报格式 | `outcome` 由 `determine_outcome` 决定；`failed_targets` 与 `executor_failures` 逐字相等 |
| T4 | 熔断 | 同目标失败 2 次 → 工具拒绝（**已有测试覆盖**，可引用）；步数超限 → `executor_done=True` |
| T5 | 失败事实跨子目标 | 子目标 A 的 `executor_failures` 出现在子目标 B 的 Executor prompt 里 |
| T7 | outcome 判定规则 | `determine_outcome` 的 4 个分支 |
| T8 | 汇报后不跳 Planner | 子目标失败 → 不设 FAILURE 状态 |
| T8' | **计划级失败会被处理** | 子目标连续失败 2 次后，**必须**走 replan 或 Orchestrator 复核，不能无限重发 |
| T10 | dispatcher 拒绝理由回灌 | Executor 输出未知工具名 → 拒绝理由出现在下一轮 prompt 里 |
| T11 | dispatcher 只吃数组 | `actions` 恒为 list；单元素数组也能直连 |
| **T12** | **元素表在 prompt 末尾** | 断言 Executor prompt 里元素表出现在局部历史**之后** |
| **T13** | **感知通道没丢** | ExecutorOutput 有 `need_screenshot`；`executor.md` 含截图说明 + 设备状态 |

---

## 三、通用约束（所有子代理）

1. **不许改自己名下以外的文件。** 需要改别人文件 → 在报告里写清楚要改什么。
2. **不许删测试来让 CI 变绿。** 测试失败 → 改实现，或在报告里说明为什么该测试过时。
3. **不许改 `dispatcher.py`**（主控的文件）。
4. **不许动 `graph/state.py` 的既有字段**（`structured_decisions` /
   `cortex_last_thought` 先留着）。
5. 改完**必须自己跑**：
   ```powershell
   cd E:\MyCoding\mobile-use
   uv run pytest tests/ -x -q          # 全量
   uv run ruff check <你改的文件>      # lint
   uv run pyright <你改的文件>          # 类型（如果有 pyright）
   ```
   报告里写**实际输出**（不是「应该能过」）。
6. **注释和文档用中文**，风格对齐现有代码（现有代码的中文注释会解释「为什么」，
   不是「是什么」）。
7. 每个子代理的最终回复必须包含：
   - 改了哪些文件、每个文件改了什么
   - **实际跑出来的**测试/lint/类型检查输出
   - 你没做但应该做的事（如果被文件所有权挡住）
   - 你发现但没权限改的问题

---

## 四、给子代理的原始需求（用于生成 prompt）

```
你在实现 mobile-use 的「Executor 自主执行」改造。

先读 E:\MyCoding\mobile-use\doc\executor-contract.md —— 那是唯一的接口依据，
里面有：已定好的数据模型、State 字段、你名下的文件、你的任务细节、通用约束。
再按需读 E:\MyCoding\mobile-use\doc\executor-autonomy-plan.md 的相关章节
（方案全文，含审计修正）。

严格只改契约里「你名下的文件」。

做完后跑测试和 lint，把真实输出贴出来。不要编造通过结果。
```
