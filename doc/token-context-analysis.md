# mobile-use Token 上下文构成与成本优化调研

> 调研日期：2026-10-16
> 调研范围：`minitap/mobile_use/` 全链路（planner / orchestrator / contextor / cortex / executor / summarizer / outputter）
> 方法：**离线复现各 agent 真实发送的 message list，用 tiktoken（`cl100k_base`）实测计数**，工具 schema 取自真实 `get_tools_from_wrappers()` 产物。所有数字均为实测值，非估算。
> 复现脚本：`v/probe_token_budget.py`、`v/probe_thought_growth.py`、`v/probe_minor_nodes.py`

---

## 一、调用拓扑

```
START → planner → orchestrator → convergence ┐
                                            │
        ┌───────────────────────────────────┘
        ↓
   contextor → cortex → executor → executor_tools → summarizer → convergence
        ↑                                                        │
        └──────────────── continue / replan ←──────────────────┘
                                                              └→ END → outputter
```

一次完整循环产生 **3~4 次 LLM 调用**：

| 节点 | 调用频率 | 是否带截图 | 是否带工具 schema |
|---|---|---|---|
| `planner` | 初始 + 每次 replan | 否 | 否（仅工具名列表） |
| `orchestrator` | 每次 cortex 标记 subgoal 完成时 | 否 | 否 |
| `contextor` | 仅 app-lock 校验时（冷门分支） | 否 | 否 |
| `cortex` | **每轮循环必跑** | **是** | 否（仅工具名列表） |
| `executor` | **每个 tool-call 轮** | 否 | **是（全量 16 个）** |
| `outputter` | 任务结束一次 | 否 | 否 |

---

## 二、Token 构成明细（实测）

### 2.1 Cortex —— 每轮循环跑一次

| 组成 | Token | 占本节点比 |
|---|---:|---:|
| system prompt（`cortex.md` 渲染后） | 1,747 | 36.4% |
| UI 层级树（90 节点，裁剪后） | 2,527 | 52.6% |
| 全量 `agents_thoughts`（12 条） | 300 | 6.2% |
| 结构化输出 schema（`CortexOutput`） | 196 | 4.1% |
| device info | 32 | 0.7% |
| **小计（不含截图）** | **4,802** | 100% |
| 截图（1080×2400 原图） | ~2,064 | — |
| **合计** | **~6,866** | — |

> UI 层级树此前已做过裁剪优化（原始 JSON dump 约 14k~24k tokens → 1.2k~2.5k）。
> 本次调研在此基础上继续向下分析，不重复该部分。

### 2.2 Executor —— 每个 tool-call 轮跑一次

| 组成 | 1 轮 | 4 轮 | 8 轮 |
|---|---:|---:|---:|
| **`bind_tools` schema（16 个工具）** | **3,560** | **3,560** | **3,560** |
| system prompt（`executor.md`） | 336 | 336 | 336 |
| `executor_messages`（累积） | 199 | 790 | 1,578 |
| `structured_decisions` | 59 | 59 | 59 |
| `cortex_last_thought` | 10 | 10 | 10 |
| **合计** | **4,164** | **4,755** | **5,543** |

**关键点：`bind_tools` 的 3,560 token 在每一轮都全额重发**，且随 `executor_messages` 累积，每轮总成本单调上升。

### 2.3 Orchestrator

| 组成 | Token |
|---|---:|
| system prompt（`orchestrator.md`） | 197 |
| `human.md`（goal + plan + 全量 thoughts ×60） | 1,556 |
| **合计** | **1,753** |

### 2.4 Planner

| 组成 | Token |
|---|---:|
| system prompt（`planner.md`） | 748 |
| `human.md`（goal + plan + 全量 thoughts） | 467 |
| 结构化输出 schema（`PlannerOutput`） | 80 |
| **合计** | **1,295** |

> 仅 replan 时触发，故 `human.md` 中 thoughts 部分有 `{% if action == "replan" %}` 保护，plan 阶段不计入。

### 2.5 Outputter

| 组成 | Token |
|---|---:|
| system prompt（硬编码） | 19 |
| `human.md`（goal + 全量 thoughts ×60） | 1,422 |
| **合计** | **1,441** |

### 2.6 单轮循环总量（5-subgoal 任务中段）

| 组成 | Token |
|---|---:|
| cortex（含截图） | 6,866 |
| executor × 2 轮（各含 schema 3,560） | 8,308 |
| orchestrator | 1,753 |
| **单轮合计** | **16,927** |

> **executor 的 8,308 已超过 cortex**，是单轮最大成本项。

---

## 三、工具 schema 明细

`bind_tools` 每次全量下发 16 个工具，共 **3,560 token**：

| 工具 | Token | 占比 |
|---|---:|---:|
| `long_press_on` | 646 | 18.2% |
| `focus_and_input_text` | 546 | 15.3% |
| `tap` | 499 | 14.0% |
| `focus_and_clear_text` | 482 | 13.5% |
| `swipe_percentages` | 262 | 7.4% |
| `wait_for_delay` | 192 | 5.4% |
| `swipe_coordinates` | 197 | 5.5% |
| `press_key` | 84 | 2.4% |
| `launch_app` | 82 | 2.3% |
| `stop_app` | 102 | 2.9% |
| `open_link` | 80 | 2.2% |
| `erase_one_char` | 76 | 2.1% |
| `save_note` | 102 | 2.9% |
| `back` | 68 | 1.9% |
| `read_note` | 78 | 2.2% |
| `list_notes` | 63 | 1.8% |

**Top 4 工具（`long_press_on` / `focus_and_input_text` / `tap` / `focus_and_clear_text`）合计 2,173 token，占 schema 的 61%。** 这四个都各自展开了完整的 `Target` 对象（5 字段含 `resource_id` / `resource_id_index` / `text` / `text_index` / `bounds`），存在明显的结构性重复。

---

## 四、`agents_thoughts` 累积曲线（唯一的 O(n) 项）

`graph/state.py:113` 的 `_add_agent_thoughts` 返回 `old + named_thoughts`，**全项目无任何裁剪逻辑**。而 `cortex.py:77` 每轮把整份历史逐条重发。

固定开销基线：system 1,747 + UI 2,527 + 截图 2,064 = 6,338

| Subgoal 数 | 循环轮次 | thoughts 条目 | thoughts 重发 | cortex 总上下文 | thoughts 占比 |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 12 | 288 | 6,626 | 4.3% |
| 5 | 15 | 60 | 1,440 | 7,778 | 18.5% |
| 10 | 30 | 120 | 2,880 | 9,218 | 31.2% |
| 20 | 60 | 240 | 5,760 | 12,098 | 47.6% |
| 30 | 90 | 360 | 8,640 | 14,978 | **57.7%** |

> 假设每个 subgoal 约 3 轮循环、每轮 2 个 tool call。长任务后半段，**过半数 cortex 上下文是 agent 自己写的历史日志**，且这部分成本每轮复利式累加。

同一份 `agents_thoughts` 还被 `orchestrator`（1,556）、`planner`（467）、`outputter`（1,422）、`contextor` 全量消费。

---

## 五、优化空间（按性价比排序）

### 5.1 【高】`agents_thoughts` 无界累积

**问题**：`state.py:113` 只增不减，全项目无裁剪；`cortex.py:77` 每轮全量重发。曲线见第四节。

**对比证据**：`contextor.py:148` 已定义 `MAX_AGENTS_THOUGHTS = 25` 并做了 `agents_thoughts[:MAX_AGENTS_THOUGHTS]` 截断——**正确做法已经存在于仓库中，但只接在 app-lock 这个冷门分支上，cortex 主路径没有接**。

**建议**：cortex 侧保留最近 N 轮原文 + 更早部分压缩为摘要。

**约束（重要）**：`state.agents_thoughts` 同时被以下消费方使用，**只裁剪发给 LLM 的那份，不要改动 state 中的原文**：
- `sdk/agent.py:680` — 导出 thoughts trace 文件
- `sdk/agent.py:710/728/746` — `steps_count` 统计
- `tui/events.py:112-116` — TUI 实时渲染
- `agents/outputter/outputter.py:32` — 最终输出生成

### 5.2 【高】工具 schema 每轮重发 3,560 token

**问题**：`executor.py:74-75` 每次 `bind_tools` 全量下发 16 个工具，且随累积历史每轮重付。

> 5.2 节的初步建议（「精简 docstring」）在深入实测后**被推翻**，实际可行方案见 **5.2 修正版**。

---

## 五·一、工具 schema 优化：实测修正版

### 5.2.1 先看真实的 token 去向

dump `long_press_on` 的真实 schema 后发现，**原先的假设是错的**：

| 误判 | 实际 |
|---|---|
| docstring 里的 `Args:` 会被解析成 per-parameter description | **不会**。整块留在 tool 级 `description` 里，字段自身无描述 |
| `Target` 的开销来自字段描述 | 不是。`Target` 被**完整内联展开**在 4 个工具里，每次 378 token |

`Target` 内联实测：

```
Target 内联于 tap                      378 token
Target 内联于 long_press_on            378 token
Target 内联于 focus_and_input_text     378 token
Target 内联于 focus_and_clear_text     378 token
→ 4 × 378 = 1,512 token，占 schema 的 42%
```

### 5.2.2 各变体实测

| 变体 | Token | Δ | 幅度 |
|---|---:|---:|---:|
| **BASELINE** | 3,560 | — | — |
| 仅删 `title` | 3,184 | −376 | −10.6% |
| 仅折叠 `anyOf`-null | 3,062 | −498 | −14.0% |
| 删 `title` + 折叠 `anyOf` | 2,938 | −622 | −17.5% |
| 以上 + `Target` 共享 `$ref` | 3,110 | −450 | −12.6% |
| 结构优化 + 删 `Args:` 块 | 2,844 | **−716** | **−20.1%** |

> ⚠️ **2026-10-21 更新**：上表的 −716 是**旧版工具集**的数字。拿现在的 16 个
> 工具实测（`probe_out/probe_tool_schema_surgery.py`）：
> **4,333 → 3,871 token（−462，−10.7%）**，且只有 5 个用 `Target` 模型的工具有收益
> （各 −113）—— 其余工具的 schema 里没有 `title` / `anyOf` 可砍。
> **手术已落地**：`utils/tool_schema_surgery.py` + `services/llm.py` 的
> `ToolSchemaSurgeryMixin`（挂在 `_get_request_payload` 之后，一处覆盖全部
> OpenAI 兼容 provider，对工具实现零侵入）。
| 结构优化 + 删 `Args:` + 中文化 | 3,412 | −148 | −4.2% |

**结论：真正的大头是 JSON 结构噪音，不是措辞。**

- `anyOf: [{type:"string"},{type:"null"}]` —— 每个 nullable 字段都展开，占 14.0%
- `title: "X"` / `"Width"` / `"ElementBounds"` —— 纯展示字段，占 10.6%
- tool 级 description 里的 `Args:` 块 —— 与字段语义重复，再占 7.5%

**`$ref` 共享不可行**：OpenAI 的 `tools` 是数组，每个 function schema 必须自包含，**跨工具 `$ref` 无法解析**。实测该变体反而比「结构优化」多 172 token（每个工具仍需自带一份完整 `$defs`）。

### 5.2.3 推荐方案

**只做结构手术，保持英文，合计 −716 token（−20.1%）**（**实测现值 −462 / −10.7%，见上面的更新**），落在 `bind_tools` 之后、请求体组装完成时，对工具实现零侵入：

1. 递归删除所有 `title` 键
2. `anyOf: [X, null]` 折叠为 `X`（保留 `default: null` 表示可省略）
3. 删除 tool 级 description 中的 `Args:` 段落（字段语义已在字段描述中）
4. 保留全部语义描述，不做语言替换

落到 executor：

| 轮次 | Baseline | 结构优化后 |
|---:|---:|---:|
| 1 | 4,164 | **3,448** |
| 4 | 4,761 | **4,045** |
| 8 | 5,557 | **4,841** |

> ⚠️ `anyOf` 折叠会丢失 `nullable` 语义。多数 provider 依据 `default: null` + 非 required 判断可省略，但**上线前需对目标 provider 做一次真实 tool-call 验证**。

### 5.2.4 关于「描述简化」的边界

`Target` 占 42%，但它**不能在工具之间共享**。能做的只有让它自身更小（结构手术），或减少需要 `Target` 的工具数量。schema description 里的冗余（如 `long_press_on` 的 `duration_ms` 四行 bullet 共约 60 token）值得单独精简，但相比结构噪音属于次要项。

---

## 五·二、中文化：实测结论是**不划算**

用户提出的假设是「提示词换中文更省」。**实测否定这一假设。**

### 5.2.1 token 密度：中文全面更贵

10 组同义句逐条对照，**中文全部更贵**：

| 英文 | 中文 |
|---:|---:|
| 11 | 16 |
| 13 | 21 |
| 11 | 18 |
| 22 | 29 |
| 23 | 33 |
| 21 | 28 |
| 28 | 48 |
| 20 | 31 |
| 40 | 58 |
| 17 | 27 |
| **206** | **309（+50.0%）** |

**机制**：cl100k_base 对英文约 4 字符/token，对中文约 1 字/token。中文语义虽更紧凑，但字符/token 效率低约 4 倍，**净效果是贵 50%**。

### 5.2.2 放大到 system prompt

| 文件 | 现值 | 全量中文化后 | Δ |
|---|---:|---:|---:|
| `cortex.md` | 1,747 | ~2,620 | **+873** |
| `planner.md` | 748 | ~1,122 | **+374** |
| `executor.md` | 336 | ~504 | **+168** |

### 5.2.3 落到工具 schema

结构优化后再中文化，**从 2,844 反弹到 3,412（+568）**——把已经省下的又吐回去还多。

### 5.2.4 补充：模型侧因素

上述用 `cl100k_base`（OpenAI 系）测得。默认模型是 `google/gemini-2.5-pro`，其分词器对中文的处理不同，比例会变但**「中文 token 密度低于英文」这一方向不会反转**——多字节字符在任何 BPE 里都不占优势。

**结论：中文化的收益是负的，不建议为了省 token 而做。** 中文提示词的其他价值（模型的中文理解能力、与业务术语一致性）是独立议题，不在本节成本讨论范围内。

### 5.3 ✅ 截图已缩分辨率（2026-10-17 落地）
> ⚠️ 下面是 2026-10-16 的**旧结论**，已经过期。
>
> 10-17 的 `78bc8a2` / `34084bf` 就实现了缩放：`utils/image_compress.py` 的
> `compress_screenshot_base64` 默认带 `max_width=640 / max_height=1536`
> （`DEFAULT_MAX_WIDTH/HEIGHT`），**截图早就缩过了**。
- `android_controller.py:299-310`
- `ios_controller.py:339-356`
- `cloud_device_controller.py:407-418`

1080×2400 按 Gemini tile 计价约 2,064 token；降至 540×1200 约 516 token，**省约 75%**。

**风险**：影响 cortex 的视觉判断质量。UI 元素坐标已从层级树获取，不依赖像素定位，但颜色/图标/徽标等视觉线索可能受影响。**建议真机 A/B 对比后再上线。**

### 5.4 【中】`executor_messages` 单轮内线性累积

实测 1/4/8 轮为 199 / 790 / 1,578 token。
`cortex.py:139` 的 `RemoveMessage(REMOVE_ALL_MESSAGES)` 会在下一轮清空，**因此跨轮不累积**，仅影响单个 cortex 决策内的多工具连续调用场景。优先级低于前两项。

### 5.5 ✅ telemetry 已有 token 埋点（2026-10-21 补）

> ⚠️ 下面是 2026-10-16 的旧结论，已经过期。
>
> 10-19 的 `deda761` 接上了 usage 捕获，10-21 又补了**缓存命中量**
> （`utils/llm_usage.py` 的 `extract_cache_tokens` + telemetry 的 `cached_tokens`）。
> 顺带修掉一个根因：langchain-openai 在**流式**请求下、且 `stream_usage` 未设时
> **不转发 provider 报的 usage**（`usage_metadata` 直接是 `None`）——
> `get_minimax_llm()` 现在显式传 `stream_usage=True`。
> 详见 `doc/cache-hit-rate-audit-20261020.md` 第 7 节。

#### 5.5 【历史】telemetry 缺少 token 埋点

`services/telemetry.py` 中 `capture_executor_action` / `capture_cortex_decision` / `capture_task_completed` **均无 token / usage 字段**。

本报告所有数字均为离线复现所得。**要做优化收益验证，需先将 `response.usage` 接入 telemetry**，否则改动效果无法量化。

---

## 六、附带发现：`MAX_MESSAGES_IN_HISTORY` 是失效的

`constants.py:2` 定义 `MAX_MESSAGES_IN_HISTORY = 25`，`summarizer.py` 据此裁剪 `state.messages`。

但全项目 **没有任何一处向 `state.messages` 写入**（grep 仅在 `state.py:26` 定义处与 `summarizer.py` 自身命中）。

真正累积的是 `agents_thoughts` 与 `executor_messages`，**两者都不受该上限保护**。

> 即：仓库看起来具备历史裁剪机制，实际上对 token 成本无任何作用。

---

## 六·五、附带发现：`cortex.py` 设备信息拼接存在运算符优先级 Bug

`agents/cortex/cortex.py:67-75`：

```python
HumanMessage(
    content="Here are my device info:\n"
    + self.ctx.device.to_str()
    + f"Device date: {state.device_date}\n"
    if state.device_date
    else "" + f"Focused app info: {state.focused_app_info}\n"
    if state.focused_app_info
    else ""
)
```

三元表达式的优先级低于 `+`，整个表达式实际被解析为：

```python
("Here are my device info:\n" + device.to_str() + f"Device date: {...}\n" if state.device_date
 else "") + (f"Focused app info: {...}\n" if state.focused_app_info else "")
```

**实测四种取值组合的结果：**

| `device_date` | `focused_app_info` | 实际输出 | 问题 |
|---|---|---|---|
| 有 | 有 | `Here are my device info:\n<device>Device date: ...` | **focused app 丢失** |
| 无 | 有 | `Focused app info: com.whatsapp` | **整段设备信息丢失** |
| 有 | 无 | `Here are my device info:\n<device>Device date: ...` | 正常 |
| 无 | 无 | `""` | 空消息 |

**影响**：`focused_app_info` 由 `contextor.py:73` 每轮写入，在 Android 上几乎恒为有值。当 `device_date` 存在时，**cortex 永远收不到当前前台应用信息**；当 `device_date` 缺失时，**收不到任何设备信息**（分辨率等）。

> 注意：`contextor.py:76` 的 `agents_thoughts: [agent_outcome]` 在正常路径下 `agent_outcome` 恒为 `None`，经 `state.py:88` 过滤后为空列表——即**正常任务中 contextor 每次循环都向 `agents_thoughts` 追加一个空项**（不增加 token，但说明该字段在本路径下无实际作用）。

**建议**：改为显式拼接，避免三元表达式歧义。属正确性修复，与 token 优化无关，建议独立提交。

---

## 七、数据可信度说明

| 项 | 可信度 | 说明 |
|---|---|---|
| Cortex / Executor / Planner / Orchestrator / Outputter token | 高 | 用真实模板 + 真实 Pydantic schema + 真实 `get_tools_from_wrappers()` 产物复现 |
| `agents_thoughts` 增长曲线 | 中 | 增长规律确定（线性、可复现），但具体数值依赖「每 subgoal 轮次」与「每轮 tool call 数」两个假设参数 |
| 截图 token 数 | 中 | 按 Gemini tile 计价公式（`(w//768+1) × (h//768+1) × 258`）估算，非 API 返回的 `usage` 实测值 |
| 端到端真实总量 | **低（未测）** | 需接 telemetry 后用真实 `response.usage` 校准 |

**已知测量偏差（已修正，记录以免复现时踩坑）**：
`outputter/human.md` 使用 `{% for thought in agents_thoughts %}` 遍历，而 `planner/human.md` 与 `orchestrator/human.md` 使用 `{{ agent_thoughts }}` 直接插值。构造探针时，**outputter 必须传 list，planner/orchestrator 必须传 join 后的字符串**。传错类型会让 outputter 退化为「按字符迭代」，实测虚高到 12,919 token（真实值 1,441）。

---

## 八、建议的落地顺序

| # | 动作 | 预期收益 | 风险 |
|---|---|---|---|
| 1 | **接 token telemetry**（5.5） | 验证手段本身 | 无 |
| 2 | ✅ **工具 schema 结构手术**（5.2.3） | **−462/次（−10.7%，实测）** | 已落地。`anyOf` 折叠的 nullable 风险已验：provider 不校验 tool_call 参数的 schema（显式 null 两种 schema 都 200） |
| 3 | **裁剪 `agents_thoughts` 发送量**（5.1） | 长任务最高可省 50%+ | 中，影响长程记忆，需定裁剪策略 |
| 4 | **截图降分辨率 A/B**（5.3） | −75% 截图成本 | 中，影响视觉判断，需真机对比 |
| 5 | `executor_messages` 截断（5.4） | 单轮内 ~1.5k | 低 |
| — | ~~中文化提示词~~ | **负收益 +50%** | 不做 |

**注意 2 与 3 的量级差异**：schema 结构手术是**每轮固定省 462**（实测），确定性强、无语义损失，已落地；`agents_thoughts` 裁剪收益随任务长度增长，但需要你先定策略（见 5.1 的约束说明）。

---

## 九、复现方式

```powershell
# 工具 schema 变体（含 v1 的错误示范，保留以说明踩坑）
python v\probe_tool_schema_variants.py
python v\probe_tool_schema_variants_v2.py   # ← 修正版，以此为准

# 中文化 token 密度
python v\probe_prompt_zh.py

# 各 agent 上下文构成 / 增长曲线
python v\probe_token_budget.py
python v\probe_thought_growth.py
python v\probe_minor_nodes.py
```

> `v/probe_tool_schema_variants.py`（v1）**输出结论是错的**，它误以为 docstring 的 `Args:` 会被解析成字段描述，导致按属性名替换几乎不命中。保留该文件用于记录这个坑，**引用数据时以 v2 为准**。
