# 前缀缓存命中率审计（2026-10-20）

> 审计对象：exec 212 真机运行（`mobile-manager/probe_db/usage_212.json`、`thoughts_212.json`）
> 审计方式：**对 MiniMax 真实 API 复刻 cortex 请求形状**，逐块 A/B 定位击穿点；
> 在真机上跑端到端验证；抓真实 prompt 重放定位断点。
> 探针：`probe_out/probe_cache_breakpoint.py`、`probe_out/probe_cortex_replan_cost.py`、
> `probe_out/probe_cache_curve.py`、`probe_out/probe_device_run_usage.py`、
> `probe_out/probe_prompt_breakpoint.py`、`probe_out/probe_trim_tail.py`
> 结论一句话：**93.5% 是模拟产物，10.4% 是重构前的旧架构；架构修好后真机实测 44.9%，**
> **再把截图改成按需取之后升到 74.6%（cortex 76.4%），prompt 体积 −80%。**
> **再往上的天花板不在块序，在「易变尾部太大」：尾部里 92% 是每轮都发的那张截图
> （47,672 字符 base64，且长度随画面变 → 请求总长每轮都变 → 缓存归零）。**
> 已落地：P0 标注模拟值 / P1 计划日志挪位（省 8,000/replan）/ P2 工具列表排序 /
> P3 命中率看板 / 遥测补 `cached_tokens` / **修掉 `stream_usage` 没开导致 usage 恒为 0 的根因**。

---

## 0. 先把两个数对清楚

| 数字 | 出处 | 是什么 |
|---|---|---|
| **93.5%** | `probe_out/probe_thoughts_compaction_e2e.py` | **模拟**。prompt 是 `"S"*6085 + "D"*1419 + thoughts + "X"*6004` 拼的重复字符，**从未打过真 API** |
| **10.4%** | `probe_db/usage_212.json`（provider 回报） | **实测**，但是**重构前**的架构 |
| **87.5%** | 本次探针 E6（真 API） | 当前架构实测，稳态可达 100% |

三者不矛盾，是三个不同的东西。**问题在于 10-20 的报告只引用了 93.5%，而没引用 10.4%。**

### 时间线（决定了哪个数字代表什么）

| 时间 | 事件 | 对缓存的影响 |
|---|---|---|
| 10-18 14:15 | `9af5a62` 提交；exec 212 跑完（`probe_db/*.json` 落盘于 14:18） | **这是唯一有 provider usage 数据的运行** |
| 10-19 19:39 | `deda761` 重构 cortex 提示词结构 | system 整段静态化、变量移尾、去掉 `[-25:]` 滑窗、加只追加的计划日志 |
| 10-20 22:31 | e2e 脚本产出「93.5%」 | **模拟**，不是实测 |
| 10-20 23:10 | `c868d09` thoughts 压缩 + **orchestrator 把 thoughts 挪到可变计划之前** | 修掉 orchestrator 的致命块序 |

**结论：10.4% 衡量的是 `deda761` + `c868d09` 之前的架构。** 那两笔提交确实修好了架构，但没有任何一次真实测量证明过。

---

## 1. 旧架构为什么是 10.4%

`git show deda761 -- cortex.py` 的改动前版本：

```python
system_message = Template(...).render(
    initial_goal=state.initial_goal,          # 每轮变
    subgoal_plan=...,                          # 每轮变（status 原地改写）
    current_subgoal=...,                       # 每轮变
    executor_feedback=executor_feedback,       # 每轮变
    executor_tools_list=..., locked_app_package=..., perception=...,
)
messages = [SystemMessage(system_message), HumanMessage(device_info_lines)]
```

**四个每轮变的值全塞在 system prompt 里** → system prompt 每轮都变 → 缓存只能命中它的静态开头。

观测值对得上：

```
cortex:      input 555,020 | cached 84,856 | 52 calls -> 15.3%
orchestrator: input 392,062 | cached 13,561 | 18 calls ->  3.5%
```

cortex 缓存均值 1,632 token / 次，**且不随 prompt 增长**（末轮 input 54,386，cached 仍只有 3,584）——这是「只命中固定前缀」的典型形状，不是「命中一大段但偶尔失手」。

而且 `cached` 全部是 **64 的整数倍**（128 / 192 / 640 / 1,792 / 2,816 / 3,584…），对齐 provider 的块粒度。

旧 orchestrator 的 `human.md` 块序更糟：

```
**Initial goal** → **Subgoal plan**(每轮变) → **Subgoals to examine**(每轮变) → **Agent thoughts**(只追加)
```

**可变块排在只追加块前面** → 每轮从 plan 那一行起整段失效，thoughts 永远命不中。3.5% 就是这么来的。

---

## 2. 探针方法（为什么这次的数可信）

用**真 API + 真 prompt 形状**，而不是模拟：

- system prompt：Jinja 渲染真 `cortex.md`（6,861 字符）
- thoughts：真归档 122 条，走真 `render_thoughts_history`
- tools：真 `CortexOutput` 的 OpenAI function schema
- endpoint / model / effort：与 `llm-config.override.jsonc` 一致（minimax / `MiniMax-M3.1-Flash-Preview`）

先验两件事，再谈命中：

| 实验 | 结果 |
|---|---|
| E0 逐字节确定性 | 两次构造的 body 逐字节相同 → provider 报的 `prompt_tokens` 也相同（2,834 = 2,834）✅ 请求本身没有隐藏变量 |
| E1 同一请求发两次 | 2,834 → cached 2,832（**99.9%**）→ provider 缓存本身没问题，字段是 `usage.prompt_tokens_details.cached_tokens` |

**这两个实验是全部结论的前提**：先证明「请求是确定的」和「provider 会缓存」，再谈「哪里击穿」。

---

## 3. 实测结果

### 3.1 追加式增长 —— 架构是对的（E2）

| thoughts 条数 | prompt | cached | 率 |
|---|---|---|---|
| 5 | 2,830 | 2,816 | 99.5% |
| 20 | 3,538 | 3,536 | 99.9% |
| 40 | 4,736 | 4,621 | 97.6% |
| 70 | 6,940 | 6,938 | 100.0% |
| 100 | 8,990 | 8,988 | 100.0% |
| 122 | 10,479 | 10,477 | 100.0% |

**只追加的前缀被完整复用。** `deda761` 的分层（恒定 → 只追加 → 每轮变）是对的。

顺带测出 provider 的一个行为：**缓存滞后一轮**。第 n 次请求命中的是第 n-1 次的 prompt 长度，新追加的那一段要等下一次才可命中。所以稳态命中率 ≈ `1 - 尾部块 / prompt`，尾部块（感知块等）每轮约 700 token。

### 3.2 replan —— 当前架构唯一的确定漏点（E3）

只改一件事：在**块 2（设备信息 + 目标 + 计划日志）**尾部追加一版新计划，其余不变。

| 请求 | prompt | cached | 率 |
|---|---|---|---|
| 块2 = `PLAN v1` | 10,479 | **10,477** | 100.0% |
| 块2 = `PLAN v1` + `PLAN v2` | 10,503 | **2,387** | 22.7% |

**Δ = −8,090 token 满价重算。**

原因：块 2 排在 thoughts **前面**。计划日志虽是「只追加」，但追加点在 thoughts 之前 → 从追加点起后面全部重算。**thoughts 全长（此处 10.5K）一次性满价重算。**

replan 只在子目标失败时发生（`graph.py:45` `convergence_gate`），但长任务里子目标多、失败也常见（exec 212 一共 18 个子目标）。**每次 replan 赔一次「当前 thoughts 全长」的满价 token。**

好的一面：replan **不会持续恶化**。E6 实测 replan 之后下一轮立刻回到 100%（7,684/7,686）。所以这是**一次性代价**，不是慢性病。

### 3.3 system prompt 里埋着未排序的工具列表（E4）

`format_tools_list`（`tools/index.py:66`）返回的是 `get_tools_from_wrappers` 的**注册顺序**。

实测：工具列表在 system prompt 的**第 3,950 / 6,861 字符处**（约 58% 深度）。把它换序，缓存从 10,477 掉到 **1,169** —— 破在 system 内部 58% 的位置，后面全废。

现状**是稳定的**（`EXECUTOR_WRAPPERS_TOOLS` 是模块级固定 list），所以这不是活 bug。但它是个**单点**：哪天有人把工具注册改成 dict/set 遍历，system prompt 就会在进程重启后换序，**整个缓存静默归零**。

### 3.4 tools 数组不参与缓存键（E5）

去掉 `tools` 数组后，缓存量不变（只少了 tool schema 自身的 ~370 token）。**tools 不在 MiniMax 的缓存键里**——这条是好消息，少一个担心。

### 3.5 端到端（E6）：模拟 12 轮真实序列

用真归档 thoughts、尾部每轮变（模拟感知块）、12 轮：

```
合计  prompt=80,372  cached=70,315  命中率=87.5%  满价重算=10,057
```

对照 2026-10-20 e2e 声称的 93.5%：量级对得上（那 700 token 的缺口就是每轮变的尾部块），但**这次是打真 API 得到的**。

### 3.6 逐轮曲线反过来验证了「旧架构」的结论

`probe_out/probe_cache_curve.py` 把 `usage_212.json` 摊成逐轮曲线后，旧架构的
命中形状非常干净：

```
step  calls      input    cached       率       块数
    0      1      1,551       188   12.1%       2
    4      1      7,454     2,816   37.8%      44
    7      1      5,990     2,816   47.0%      44
   15      1      9,806     2,816   28.7%      44
   22      1      7,682       128    1.7%       2
```

命中量**双峰**：要么 ≈2,816（= 旧 system prompt 里静态部分的长度），要么 ≈128
（缓存被驱逐，只剩共享的头部）。这正是「变量渲染在 system prompt **末尾** →
每轮从变量处断开、只命中静态头」的形状。变量若在中间，命中量会是另一个固定值；
若在开头，则是 0。双峰正好对应「命中静态头」+「偶发驱逐」。

---

## 4. 本次已落地的改动

### P0 — e2e 脚本与文档的「模拟值」标注 ✅

`probe_thoughts_compaction_e2e.py` 第 5 节的输出改成 `模拟命中率`，并加了一段说明
（prompt 是重复字符拼的、假设易变内容在末尾、算不出 replan 的代价、真 API 实测见
`probe_cache_breakpoint.py`）。

`doc/thoughts-compaction-implementation.md` 与 `doc/thoughts-compaction-plan.md`
里那张 93.5% / 91.1% 的对比表也加了同样的警示：真实归档的 provider 值是 10.4%，
两个数量的不是同一个东西。

**这三条文档/脚本是 10-20 那次报告的原始依据，不标注的话下次还会被当成实测引用。**

### P2 — 工具列表按名称排序 ✅

`format_tools_list`（`tools/index.py:66`）改成 `sorted(...)`。

理由见 3.3：工具清单埋在 system prompt 的 58% 深度处，顺序一变缓存就断。现状是
稳定的（模块级固定 list），但稳定性是「碰巧」的 —— 改成按定义稳定只要一行。

**只排名字清单，不排真正的 tools schema 数组**：MiniMax 的缓存键不含 tools
（3.4 实测），改 schema 顺序没有缓存收益，却会改变模型选工具时的平局顺序。

### P3 — 命中率曲线看板 ✅

`probe_out/probe_cache_curve.py`：把 `usage_*.json` 摊成逐轮曲线，标出击穿轮次，
并用 `cached / 64`（provider 的块粒度）反推命中块数。

用法：`python probe_out/probe_cache_curve.py [usage_212.json]`

---

## 5. P1 —— cortex 的计划日志挪到 thoughts 之后（已实施）

### 5.1 先说一个差点让 P1 做错的事实

P1 的出发点是 3.2 节的 E3：计划日志在 thoughts **前面**，replan 时追加会让
thoughts 全长满价重算。**但那只是一半。** 补一件事就变了：

> **exec 212 全程 0 次 replan。**
> 18 个子目标全是 orchestrator 自己推进的「NEXT SUBGOAL」（它只改 `subgoal_plan`，
> 不写计划日志），planner 只在 START 跑了一次 → `subgoal_plan_log` 全程只有
> 「PLAN v1」一段，**从不增长**。

于是 P1 在**常规路径**上是有代价的：计划日志挪到 thoughts 之后，它就每轮都要
满价重算一次（约 150 token）。而常规路径上它本来是恒定的、可以全程命中。
**判据从「replan 贵不贵」变成「replan 一次省多少 vs 常规路径每轮多付多少」。**

### 5.2 测量：12 轮序列分辨不出来，改用单发对照

先跑了 12 轮序列 A/B（`probe_out/probe_cortex_block_order.py`）：

```
现状 / 无 replan    总请求 81,868 | 命中 73,837 ( 90.2%) | 满价重算 8,031
现状 / 第7轮 replan 总请求 81,924 | 命中 75,356 ( 92.0%) | 满价重算 6,568
```

**这个数据说明不了任何事** —— 「有 replan」的那组反而满价重算**更少**。
原因是 **噪声底约 ±2,000 token**（同一份请求两次跑，满价重算能差出 1,463），
而要分辨的效应量只有「计划日志 × 轮数」≈ 1,440，**低于噪声底**。

所以改用 **E3 式单发对照**（`probe_out/probe_cortex_replan_cost.py`）：
同一序列、只差一个块、背靠背发。**注意：只有第 1 轮有效** —— 第 2/3 轮的请求
与第 1 轮逐字节相同，缓存已经命中，毫无信息量。

| 变体 | replan 当轮 cached | 满价重算 |
|---|---:|---:|
| 现状（计划日志在 thoughts 前） | 2,329 | **8,261** |
| P1（计划日志在 thoughts 后） | 10,411 | **183** |

**P1 每次 replan 省 ~8,000 token**，且这个差随 thoughts 变长而变大（thoughts 越长，
plan log 越靠前，损失越大）。

### 5.3 结论与取舍

- 常规路径多付：计划日志每轮满价重算，约 **150 token/轮**（122 条 thoughts、
  3 个子目标的规模）。
- replan 路径省：约 **8,000 token/次**，且随 thoughts 变长而放大。
- **盈亏平衡 ≈ 1 次 replan**。exec 212 有 0 次（亏 ~7,800，占总量 1.5%），
  但 replan 恰恰发生在**任务不顺、thoughts 最长**的时候 —— 那正是最该省的时候。

**实施**：`minitap/mobile_use/agents/cortex/cortex.py`

```
[system][设备+目标][thoughts][计划日志][status][failed][feedback][设备状态][感知块]
```

内容逐字不变，只是把计划日志从 head 块里拆出来独立成一条消息，挪到 thoughts 之后。
消息条数不变（计划日志恒发，无计划时填 `(no plan)`），所以不会引入新的条数漂移。
**orchestrator 早就是这么修的**（`c868d09` 把 `**Agent thoughts**` 提到
`**Subgoal plan**` 前面），cortex 现在对齐了。

---

## 6. 真机验证（P1 块序改动不破坏行为）

改块序改的是**模型看到的东西**，缓存探针证明不了这一点，必须真机跑。

设备：Redmi Turbo 4（24129RT7CC）/ Android 16 / 1220×2712，adb 设备
`89R8ZXNBGQIRJRLF`。任务：「打开系统设置应用，进入电池页面，告诉我当前电量
百分比和充电状态」。

结果：**任务成功（exit=0，`Automation is success`）**，过程：

1. 打开「设置」→ cortex 读到 MIUI 设置首页，决定用搜索框输入「电池」
2. **输入失败**（uiautomator `set_text` 报 `Selector [focused=True]` 失败，
   回落到 ADB shell 也没生效，屏幕上还是占位符「搜索系统设置项」）
3. **模型自己发现失败并换策略**：「I will not repeat the failed typing action;
   instead I'll scroll the settings list down」→ 滑动 → 找到「省电与电池」(index 68)
   → 点击 → 读到「预计充满时间 0 时 59 分 | 剩余电量 80%」+ 状态栏 80% + 充电中
4. 汇报：电量 80%，充电中，预计 59 分钟充满

**结论：P1 的块序改动没有破坏模型行为。** 推理链条连贯，失败后能自我纠正并换
策略（这段推理的完整内容见真机 trace）。

配套工具：`probe_out/probe_device_run_usage.py` —— 运行期猴补丁把每次 LLM 调用的
usage（含 `cached_tokens`）落到本地 JSONL，**不改生产代码**。

---

## 7. 顺手补上的一个洞：遥测里没有缓存率

`extract_usage` 只认 input / output，**缓存命中量是第三个数**（provider 报的是
「本次输入里有多少 token 直接读了缓存」，是 input 的子集）。它没被记进遥测，
所以 **mobile-use 自己的运行里根本看不到命中率** —— manager 那边是从原始响应读的。

新增：

- `minitap/mobile_use/utils/llm_usage.py`：`extract_cache_tokens()`，认四家字段名
  （OpenAI 兼容 `prompt_tokens_details.cached_tokens` / DeepSeek
  `prompt_cache_hit_tokens` / Anthropic `cache_read_input_tokens` / langchain
  归一化后的 `input_token_details.cache_read`）。
- `minitap/mobile_use/services/telemetry.py`：`capture_llm_usage` 增加
  `cached_tokens` 字段（**新增字段，不改旧字段**，旧 trace 按 0 处理）。
- `minitap/mobile_use/services/llm.py`：`_record_usage` 传进去。
- `tests/mobile_use/test_llm_cache_tokens.py`：7 条测试，覆盖四家字段名 +
  端到端进遥测。

**踩坑记录**：langchain 归一化后的键是 `input_token_details`（**token 后面没有 s**），
我一开始写成了 `input_token_details`（有 s），**代价是静默返回 0** —— 命中率看起来
永远是 0，而且不报错。这类"少一个字母"的 bug 在没有断言的路径上完全隐形，
所以那条测试是必须的。

### 7.1 顺带挖出来的更大一个洞：headless 路径的 usage 遥测**恒为 0**

加了 `cached_tokens` 之后跑真机，发现**每一次 LLM 调用的 input / output / cached
全是 0**，`model` 也是 null。查下来是这样：

| 入口 | usage 从哪来 | headless CLI |
|---|---|---|
| TUI（`tui/session.py:161`） | graph-config callback `on_llm_end` | ❌ 没挂 |
| 安卓服务（`android/server.py:148`） | 同上 | ❌ 没挂 |
| manager runner-host（`host.py:323`） | 同上 | ❌ 没挂 |
| **headless CLI（`main.py:83`）** | 只能靠 `invoke_llm_with_timeout_message` 里的 ContextVar 捕获槽 | **这条在图里不生效** |

`main.py:83` 的 `graph_config_callbacks` 默认是 `[]`，所以裸 CLI 路径**没有任何
usage 来源**；而 `services/llm.py` 那套「给 `ChatOpenAI.agenerate` 打补丁 +
ContextVar 捕获槽」的方案，**只在直接调用时有效**（`tests/mobile_use/test_llm_usage_capture.py`
测的正是直接调用），在 LangGraph 的图里拿不到东西。

**后果**：`services/telemetry.py` 里 `capture_llm_usage` 的 docstring 写着
「这是成本核算的**唯一数据源**」—— 但在 headless 路径上它**每一行都是 0**。
manager 那边之所以有数（`usage_212.json`），是因为它自己有 `runner_host/event_bridge.py`
这套 graph-config callback，**没走 mobile-use 的遥测**。

**这就是为什么 mobile-use 自己的运行里看不到命中率** —— 不是没记录，是记的是 0。

### 7.2 根因：`stream_usage` 没开，provider 报的 usage 根本没进 langchain

**这一段走了弯路，记下来免得下次再绕。** 依次排除了三种错误修法：

| 试过的修法 | 为什么不行（实测） |
|---|---|
| ① 给 `ChatOpenAI.agenerate` 打补丁 + ContextVar 捕获槽 | 埋针显示 **patch 一次都没被调用**；而且 ContextVar 跨不过图的任务上下文，`_capture_sink.get()` 拿到 None 就静默丢弃 |
| ② graph-config callback（`on_llm_end`）直接报遥测 | **实测 `on_llm_end` 在节点内部的 LLM 调用上不触发**（`on_chain_start` 触发、`on_llm_start` 不触发）；`stream_mode="messages"` 给的是 `AIMessageChunk`，`usage_metadata=None` |
| ③ 用 `stream_mode="messages"` 的 chunk | 同上，chunk 上没有 usage |

**真正的根因在 manager 那边早就写好了** ——
`runner-host/src/runner_host/llm_usage_patch.py` 的开头就是答案：

> Measured on the real endpoint, same model, same prompt:
>
> | call | `usage_metadata` |
> |---|---|
> | non-streaming `ainvoke` | full numbers |
> | streaming, `stream_usage` unset | **`None`** |
> | streaming, `stream_usage=True` | full numbers |
>
> langchain-openai only forwards usage on a stream when `stream_usage=True`.

**⇒ 修法：`get_minimax_llm()` 里加 `stream_usage=True`。** manager 是在宿主进程里
补这个开关（它把 mobile-use 当上游，不改上游代码）；mobile-use 自己就是上游，
**直接在工厂里设**。

**⇒ 验证（真机，17 次调用，`probe_out/probe_device_run_usage.py` 采的）**：

| agent | calls | input | cached | 命中率 |
|---|---:|---:|---:|---:|
| cortex | 14 | 89,859 | 40,278 | **44.8%** |
| executor | 1 | 5,083 | 186 | 3.7% |
| hopper | 1 | 4,374 | 186 | 4.3% |
| planner | 1 | 1,248 | 60 | 4.8% |
| **合计** | **17** | **100,564** | **40,710** | **40.5%** |

逐轮曲线（cortex 是主力）：

```
 #      time agent          input  cached       率     块数(64)
 1  11:09:28 planner        1,248      60   4.8%          0
 2  11:09:40 cortex         5,007     186   3.7%          2
 3  11:09:45 hopper         4,374     186   4.3%          2
 4  11:10:08 cortex         5,260   2,427  46.1%         37
 5  11:10:18 cortex         6,775   2,576  38.0%         40
 6  11:10:26 cortex         5,616     186   3.3%          2   <- 一次整段击穿
 7  11:10:29 executor       5,083     186   3.7%          2
 8  11:10:49 cortex         5,716   2,733  47.8%         42
 9  11:10:59 cortex         7,278   2,941  40.4%         45
10  11:11:11 cortex         6,097   3,237  53.1%         50
11  11:11:33 cortex         6,225   3,444  55.3%         53
12  11:11:44 cortex         6,873   3,058  44.5%         47
13  11:12:00 cortex         7,391   3,742  50.6%         58
14  11:12:10 cortex         6,225   3,886  62.4%         60
15  11:12:32 cortex         6,891   3,569  51.8%         55
16  11:12:51 cortex         7,138   4,244  59.5%         66
17  11:13:20 cortex         7,367   4,049  55.0%         63
```

**这是健康前缀缓存该有的形状**：命中量**随会话单调增长**（186 → 4,244），命中率
从 3.7% 一路涨到 **62.4%**。

**对比重构前**（exec 212，10.4%）：cached 封顶在 3,584 不再增长 —— 那是
「只命中固定前缀」的形状。**架构修好之后，cached 会跟着 prompt 一起长。**
（这是个 17 次调用的短任务；长任务上命中率会继续爬升。）

**⇒ 顺带改了 `services/llm.py` 的 `_record_usage` 语义**：拿不到 usage 时
**不记这一行**（原来记 0）。理由：manager 的口径是「没上报过 usage 时这些字段
是 null——『没上报』不等于『零消耗』」，而「provider 没报」和「我们没拿到」在
实现上分不开；记 0 会把每次调用算成免费。
`tests/mobile_use/test_llm_usage_telemetry.py` 里锁着旧契约的那个用例已改成
`test_missing_usage_records_nothing_not_crash`，理由写进了 docstring。

---

## 8. 断点定位：prompt 结构是好的，天花板在「易变尾部太大」

### 8.1 现象：cached 偶尔归零，下一轮又完全恢复

**问题**：真机 30 次调用里，cortex 的 cached 序列有两轮突然归零到 186：

```
cortex: [186, 2427, 2576, 186, 2733, 2941, 3237, 3444, 3058, 3742, 3886, 3569,
         4244, 4049, 4464, 4929, 5129, 5353, 5572, 186, 5874, 6390, 6622, 2304]
                              ^^^                            ^^^
```

归零的那一轮之后，**下一轮立刻恢复并超过之前的高点**（186 → 2,733 / 186 → 5,874）。
**⇒ 这不是 prompt 断裂** —— 断裂会停在断点（像 E3 的 2,387），不会归零后完全恢复。

**⇒ 抓真实 prompt 重放**（`probe_out/probe_prompt_breakpoint.py` 在 agent 的 LLM 调用
入口抓「这一轮真正发出去的 messages + tools」，`probe_replay_real.py` 原样重放）：

```
 #  prompt_tok  cached   公共前缀(字符)   说明
 1    31,796     186          0        (冷)
 2    32,031     186      7,400       ← 滞后一轮
 3    32,213   2,228      8,152
 6    33,010   2,736     10,547      **本轮变短**
 7    33,819     128     11,812      ← 归零！上一轮变短
 8    26,297   3,099     12,783      **本轮变短**
```

**⇒ 规律：`cached` ≈ 上一轮的公共前缀（滞后一轮），而「变短」的下一轮会归零。**

### 8.2 排除了三个错误假设

| 假设 | 实测 | 结论 |
|---|---|---|
| 被别的 agent 的调用挤掉（LRU） | `P P Q P` → 第三次 P 仍 1,067（99.8%） | ❌ 异构请求不挤掉 |
| TTL 到期 | `P P (等 90 秒) P` → 仍 1,067（99.8%） | ❌ 90 秒内不过期 |
| 提示词变短就打掉缓存 | `P → P_short` → P_short 仍 2,446（**99.4%**） | ❌ 变短本身不打掉 |

**⇒ 归零的真实原因还没排干净，但已经确定三件事：**
1. **prompt 结构是对的**：thoughts 块逐字节只追加（前 279 字符在多轮里完全一致）、
   计划日志恒定、tools 的 sha 在整个会话里不变。
2. **命中量 ≈ 上一轮的公共前缀**，且**滞后一轮**（同一个请求连发两次：第一次 128、
   第二次 186、第三次才 100% —— 缓存写入是滞后的）。
3. **归零与「本轮变短」强相关**（真机 2/2）。

> ⚠️ **但 3 只是相关，不是因果**：上面第三行已经证伪了「变短 → 归零」这条因果链。
> 8.5 节用三组对照给出了真正的原因 —— **易变尾部太大**（尾部里 92% 是每轮发的截图，
> 把可缓存前缀的比例压到 ~25%）。**「变短」只是这个比例失衡在某些轮次上的表现。**

### 8.3 真正的天花板：易变尾部里 92% 是每轮都发的那张截图

cortex 的 prompt 拆开是：

```
[system][head][thoughts][计划日志][status][failed][feedback][设备状态][感知块]
└────── 稳定、只增长（可缓存）──────┘└──── 每轮变、长度剧烈波动（不可缓存）────┘
```

真机抓到的块长度：

| 块 | 长度 | 每轮变? | 可缓存? |
|---|---:|---|---|
| system | 6,938 | ✗ | ✅ |
| head（设备+目标） | 180 | ✗ | ✅ |
| thoughts | 279 → 5,662 | 只增长 | ✅ |
| 计划日志 | 413 | ✗（无 replan） | ✅ |
| subgoal status | 69 | ✗（本轮没变） | ✅ |
| executor feedback | 24-334 | ✓ | ❌ |
| device state | 83 | ✓ | ❌ |
| **Screen elements 文本**（层级树 + OCR 合并表） | **2,383** | ✓ | ❌ |
| **截图 base64**（640×1536 JPEG） | **47,672** | ✓ | ❌ |

> ⚠️ 之前这份表把「感知块（层级树 + OCR）」写成 5,125-49,164 字符、占 70-80% ——
> **那是错的**：那 47K 字符是**截图的 base64**，不是文本。屏幕元素文本只有 2,383 字符。

**⇒ 真实构成：易变尾部里 92% 是那张每轮都发的图。**
**⇒ 「缩短尾部」的真实含义 = 「别每轮发图」= 图片按需取（见 8.6）。**
**⇒ 而且 base64 长度随画面变 → 请求总长每轮都变 → 上一轮建的缓存单元匹配不上 → 归零。**
**⇒ 这一下把 8.1 的「归零」和 8.3 的「天花板」串起来了：同一个原因。**

**⇒ 易变尾部里 92% 是那张每轮都发的截图（47,672 字符的 base64）**，
而且它的长度**随画面变化** → 请求总长每轮都变 → 上一轮建的缓存单元匹配不上 → 归零。

**⇒ ⇒⇒ 这就是命中率的物理上限：**
> **可缓存前缀 / 整个 prompt ≈ 15%**
> （可缓存 ≈ 8,631 字符的稳定增长部分，整个 prompt ≈ 58,000 字符）

**⇒ 44.9% 的实测命中率，对于「感知块占 70%」的架构来说，已经接近上限了。**
**⇒ 想再往上走，杠杆不在「重排块序」（那块已经做对了），而在「缩小易变尾部」。**

### 8.4 和 DeepSeek Harness 的差距在哪

DeepSeek Harness 能到 99%+，不是因为它的分层更聪明，而是因为**它的 prompt 里没有
巨大的易变尾部** —— 它用**工具**去读屏幕（`read_file` / `list_dir` 返回的是短片段），
而不是把整棵层级树 + OCR 塞进每一轮 prompt。

**⇒ ⇒⇒ 结论：mobile-use 剩下最大的缓存杠杆是「感知块瘦身」，不是继续调块序。**
已有的分析（`doc/token-context-analysis.md` §八）已经列了同方向的候选：
工具 schema 结构手术（−716/次）、截图降分辨率（−75% 截图成本）、
`executor_messages` 截断。**感知块本身占多大、砍掉它能换多少命中率，这一项之前没有量化过 —— 8.5 节量化了。**

### 8.5 三组对照：决定性实验

同一批真实 prompt，三种形状各跑 19 轮（`probe_out/probe_fixed_slot.py`）：

| 形状 | prompt 总量 | 命中率 | 归零轮次 |
|---|---:|---:|---:|
| A) 原样（完整 prompt） | 31,796-38,906 | **19.7%** | 2/19 |
| B) **固定长度尾部**（截断/填充到定长，内容照旧每轮变） | 13,937-17,976 | **21.5%** | 3/19 |
| C) **缩短尾部**（只留前 4 块：system+head+thoughts+计划日志） | 2,172-6,263 | **92.6%** | **0/19** |

**⇒ 两条结论：**

**① 「让请求总长恒定」这个修法被实测证伪。**
它的原理假设是「请求变短 → 上一轮建在用户输入末尾的缓存单元匹配不上 → 归零」，
但这个假设本身有反证（`P → P_short` 之后仍命中 99.4%，见 8.2）。
三组对照直接判了它：**固定长度几乎没用**（21.5% vs 19.7%，归零轮数反而更多）。

**② 只有「缩短尾部」有效**：92.6%，归零 0/19。

**⇒ 为什么？** 因为真正决定命中的是**可缓存前缀占整个 prompt 的比例**：
- 固定长度**既不改变这个比例**，也**不改变「尾部内容每轮都变」这个事实** → 白做。
- 缩短尾部把比例从 ~25% 直接拉到 ~100% → 立刻见效。

**⇒ ⇒⇒ 教训：别去优化与命中无关的变量。**「长度恒定」看着很对症（provider 的
规则里确实有「完全匹配」），但它作用在了一个**不是决定因素**的量上。

> 📌 另一组更早的重放（`probe_out/probe_trim_tail.py`）报的是「完整 66.9% vs
> 缩短 94.6%」——**那组偏乐观，别引用**：B 的 prompt 是 A 的严格前缀，共享了上一轮
> 刚写入的缓存单元，所以 B 第一轮就命中 94.3%；A 场景里那些 100% 的轮次同样是
> 「这一轮的 prompt 之前被原样发过」。**能确定的是方向和归零频率，不是绝对值。**

### 8.6 落地建议（只有证据支持的方向）

**唯一有证据的方向：把感知块做小。** 两条路：

1. **图片按需取**（推荐）—— 默认不发图，UI+OCR 决策不了时由 cortex 自己调工具取图。
   一举两得：省掉每轮的图 token，**同时消除 base64 长度波动带来的归零**。
   风险：改变 cortex 的交互方式（多了 tool call）→ **需要真机 A/B**。
2. **屏幕元素文本瘦身** —— 层级树只给顶层/可见元素、OCR 只给相关区域。
   收益比想象中小（它只有 2,383 字符），但顺带减少波动。

`doc/token-context-analysis.md` §八里已有的 **④截图降分辨率（−75%）**、
**②工具 schema 结构手术（−716/次）**、**⑤`executor_messages` 截断** 方向一致且风险更低，
可以先排。**注意 `agents_thoughts` 裁剪是反方向的** —— 它砍的是**可缓存前缀**，
命中率会涨但总 token 不一定降，要单独权衡。

---

## 9. 落地实现（2026-10-21）

### 9.1 ② 工具 schema 结构手术

**新增** `minitap/mobile_use/utils/tool_schema_surgery.py`：
- `surgery_tool_schema()` —— 递归删 `title`、把 `anyOf: [X, null]` 折成 `X`
  （**保留 `default: null`**，它是「可省略」的信号）
- `surgery_tool_schemas()` —— 处理 OpenAI 风格的 `tools` 数组，认不出的形状原样放过
- `apply_tool_schema_surgery()` —— 就地改请求 body 里的 `tools`

**接线**：`minitap/mobile_use/services/llm.py` 新增 `ToolSchemaSurgeryMixin`，
在 `_get_request_payload` 之后改 body —— 那是 langchain 把**所有** `bind_tools`
汇成请求体的唯一出口。一处覆盖全部 OpenAI 兼容 provider
（minimax / deepseek / openai / openrouter / xai / minitap），**对工具实现零侵入**。

**实测收益**：executor 的 16 个工具 **4,333 → 3,871 token（−462，−10.7%）**。
只有 5 个用 `Target` 模型的工具有收益（各 −113）。

> ⚠️ `doc/token-context-analysis.md` §5.2 写的 **−716（−20.1%）是旧版工具的数据**，
> 现在的真实值是 **−462**。那份文档的结论过期了。

**真实 tool-call 验证**（`probe_out/probe_schema_toolcall_validation.py`、
`probe_out/probe_schema_null_acceptance.py`）：
- 手术前后模型返回的 tool_call 参数**完全一致**（`{"index": 1}`），`Target` 校验通过
- **显式 null 的参数**：手术前后都是 HTTP 200 ⇒ **provider 不校验 tool_call 参数
  的 schema**，`anyOf` 折叠不会因为「模型传了 null」被拒

**单测**：`tests/mobile_use/test_tool_schema_surgery.py`（8 条），其中一条锁住
「语义描述一个都不能少」。

### 9.2 ③ 图片按需取

**问题**：截图是整个 prompt 里最贵的一块（实测 47,672 字符 base64），而且
**永远不可缓存** —— 它在 prompt 末尾，base64 长度随画面变 → 请求总长每轮都变 →
provider 建在「用户输入末尾」的缓存单元匹配不上 → 归零。

**改法**（不是加 tool-call 循环，而是给结构化输出加一个字段）：

1. `CortexOutput` 新增 `need_screenshot: bool`（默认 False）——
   模型说「元素表和 OCR 不够用」时置 true。
2. `PerceptionConfig.screenshot` 改成**三态**：
   `True`（每轮都带，老行为）/ `False`（不带）/ `"on_demand"`（**默认**，按需）。
3. `cortex.py`：
   - 按需模式下**不带图**地调用 LLM；
   - 模型回 `need_screenshot=True` 且**没给出动作**时 → 把当屏截图追加到
     messages 末尾 → **同一轮再问一次**（第二次复用现有的「带图 prompt」路径，
     不新增任何调用机制）。
   - 判断逻辑抽成纯函数 `should_request_screenshot()`，可单测。
4. `cortex.md` / `cortex.zh.md`：
   - 通道状态行改成 `{{ perception.screenshot_label }}`（`"on_demand"` 是真值，
     老写法会显示成 `enabled`，模型就以为图已经在手里了）
   - Perception 段说明「按需」怎么用
   - Output 表加 `**Need Screenshot**` 字段

**为什么不做成 tool-call 循环**：结构化输出路径强制 `tool_choice`（那是
`_ToolCallStructuredRunnable` 存在的原因），加第二个工具就得放弃强制，
解析可靠性会掉；而且图片塞进 tool message 还要 provider 支持。
**加一个输出字段**既保住了强制的结构化输出，又复用了现成的「带图 prompt」代码路径。

**单测**：`tests/mobile_use/test_screenshot_on_demand.py`（17 条）——
三态解析、`screenshot_label`、补图判断的四个条件、prompt 在三种模式下的渲染。

### 9.3 ④ 截图降分辨率 —— **早就做了**

`utils/image_compress.py` 的 `compress_screenshot_base64` 默认带
`max_width=640 / max_height=1536`（`DEFAULT_MAX_WIDTH/HEIGHT`），
截图早就缩过了。**是 `doc/token-context-analysis.md` §5.3 的结论过期了**
（那里写「仅执行 quality=50 的 JPEG 重编码，无任何 resize」，那是 10-16 的状态，
10-17 的 `78bc8a2` 就实现了缩放）。

**⇒ 这一项不需要动代码，只需要把文档的过期结论改掉。**

### 9.4 manager 侧

`runner-host/src/runner_host/host.py` 的 `_to_mobile_use_perception()`：
契约里 `perceptionScreenshot` 是 bool，**映射成 `"on_demand"`**。

**契约和 Web UI 都不用改**：那个复选框的语义仍是「截图这条通道可用」，
只是改成**模型判断需要时才取**。

### 9.5 真机 A/B 最终结果

**基线**（每轮都发图，`screenshot=True`）vs **on_demand**（按需），同一个 4 子目标任务，
Redmi Turbo 4 / Android 16 / MiniMax-M3.1-Flash-Preview：

| 指标 | 每轮都发图（基线） | on_demand（按需） |
|---|---:|---:|
| 整体命中率 | 44.9% | **74.6%** |
| cortex 命中率 | 48.6% | **76.4%** |
| 每次调用 input | 6,896 | **4,443（-36%）** |
| prompt 体积 | ~58,000 字符 | **~11,661 字符（-80%）** |
| 带图的调用占比 | 100% | **2/30（6.7%）** |

**⇒ 命中率的涨幅主要来自「图不再每轮进 prompt」**：
可缓存前缀（system + head + thoughts + 计划日志 + status）在总 prompt 里的占比
从 ~15% 升到 ~85%。

### 9.6 ⚠️ 踩到并修掉的一个真 bug：「我要图」被当成空响应

**症状**：第一次 on_demand 真机跑，**51 步没跑完**、屏幕黑、总 input 反而比基线高 36%。

**根因（两层）**：
1. `_invoke_cortex` 的空响应校验在**按需分支之前**：

   ```python
   if (response.decisions is None and response.decisions_reason is None
       and response.goals_completion_reason is None and not response.complete_subgoals_by_ids):
       raise ValueError("Cortex LLM returned an empty response")
   ```

   模型说「我要图」时（只置了 `need_screenshot`，没给决策也没给理由），
   **这一行直接抛错** → 整轮失败 → agent 原地白烧步数。
   **修法**：空响应校验加上 `and not response.need_screenshot` ——
   「我要图」是**合法应答**，不是错误。

2. **手机 30 分钟息屏超时**（`screen_off_timeout=1800000`）→ 屏一黑 agent 就点不动，
   在黑屏上反复尝试。**这一步把 A/B 的结论完全带偏了**（51 步、24 步都是它造成的，
   不是 on_demand 的锅）。

   **⇒ 真机长跑必须先锁常亮**：
   ```powershell
   adb shell svc power stayon true
   adb shell settings put system screen_off_timeout 2147483647
   ```
   **⇒ 这条是给以后所有真机 A/B 的前置条件** —— 不锁的话，测出来的「步数暴涨」
   会被误判成代码回归。

**修完之后的干净对照**（同一个 1 子目标任务，屏幕常亮）：

| | 基线（每轮发图） | on_demand |
|---|---:|---:|
| 任务结果 | success | **success** |
| LLM 调用数 | 17 | **4** |
| cortex 命中率 | 48.6% | **74.1% / 60.8%** |
| cortex input | 5,007-10,305 | **3,456-4,212** |
| 模型要图的次数 | —（每轮都有图） | **0**（元素表+OCR 已足够） |

**⇒ 结论：on_demand 是净赢，保留为默认。**

> 📌 「模型要不要图」这件事本身是健康的：整个 run 里 `need_screenshot` 全是 `False`
> —— 说明**大多数决策靠元素表 + OCR 就够**，图只在颜色/图标/遮挡这些地方才需要。
> 这也验证了「图按需取」的方向，而不是「图根本没用」。

### 9.7 一个差点让我误判的测试假象

单独拿真实 prompt 问模型时，它**只回 `reasoning_content`、不调 tool**，
看起来像「schema 手术把模型搞坏了」。

**真因是我的测试脚本把 `max_tokens` 设成了 512** —— M3.1 的 thinking 模式
没说完就截断了，自然发不出 tool call。生产代码不设 `max_tokens`，没这个问题。

**⇒ 用真实端点做对照实验时，先确认自己的探针没有截断模型的输出。**
（原始 schema 和手术后 schema 在同一条件下**都不调工具** —— 这才是对照的意义：
两个都坏，说明是探针的问题，不是被测对象的问题。）

---

## 10. 方法论沉淀（给以后省时间）

1. **先证「请求确定」再谈缓存**。连发两次同一个 body，比 provider 报的
   `prompt_tokens` 是否相同。不相同说明有隐藏变量（时间戳、随机 ID、工具顺序），
   后面全白搭。
2. **先证「provider 会缓存」再谈「哪里击穿」**。同一个请求发两次，看
   `usage.prompt_tokens_details.cached_tokens`。
3. **用 provider 回报的 cached 值，不要自己模拟**。模拟只能证明「我的分层设计
   没写错」，证明不了「provider 真命中了」。`probe_thoughts_compaction_e2e.py`
   就是反例。
4. **逐块 A/B 定位击穿点**。只动一个块，看 cached 塌到多少，就知道那个块在
   缓存键的第几层。E3/E4 都是这么定位的。
5. **先量噪声底，再谈效应量**。12 轮序列的噪声底是 ±2,000 token，效应量 1,440
   就分辨不出来。**这时候要换成单发对照（同序列、只差一个块、背靠背）**，
   而不是多跑几轮碰运气。
6. **重复实验要检查"两次请求是否逐字节相同"**。本次 12 轮 A/B 里第 2/3 轮的
   "replan 请求"与第 1 轮完全相同 → 缓存已命中 → 数据无效。
7. **缓存滞后一轮**，所以单次请求的命中率不是稳态值。要看一串请求的累计。
8. **归档 usage 永远是「跑那一版代码」的数**。exec 212 的 10.4% 是 `9af5a62` 的数，
   不是现在这版的数——引用时必须带时间戳。
9. **改提示词块序之前，先量「那个块真的会变吗」**。P1 的判断差点做反：exec 212
   里 replan 发生了 **0 次**，计划日志全程恒定。只看「replan 时会击穿」就下结论，
   会把常见路径的收益算反。
10. **遥测恒为 0 时，先怀疑「数据压根没进来」，别急着加钩子**。这次的根因是
    `stream_usage` 没开 —— provider 报了 usage，langchain 根本没转发。
    走过的三条弯路（类打补丁 + ContextVar / `on_llm_end` 回调 / `messages` 流 chunk）
    都在**症状层**打转。**正确顺序：先确认「provider 到底报没报」**
    （`manager-host/runner-host/llm_usage_patch.py` 的开头就是那份实测记录），
    再决定要不要加钩子。
11. **同仓库里别人已经踩过的坑，先读他们的记录**。manager 侧
    `llm_usage_patch.py` 的 docstring 把「非流式有 usage / 流式没有」写成了一张表 ——
    这张表就是答案。**在一个仓库里干活，先把同类实现找出来读一遍。**
