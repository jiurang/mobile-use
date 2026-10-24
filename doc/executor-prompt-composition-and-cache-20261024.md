# Executor 提示词组成 与 前缀缓存命中率（2026-10-24）

> 问：Executor 的提示词由什么组成？为什么缓存命中率低？
> 对照：DeepSeek 官方的缓存规则 + Hermes（Nous Research）的提示词模板。
>
> 复现：
> - `scripts/probe_executor_cache_measure.py`（组成表 + 真 API 命中率）
> - `scripts/probe_executor_cache_measure.py --offline`（前缀结构，不打 API）
> - `scripts/probe_real_element_table_size.py`（真机元素表体积）
> - 原始输出：`probe_out/executor_cache_measure_20261024.txt`
>
> ⚠️ 本文所有命中率都来自**真 API 的 `usage`**（MiniMax-M3.1-Flash-Preview，
> 与 `llm-config.override.jsonc` 的 executor 配置一致）。离线那部分只用来
> 「定位断在哪个块」，不当命中率结论 —— 这是 `doc/cache-hit-rate-audit-20261020.md`
> 留下的教训（93.5% 那个数是重复字符拼出来的模拟值）。

---

## 一、一句话结论

**Executor 的提示词布局是对的（顺序值 20.9 个百分点），命中率低不是布局问题，
是「易变尾巴」太大：真实屏幕的元素表平均 1,150 token，占整份 prompt 的四成，
它每轮都变 ⇒ 永远满价。**

> 命中率 = 公共前缀 ÷ 总长。公共前缀 = system + 目标 + 局部历史。
> 离线算出的比例和真 API 实测吻合（第 8 轮：离线 62.9% / 实测 64.7%）。

---

## 二、提示词组成（走生产代码 `build_executor_messages` 渲染）

| # | 块 | 角色 | 内容 | 作用 | 每轮变？ | 体积 |
|---|-----|------|------|------|---------|------|
| 0 | system | system | `executor.md` 的 `system_message` 宏 | 身份 + 循环 + **文本工具表** + 坐标/滑动语义 + 感知说明 + 输出契约 | ❌ 不变（任务内） | 4,238 字符 / **1,140 tok**（zh 版 1,203 tok） |
| 1 | 目标 | user | `goal_message` | 本子目标 | ⚠️ 换子目标才变 | 38 字符 / 16 tok |
| 2..k | 局部历史 | assistant + tool + user 三段一组 | `state.executor_messages` | 动作 → 工具回执 → **「上一步的结果」** | ✅ **只追加** | 每步 ~290 字符 |
| k+1 | 失败账本 | user | `failures_message` | 已失败目标，别再复用 | ⚠️ 增长时变 | 小 |
| k+2 | 设备状态 | user | `device_state_message` | 日期 / 前台 app / 息屏 / 无感知通道 | ✅ 每轮变 | 66 字符 |
| k+3 | **屏幕元素表** | user | `screen_elements_message` / `ocr_message` | 当屏的唯一定位依据 | ✅ **每轮变（重编号）** | **真机 631~1,475 tok** + 表头 96 tok |

实测的第一轮组成（60 行表）：

```
  块  角色      字符数  每轮变?  内容预览
   0  system     4,238  不变    # You are the **Executor** ...
   1  user          38  不变    Your goal: 进入 WLAN 页面并读出当前连接的 Wi-Fi 名称
   2  assistant    201  不变    [{"id": "call_00", ...}]
   3  tool          40  不变    Tapped element (610, 1258) successfully.
   4  user          47  不变    [上一步的结果] 前台应用没变（还是 com.example.app）；...
   ...                              ← 历史每步追加三块
  11  user          66  变      Device date: ... Focused app info: ...
  12  user       2,944  变      Screen elements, one per line: [i] "text" | id=...
```

**顺序的意图**：前缀缓存只认「从头开始的连续前缀」。所以**每轮变的东西一律靠后**，
把局部历史夹在中间 —— 历史只追加 ⇒ 它能被缓存覆盖；元素表每轮重编号 ⇒ 放最后，
它之前的全部命中，它自己满价。

---

## 三、真 API 实测

### 3.1 前提校验

| 实验 | 结果 |
|---|---|
| 同一请求连发两次 | 3,750 tok → cached 3,584（**95.6%**）⇒ provider 缓存本身没问题 |
| 同一请求第二次**整段**命中 | ⇒ provider 会把「用户输入末尾」固化成一个可复用单元（见 §四） |

⚠️ **这条也是坑**：同一个请求第二次发会整段命中，所以**上一组序列跑过的请求
会污染下一组的首轮**。本轮所有序列都用**唯一的目标文本**冲刷缓存，各组互不污染
（第一版探针没做这件事，数据是脏的，已作废）。

### 3.2 A/B：顺序值多少

只改顺序、内容一字不动（60 行表，8 轮）：

| 布局 | 命中率 | 满价重算 |
|---|---|---|
| **生产布局**（元素表在最后） | **48.9%** | 15,644 |
| 旧布局（元素表在局部历史之前） | **28.0%** | 22,039 |

⇒ **顺序值 +20.9 个百分点**。旧布局第 2~4 轮直接掉到 128~187 tok（3~5%），
第 5~7 轮因为「公共前缀」被单独固化才回到 2,048，第 8 轮又掉回 186。

### 3.3 元素表体积 = 命中率的死穴

同一布局，只改元素表行数（每组 6 轮）：

| 元素表行数 | 总命中率 | 稳态（第 3~6 轮） |
|---:|---:|---|
| 0（无表） | 68.4% | **92.6% → 93.7%** |
| 20 | 55.4% | 75.5% → 78.7% |
| 60 | 43.1% | 57.9% → 62.2% |
| 120 | 31.9% | 43.0% → 47.4% |
| 200 | 23.9% | 32.0% → 35.9% |

**几乎线性反比。** 元素表越大，命中率越低 —— 因为它是分母，不是分子。

### 3.4 真机上尾巴到底多大

`probe_real_element_table_size.py`，Redmi Turbo 4 / Android 16：

| 屏幕 | 原始节点 | 入表行数 | 表 token |
|---|---:|---:|---:|
| home | 64 | 64 | 1,233 |
| recents | 34 | 34 | 631 |
| settings | 83 | 83 | 1,475 |
| notif | 51 | 51 | 938 |

平均 **≈ 1,150 tok**（表头固定 96 tok）⇒ 换算成命中率天花板
（system+goal ≈ 1,156 tok 为固定前缀）：

| 历史长度 | 命中率天花板 |
|---|---:|
| 0 tok | 46.9% |
| 500 tok | 56.2% |
| 1,500 tok | 67.6% |
| 3,000 tok | 76.7% |

### 3.5 冷启动：每个序列前两轮几乎不命中

所有序列的第 1、2 轮 cached 都只有 128~187 tok（2~9%），第 3 轮才跳到稳态。
**换子目标 = 冷启动**（目标文本一换，前缀单元就变），成本 ≈ system + goal ≈ 1,156 tok，
再加上这两轮的预热。子目标越短，这笔固定成本摊得越薄。

---

## 四、DeepSeek 官方的缓存规则（为什么是这个形状）

来源：<https://api-docs.deepseek.com/guides/kv_cache>

- 磁盘缓存默认开启；后续请求只要**前缀重叠**，重叠部分直接读缓存。
- **滑动窗口注意力**下，缓存以**独立的完整单元**存储 ——
  「后续请求必须**完全匹配**某个 cache prefix unit 才算命中」。
- 前缀什么时候被固化，三种：
  1. **请求边界**：每个请求在「用户输入末尾」和「模型输出末尾」各产生一个单元；
  2. **公共前缀检测**：多次请求的公共部分被固化成独立单元；
  3. **固定 token 间隔**：长输入按固定间隔切单元，避免长前缀永远到不了结束位置。
- usage 字段：`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`。
- best-effort，不保证 100%；构建要几秒；几小时~几天自动清理。

**这三条规则把实测的三个现象全解释了**：

| 实测现象 | 规则给出的解释 |
|---|---|
| 尾巴（元素表）永远满价 | 它在每个请求的**末尾** ⇒ 固化出来的单元每轮都不同 ⇒ 永远匹配不上 |
| 前两轮冷 | 公共前缀单元要**下一次请求**才被固化（规则 1/2） |
| 同一请求第二次发 → 95.6% | 整段请求在「用户输入末尾」被固化成**一个完整单元**，第二次完全匹配 |

⇒ **可操作的一句话**：缓存能命中的，只有「上一轮也一模一样的那段前缀」。
凡是每轮都会变的，放哪儿都没用 —— 放最后至少不连累别人。

---

## 五、Hermes 的提示词设计（Nous Research 官方模板）

来源：<https://github.com/NousResearch/Hermes-Function-Calling>（README 原文）

```
<|im_start|>system
You are a function calling AI model. You are provided with function signatures within
<tools></tools> XML tags. ... Here are the available tools: <tools> [ ...JSON... ] </tools>
Use the following pydantic model json schema for each tool call you will make:
{"title": "FunctionCall", "type": "object", ...}
For each function call return a json object with function name and arguments within
<tool_call></tool_call> XML tags as follows: ...
<|im_end|>
<|im_start|>user
Fetch the stock fundamentals data for Tesla (TSLA)<|im_end|>
<|im_start|>assistant
<tool_call>{"name": "...", "arguments": {...}}</tool_call><|im_end|>
<|im_start|>tool
<tool_response>{"name": "...", "content": {...}}</tool_response><|im_end|>
```

四个设计要点，和 mobile-use 的对应关系：

| Hermes 的做法 | mobile-use 现状 |
|---|---|
| **静态契约全部前置**（身份 + 工具定义 + 输出 schema + 格式约定，都在 system 段） | ✅ 已做：system 段承载身份/工具表/输出契约，全静态 1,140 tok |
| **工具定义走「文本 + 标签」**，不用 provider 的 tools 数组 | ✅ 已做：只绑 `ExecutorOutput` 一个 schema；16 个设备工具用 **365 tok 文本表**教（绑全量 schema 是 4,008 tok 且不进缓存键） |
| **推理写在自己的回合里**（Hermes 3 的 `<scratch_pad>` GOAP 在 assistant 回合**内部** = 历史的一部分，不是每轮重塞 prompt 头部） | ✅ 已做：`agent_thought` 字段（"先说上一步的结果，再说明为什么这个动作"）= scratch_pad 的字段化版本，天然追加式 |
| **工具结果原样回灌**（`<tool_response>` 放完整 JSON） | ✅ 已做：错误逐字保留，只有过长的成功结果截断到 120 tok |

⇒ **结论：Executor 的提示词设计与 Hermes 是同构的，而且已经踩对了 Hermes 的那几个关键点。**
差距不在「怎么组织」，在**易变内容有多大**。

---

## 六、能做什么（按性价比排序）

1. **缩尾巴** —— 唯一真正有效的杠杆。实测 60 行 → 20 行，稳态命中率 62% → 78%。
   真机平均尾巴 1,150 tok，可下手的地方：
   - 只发**可交互或有文字**的行（`scripts/verify_ui_prune.py` 已有裁剪器，
     且已验证「可交互节点召回率必须 100%」这条底线）；
   - 压行格式：`[i] "text" | id=… | Class | (x,y,WxH) | flags` 里，`Class` 在多数行
     是冗余的（flags 已经说明可交互性）；
   - 相同 id 的多行合并成一行 + 坐标列表。
2. **增前缀** —— 历史越长命中率越高（0 → 1,500 tok：47% → 68%）。
   ⇒ 子目标别切太碎（换子目标 = 冷启动）。
3. **减少 Executor 调用次数** —— 它是调用量最大的 agent，一轮一个动作 ⇒ 每轮重发整份 prompt。
4. **别动顺序** —— 已实测值 20.9 个百分点，任何把易变内容插到历史前面的改动都是净亏。

---

## 七、本次落地：「失败事实」从尾部独立块 → 并入局部历史（append-only 流）

### 7.1 为什么改这一块

失败账本（`executor_failures`）是**跨子目标累积**的 append-only 账本，但它在提示词里是
**尾部的一个独立块**（`[这些目标在本任务里已经失败过：…]`）⇒ **每轮都要全价重发**，
而且它排在元素表**前面** ⇒ 账本一变，后面所有内容全部重算。

实测账本体积（`scripts/probe_executor_merged_append.py`）：

| 失败条数 | 账本块大小 | 每轮重发 |
|---|---|---|
| 0 | 0 字符 | — |
| 4 | 366 字符 | 全价 |
| 12 | 922 字符 | 全价 |
| 24 | 1,766 字符 | 全价 |

### 7.2 改法（三条决策）

1. **事实只追加，不下结论**（沿用既有纪律：`contextor.py:90`「只记录结果」、
   `executor-zh.md:69/73`）。失败行写成
   `"点赞" like_button 这一步失败了` ——
   **只说「这一步失败了 + 是谁」**，不说「不要再用这个目标」「这个目标无效」。
   「这个动作没用」是**相对于目标**的判断，框架不知道目标，做不了这个判断。
2. **并进 Contextor，不新开一条 user 消息** ——
   失败事实落在**产生它的那一步旁边**，模型读历史时不用跨块对账；
   `[上一步的结果]` 那一行本来就 append-only，追加它**几乎免费**。
3. **索引去掉裸 index** —— 元素索引每轮重编号，带上会让模型点错元素。
   新增 `describe_target(label, resource_id)`（无 index）专供 Executor；
   Reporter 的 `describe()`（带 index、逐字搬运契约）**保持不动**。

落地后的提示词形态（`scripts/render_executor_prompt_actual.py`，走生产代码渲染）：

```
[HumanMessage] [上一步的结果] "点赞" like_button 这一步失败了；前台应用没变（还是 com.android.settings）；屏幕内容没变
[HumanMessage] 这些目标在本任务里已经失败过："点赞" like_button (no element with resource_id on the current screen)
```

跨子目标的**索引块保留原位**（历史之后），只改措辞与去 index ——
因为局部历史**每换子目标就清空**，失败事实放进历史会丢；账本只增不减、跨子目标。

### 7.3 实测收益（真 API，MiniMax-M3.1-Flash-Preview，8 轮，60 行元素表）

| 失败条数 | 现状（独立块） | 提案（并进历史） | 差 |
|---|---|---|---|
| 4 条 | 49.6% | **56.9%** | +7.4 点（8 轮少烧 2,335 tok） |
| 12 条 | 45.3% | **52.0%** | +6.8 点（8 轮少烧 2,089 tok） |

复现：`.venv\Scripts\python.exe scripts\probe_executor_merged_append.py`

### 7.4 改了哪些文件

| 文件 | 改动 |
|---|---|
| `utils/failed_targets.py` | 新增 `describe_target(label, resource_id)`（不带 index） |
| `graph/state.py` | 新增瞬时字段 `last_action_failure`（`take_last`，默认 None） |
| `tools/utils.py` | 新增 `last_action_failure_fact(target)`（原始数据） |
| `tools/mobile/tap.py` / `long_press_on.py` / `focus_and_input_text.py` / `focus_and_clear_text.py` | 成功→清空；失败→写事实；熔断分支→清空（熔断=没执行，不是试了但失败） |
| `agents/contextor/contextor.py` | `render_action_result` 把失败事实作为第 0 段并进 `[上一步的结果]`；读完即清 |
| `agents/executor/prompt.py` | `render_failures_message` 改用 `describe_target`（去 index） |
| `agents/executor/executor-zh.md` / `executor.md` | 失败块措辞改中性（中文/英文同步） |

**没动的**：元素表（当前屏，追加会堆旧屏幕）、设备状态（状态就该在尾部替换）、
缓存布局顺序（已实测值 20.9 个百分点）、失败条数上限（用户明确拒绝「加限制」）。

### 7.5 真机验证（Redmi Turbo 4 / Android 16 / HyperOS）

脚本：`scripts/probe_executor_failure_fact_e2e.py`
跑法：`.venv\Scripts\python.exe scripts\probe_executor_failure_fact_e2e.py`

**做法**：在**真机**上跑完整图（真 adb、真控制器、真 Contextor、真 prompt 渲染），
只把模型的**第一个动作**换成一次注定失败的点击 ⇒ 失败是**真的**
（真报错、真回执，不是模拟值）。之后交回真模型自己跑完整个任务。

**断言 1：失败事实进了 prompt**（第 2 轮真机跑，抓到的真实 prompt 原文）：

```
[上一步的结果] "探针不存在的元素" definitely_not_here 这一步失败了；
前台应用没变（还是 com.android.settings）；屏幕内容没变
```

- ✔ **没有裸 index** —— 描述部分只有「可读文字 + resource-id 尾段」
  （账本里的 `reason` 是工具**逐字**报错原文，含 `(index=0)`，那是事实不是框架结论）
- ✔ **没有结论词** —— 没有「不要再用 / 无效 / 没用 / 换个」这类判断

**断言 2：失败事实能跨子目标存活**（第 3 轮真机跑，29+ 轮 prompt 全程在场）：

```
[账本] Targets that already failed in this task: "探针不存在的元素" definitely_not_here (…); "搜索系统设置项" input (…); "高级设置" title (…)
```

- 局部历史（`[上一步的结果]`）**换子目标就清空** —— 这是设计如此，
  所以跨子目标靠**失败索引块**兜底；两条路都在，同一条失败事实不会漏。

**断言 3：真实场景（不是探针注入）同样工作** —— 第 3 轮一共抓到 **3 条真实失败**，
其中 2 条是**模型自己点错**（滚动把元素带走了），不是探针造的：

| # | 失败目标 | 来源 |
|---|---|---|
| 1 | `"探针不存在的元素" definitely_not_here` | 探针注入（第一次动作） |
| 2 | `"搜索系统设置项" input` | **模型自己点错**（`index=15` 已被滚动带走） |
| 3 | `"高级设置" title` | **模型自己点错**（列表刷新后元素不在原位） |

三条在局部历史里各占一行、在账本里各占一条 —— **同一个失败事实，两条路都在**：

```
[上一步的结果] "搜索系统设置项" input 这一步失败了；前台应用没变（还是 com.android.settings）；屏幕上新出现了：V…
[上一步的结果] "高级设置" title 这一步失败了；前台应用没变（还是 com.android.settings）；屏幕上新出现了：HIKVISION_BB3650,WLAN 信号强度为一格。,安全网络…
```

⇒ 链路不是只在「人造失败」下成立，**模型自己点错时走的是同一条路**。

**缓存命中率（真实 API usage，executor 逐轮）** —— 真机两轮跑：

**第 3 轮（34 次 executor 调用，长任务同子目标内连续跑）**：

| 轮 | input | cached | 命中率 | 说明 |
|---|---|---|---|---|
| 1 | 4,757 | 128 | 2.7% | 冷启动（失败事实那一轮） |
| 2 | 3,914 | 2,289 | 58.5% | 进稳态 |
| 3~26 | 4.4k→9.8k | 2.5k→9.1k | 57.3% → **92.6%** | 稳态，且**随历史变长一路涨** |
| 27 | 10,025 | 185 | 1.8% | 单轮抖动（缓存未命中，非边界） |
| 28 | 10,392 | 9,557 | 92.0% | 回到稳态 |
| 29~30 | 12.7k | 10.0k | 76.4% / 78.5% | 稳态 |
| **31~32** | 4.4k | **0** | **0%** | **换子目标 → 缓存归零** |
| 33 | 4,698 | 128 | 2.7% | 冷启动 |
| 34 | 4,948 | 2,320 | 46.9% | 进稳态 |

合计 **250,060 input / 168,955 cached = 67.6%**（全部 44 个 agent 调用：312,869 / 176,053 = 56.3%）。

**第 2 轮（8 次 executor 调用，3 次子目标切换）**：41,324 / 12,221 = **29.6%**。

⚠️ **这两个数都不是「新设计的命中率」**，是**真实任务的端到端值**，差别全在
「换了几次子目标」：

| 跑法 | 子目标切换次数 | executor 命中率 |
|---|---|---|
| 第 3 轮（长任务，同子目标内连跑 30 轮） | 1 次 | **67.6%** |
| 第 2 轮（3 个子目标，频繁切换） | 2 次 | 29.6% |

⇒ 真机数据把「**换子目标 = 缓存冷启动**」坐实了：第 31/32 轮 `cached` 直接 **归零**
（不是掉到 4%，是 0 —— 新子目标的 prompt 跟旧的前缀一点重合都没有）。
这跟 7.3 里「历史越长命中率越高」是同一条规律，只是子目标边界比「多聊一轮」重得多。

⇒ **「别切太碎」这条从「推测」升级成「实测」**：同样一件事，子目标内连续跑 30 轮
能到 67.6%，每换一个子目标就付一次冷启动。

---

## 八、一个待验证的方向（没做）

DeepSeek 规则 3「固定 token 间隔切单元」意味着：**如果请求足够长，
中间位置也可能存在可复用单元**。当前布局下中间都是历史（每轮变），
所以用不上。但如果将来想「把元素表留在原地、只发增量」，
需要先确认 provider 的固定间隔切点在哪 —— 这条本轮**没有实测**，不作为结论。
