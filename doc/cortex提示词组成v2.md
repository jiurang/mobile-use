# cortex 提示词组成（v2）—— 面向前缀缓存的分层重排与质疑

> 整理日期：2026-10-19
> 前置：[提示词组成v1.md](提示词组成v1.md)（现状 23 块）、[token-context-analysis.md](token-context-analysis.md)（各 agent token 实测）
> 目标：把「请求与上一轮请求的最长公共前缀」最大化，从而让 provider 的前缀缓存命中。
> 本文分两部分：**一、对你的分层逐条核对与质疑**；**二、据此给出的可落地分层**。

---

## 0. 先把判据说清楚（措辞纠正）

**「缓存名字」这个说法要改掉。** 前缀缓存的键不是一段内容的「名字」，而是**请求开头那串 token 序列本身**。你能控制的只有「从第几个 token 开始和上一轮不同」，不存在「给某块起个名字就能命中」这回事。所以本文统一用：

- **稳定段**：请求从第 1 个 token 到此处逐字节不变 → 这段可被复用。
- **断点**：第一个发生变化的 token 位置；它之后的一切都要重算。

**判据只有一条，一刀切**：

> 同一个任务内**恒定** → 可以留在前面前缀；同一个任务内**会变** → 必须推到末尾。

注意是「同一任务内」，不是「跨任务」。跨任务的稳定性只影响**跨任务复用率**，不改变前缀是否被击穿。

**provider 差异（决定这件事值不值得做）**：

| provider | 缓存机制 | 需要显式断点吗 |
|---|---|---|
| anthropic（[llm.py:274](../minitap/mobile_use/services/llm.py#L274)） | 显式 `cache_control`，默认 TTL 5min，写入按 1.25x 计费 | **需要**，不打点就不会缓存 |
| openai / minimax / minitap（默认 `google/gemini-2.5-pro`，[llm.py:191](../minitap/mobile_use/services/llm.py#L191)） | 自动前缀缓存，按 128/1024 token 粒度对齐 | 不需要，但要求前缀稳定 |
| google / vertexai | 隐式缓存，有最小长度门槛 | 不需要 |

**全仓 grep 目前没有任何 `cache_control` / prompt cache 相关代码**（只有 Dockerfile 里两条无关命中）。也就是说：现在**一份缓存都没主动开过**，你的分层是在为「未来接缓存」做准备。这个前提必须在文档里写明，否则会误以为收益已经存在。

---

## 一、你的分层：逐条核对

你给的原始分层（照录）：

```
# 系统提示词
1. 角色、规则、输出格式约束
2. 工具
# 计划类
计划：目标、计划列表
计划完成情况：当前计划、计划完成情况（最好只是简单的引用）
# 设备信息
设备的基本信息（不变）、状态信息（会变）：当前前台APP、UI树、OCR文本、截图base64
# 推理历史
最近 25 条 agent thoughts
```

你排出的顺序：

```
1. 系统提示词（不变）
2. 设备的基本信息（不变）
3. 用户画像 + 任务描述等（不变）- 用户自定义
4. 计划（不变）- 计划智能体产出
5. 智能体状态累加（推理历史、Current Subgoal、当前已完成）
6. 设备状态（实时变化）
```

核对结果：

| # | 你的判定 | 事实 | 结论 |
|---|---|---|---|
| 1 | 系统提示词不变 | 前半段（L1–140）确实逐字节恒定；但 **L146–155 的 `Input` 段塞在同一个 system prompt 里，每轮重渲染** | ⚠️ **部分错误** |
| 2 | 设备基本信息不变 | `ctx.device.to_str()` = host platform / platform / device_id / width / height（[context.py:46-53](../minitap/mobile_use/context.py#L46-L53)），任务内确实不变 | ✅ 但当前和「会变的东西」挤在同一行 |
| 3 | 用户画像不变 | **当前并不存在这个字段**（见 Q10） | ⚠️ 属新增 |
| 4 | 计划不变 | 计划**会变**：status / completion_reason / replan（见 Q1、Q2） | ❌ **错误** |
| 5 | 累加后前缀永远不变 | 与 `[-25:]` 滑动窗口**自相矛盾**（见 Q3） | ❌ **错误** |
| 6 | 设备状态实时变化 | 对 | ✅ 但漏了 executor feedback（见 Q4） |

---

## 二、质疑清单

### A. 与事实冲突的三条（硬伤）

#### Q1 「计划不变」不成立：`subgoal_plan` 是原地可变对象

证据链：

- 渲染点 [cortex.py:159](../minitap/mobile_use/agents/cortex/cortex.py#L159)：`"\n".join(str(subgoal) for subgoal in state.subgoal_plan)`。
- `Subgoal.__str__`（[types.py:33-48](../minitap/mobile_use/agents/planner/types.py#L33-L48)）把 **status emoji 和 completion_reason 直接写进行里**：
  `- [ID:xxx]: 描述 : ⏳.` / `... : ✅. Completion reason: ...`
- status 被四处原地改写（[planner/utils.py:24-64](../minitap/mobile_use/agents/planner/utils.py#L24-L64)）：`start_next_subgoal`（→ PENDING）、`complete_subgoals_by_ids`（→ SUCCESS）、`fail_current_subgoal`（→ FAILURE）。

**后果**：当前子目标那一行每轮从 `(not started yet)` → `⏳`；完成时再追加 completion_reason。**这一块每轮都在变，前缀从第一行 status 变化处就被击穿。**

**修法（也正是你第 5 层的本意）**：把 `Subgoal` 显式拆成两半——

- **计划骨架**（稳定）：`序号 + ID + description`，**不含 status、不含 completion_reason**。
- **进度增量**（累加）：status 流转、完成原因、current_subgoal。

只要骨架里不出现 status，它就在任务内恒定。**但这一步必须真的改渲染代码**，光按你的分层重排、仍用 `str(subgoal)`，等于没做。

#### Q2 更狠的：replan 会替换计划整段，且改写点在列表**中部**

证据：

- [planner.py:155-160](../minitap/mobile_use/agents/planner/planner.py#L155-L160)：replan 时 `subgoals_plan = completed_subgoals + new_subgoals`，新子目标走 `generate_id()`（[utils.py:67-70](../minitap/mobile_use/agents/planner/utils.py#L67-L70)）生成**随机 6 位 ID**。
- ID 也进了渲染字符串（`[ID:xxx]`，types.py:45）。
- 触发顺序：orchestrator 先 `fail_current_subgoal` 把**失败的那个**先改成 FAILURE（[orchestrator.py:96-102](../minitap/mobile_use/agents/orchestrator/orchestrator.py#L96-L102)），再进 planner 追加新子目标。失败项在列表**中部**。

**后果**：replan 时，从失败那一行起，后面全部重排 + 随机 ID → 前缀整段断裂，且**不可预测**（ID 随机）。

**修法**：

1. 计划骨架里**去掉随机 ID**（ID 只在"进度/累加"区出现，见 Q9）。
2. replan 改成**追加式**：不重写整张表，而是把「新计划」作为新的一段追加到累加区，并留下 `REPLAN #n` 分隔。老内容一律不改 → 前缀单调。

#### Q3 「累加保证前缀永远不变」与「最近 25 条」互斥

证据：

- 发送侧：[cortex.py:180](../minitap/mobile_use/agents/cortex/cortex.py#L180) `state.agents_thoughts[-MAX_CORTEX_AGENTS_THOUGHTS:]`，硬编码 `MAX_CORTEX_AGENTS_THOUGHTS = 25`（cortex.py:40）。
- 存储侧：[state.py:119](../minitap/mobile_use/graph/state.py#L119) `return old + named_thoughts`，**无界增长**，裁剪只发生在渲染时。
- token 曲线（token-context-analysis §四）：30 个 subgoal 时 thoughts 重发占 cortex 上下文 **57.7%**。

**两种「累加」不可兼得**：

| 方案 | 前缀 | 成本 |
|---|---|---|
| 全量累加、永不删 | ✅ 单调不变 | ❌ O(n) 增长，长任务过半上下文是 thoughts |
| 最近 25 条滑动窗口 | ❌ 每轮内容整体左移 | ✅ 有界 |

**所以「累加保证前缀永远不变」这句在实现上是假的** —— 只要用了 `[-25:]`，前缀每轮都在变。

**建议**：明确选一种，并写进文档。我的建议是**真累加（只追加、不删除）+ 另找 token 杠杆**：把省 token 的力气花在工具 schema 结构手术（token 报告 §五·一，−716/次）和截图降分辨率（§5.3）上，这两项收益确定、不依赖缓存。因为**缓存不解决「重发」，只让重发变便宜**；先减重发量，再让重发变便宜。

> **已按此实施**（2026-10-19）：`[-25:]` 已去掉，见 §四·一 #5。

**附带坑**：`_add_agent_thoughts`（state.py:119）给每条加 `[agent] ` 前缀（`[cortex] `/`[orchestrator] `/`[contextor] `）。而 orchestrator 每轮追加的 thought 文本各不同（[orchestrator.py:42-47](../minitap/mobile_use/agents/orchestrator/orchestrator.py#L42-L47)、99、123），且 `[contextor] None` 会被过滤为空。这些都会影响"累加区"内部的前缀连续性。

### B. 目标自相矛盾与遗漏

#### Q4 最贵的失误：`Executor Feedback` 就长在 system prompt 里，且每轮必变

证据：

- 模板位置：[cortex.md](../minitap/mobile_use/agents/cortex/cortex.md#L152) 的 L152。
- 渲染来源：[cortex.py:161](../minitap/mobile_use/agents/cortex/cortex.py#L161) ← `get_executor_agent_feedback`（[cortex.py:270-293](../minitap/mobile_use/agents/cortex/cortex.py#L270-L293)），内容含上一轮 `structured_decisions` 与 executor 的 ToolMessage 汇总。

**后果**：system prompt 在 provider 眼里是**一个缓存块**。它只要变 1 个 token，**整块 1747 token 全废**。把 feedback 留在里面 = 每轮 system prompt 必 miss。

**修法**：feedback 必须**移出 system prompt**，落到末尾易变层。你的 6 层里没有它的位置 —— 这是漏项，且是当前最贵的一项。

#### Q5 「设备基本信息（不变）」里混进了会变的东西

证据：[cortex.py:166-174](../minitap/mobile_use/agents/cortex/cortex.py#L166-L174) 把三者拼成**同一条 HumanMessage**：

```
["Here are my device info:", device.to_str()]      ← 不变（context.py:46-53）
  + f"Device date: {state.device_date}"            ← 会变
  + f"Focused app info: {state.focused_app_info}"  ← 会变（就是「当前前台 APP」）
```

**后果**：`focused_app_info` 本质是**设备状态**（contextor 每轮写当前前台包名，[contextor.py:103](../minitap/mobile_use/agents/contextor/contextor.py#L103)），却和学习不变的基本信息挤在一起 → 只要前台 app 变了，这一层立刻击穿。`device_date` 更麻烦：cortex 结束时清空、contextor 每轮重写（cortex.py:254 + contextor.py:105），存在**有的轮有值、有的轮没有** → 会周期性破坏前缀。

**修法**：把 `device.to_str()` 留在稳定层；`device_date` / `focused_app_info` 移到末尾状态层。

> 附注：v1 token 报告 §6.5 记的那个 device 拼接优先级 bug 在当前 [cortex.py:166-170](../minitap/mobile_use/agents/cortex/cortex.py#L166-L170) 已改成 list append，**已修复**，不在本次讨论范围。

#### Q6 「计划完成情况最好只是简单的引用」——小心把 ID 引没了

cortex 必须输出 `complete_subgoals_by_ids`（[cortex.md](../minitap/mobile_use/agents/cortex/cortex.md#L123) 的 L123），它需要**看到 ID**。如果「计划」层按 Q2 去掉 ID、「进度」层又只写"已完成一句话"，模型就无 ID 可用，只能瞎猜。

**修法**：ID 放在**稳定骨架**里（ID 在 replan 之外本来就稳定），进度层只放 status 与 completion_reason 增量。`goals_completion_reason` 是 required（cortex.md:126），它需要"为什么完成"的上下文，进度层要能供得上。

#### Q7 「用户画像 + 任务描述（用户自定义）」——当前并不存在

仓里的 `AgentProfile`（sdk/types/task.py）是 **LLM 配置档**（模型/provider/key），**不是用户画像**；提示词里唯一的用户输入是 `initial_goal`（state.initial_goal，[cortex.py:158](../minitap/mobile_use/agents/cortex/cortex.py#L158)）。

所以这是一块**新增**内容。位置放对了（任务内绝对不变 → 前排），但两点提醒：① 它会**每轮重发**，画像宜短；② 现在 `initial_goal` 塞在 system prompt 尾部（cortex.md:146），要跟它一起决定去留。

### C. 工程细节与顺序

#### Q8 排序原则应明说：按「首次变化时刻」，不是按语义

你的顺序大体正确，但应把原则写死：**越晚变化越靠后**。而同一任务内的"中间态"（thoughts、计划进度、feedback）彼此的顺序也互相影响 —— 谁先变，谁后面全废。据此的推荐序（见第三节）。

#### Q9 消息结构层面（不是文本层面）也在影响前缀

现状用**多条 HumanMessage**分装、且**条数随条件变化**（[cortex.py:171-206](../minitap/mobile_use/agents/cortex/cortex.py#L171-L206)：thoughts / UI 树 / OCR / 截图各自 `if`）。

即便拼接后文本前缀稳定，**消息条数与边界本身也是前缀的一部分**（chat 模板会插入 role 标记）。缺内容时"整条不 append"会让边界漂移。

**修法**：把「稳定骨架 + 累加区」压成**固定条数**的消息；末尾易变层也保持**固定条数**，缺内容时填占位符（`(none)`）而不是省略整条。

**约束**：thoughts 绝不能塞成 assistant 回合（[cortex.py:175-179](../minitap/mobile_use/agents/cortex/cortex.py#L175-L179) 有实测注释：会被 MiniMax 当 few-shot 照抄、无视 `tool_choice`）。分层重构不得破坏这一点。

#### Q10 `{% if %}` 会改变模板长度

- [cortex.md](../minitap/mobile_use/agents/cortex/cortex.md#L154-L156) 的 `{% if locked_app_package %}`：不开 app lock 时整行消失。任务内稳定（锁一次），所以**不击穿本任务前缀**，但跨配置不可复用。
- cortex.md:31-33（双通道关闭警告）、55-63（OCR 段）、86（`executor_tools_list`，随 `video_recording_enabled`，[cortex.py:152-154](../minitap/mobile_use/agents/cortex/cortex.py#L152-L154)）：均由 perception / 配置决定，**任务内固定**。

**结论**：这些不构成"任务内击穿"，可接受。真要最大化跨任务复用率，就把条件行改成"永远输出、无值时填 `none`"。**优先级低**，别为它牺牲可读性。

### D. 收益与验证（别高估）

#### Q11 上限只有约 30%，而且必须先有 telemetry

用 token 报告实测数（cortex 每轮，不含截图 ≈ 4,802）：

| 组成 | token | 每轮变？ | 能否命中缓存 |
|---|---:|---|---|
| system prompt | 1,747 | **现在是（feedback 在里面）** | 修好后✅ |
| UI 层级树 | 2,527 | 屏幕变才变 | 同屏连续轮可命中 |
| thoughts | 300 | 变（`[-25:]`） | 改成只追加后✅ |
| device info | 32 | 部分变 | 拆分后部分✅ |
| 截图 | ~2,064 | 屏幕变才变 | 同屏连续轮可命中 |

**关键事实**：占大头的 UI 树 + 截图 ≈ 4,591（占 67%）**只跟屏幕有关**，而 cortex 常见的"犹豫/等待/重试"会导致**连续多轮同屏** → 那几轮这部分是能命中的。所以真实收益 = 稳定段（约 30%）+ 同屏连续轮的额外命中，取决于任务节奏，**必须用真实 usage 量化后再拍板**。

**前置阻塞**：当前 `_record_usage`（[llm.py:135-165](../minitap/mobile_use/services/llm.py#L135-L165)）对 `with_structured_output` 路径**恒记 0**（cortex/planner/orchestrator 全走这条路）。**不先接 usage，任何优化收益都无法验证。**

#### Q12 需要核实：渲染是否确定性

要让"同屏连续轮"真的命中，UI 树 / OCR 的渲染必须**逐字节确定**（不能有 dict 顺序抖动、时间戳、浮点格式差异）。`render_hierarchy_and_labels`（[ui_hierarchy_view.py](../minitap/mobile_use/utils/ui_hierarchy_view.py)）与 `render_ocr_results`（[ocr_view.py](../minitap/mobile_use/utils/ocr_view.py)）本次未逐行核对，**列为待验证项**。如果它们不稳定，同屏也会 miss。

---

## 三、建议的 v2 分层

原则：**把「任务内恒定」压到最前、把「任务内每轮变」压到最后，中间夹「只追加」的累加区。全部内容按"首次变化时刻"排序。**

| 层 | 内容 | 任务内变化 | 建议载体 | 现状问题 |
|---|---|---|---|---|
| **L0** | 角色、规则、感知说明、元素/OCR 格式、工具表、swipe/表单、输出格式、few-shot | 恒定 | SystemMessage（静态段） | 基本 OK（cortex.md L1–140） |
| **L1** | 设备基本信息：`device.to_str()` | 恒定 | 紧随 L0 | 现与 date/前台app 混在一起 |
| **L2** | 用户画像 + `initial_goal` | 恒定 | 紧随 L1 | 现在 `initial_goal` 在 sys 尾部 L146（L2 位置，其实等价，OK） |
| **L3** | 计划骨架：`序号 + ID + description`（**无 status / 无 reason**） | 仅 replan 时**尾部追加** | 固定段 | 现在含 status/reason/随机 ID → 每轮变 |
| **L4** | 累加日志：thoughts（**只追加**） | 单调追加 | 固定段 | 现在是 `[-25:]` 滑窗 → 每轮变 |
| **L5** | 每轮易变：**完成情况索引**（`Subgoal status: [id] done/active/pending`，**替换式**、按计划序固定槽位）+ **executor feedback** + `device_date` + `focused_app` + UI 树 + OCR + 截图 | 子目标推进 / 每轮变 | 末尾**固定条数**消息 | feedback 在 sys 里；date/前台app 在 L1；条数不固定 |

与现状的映射要点：

- 现状 system prompt 里的 L146–152（Initial Goal / Subgoal Plan / Current Subgoal / **Executor Feedback**）**整段要拆出去**：
  - Initial Goal → L2
  - Subgoal Plan 的**骨架** → L3（只追加），**完成情况** → L5（替换式索引）
  - 完成情况（`Subgoal status` 索引，按计划序固定槽位）→ **L5 首位**（**替换式**、不会插行；必须排在 thoughts 之后，否则子目标一推进就会把 thoughts 前缀整段打掉）
  - **Executor Feedback → L5（最关键）**
- Locked App（cortex.md:154-156）可由 L3/L4 承载或保留原位（任务内恒定，安全）。

---

## 四、落地改动清单（代码级）

| # | 改动 | 文件/位置 | 风险 |
|---|---|---|---|
| 1 | 拆 `Subgoal` 渲染：新增 `plan_skeleton()`（只含 ID+description）与 `progress_delta()` | [planner/types.py](../minitap/mobile_use/agents/planner/types.py#L33-L48) | 低。注意别动 `__str__`——planner/orchestrator/contextor/outputter 都在用它 |
| 2 | cortex 渲染改为 `skeleton + append-only progress` | [cortex.py:156-165](../minitap/mobile_use/agents/cortex/cortex.py#L156-L165) | 中。需回归测试完成判定是否仍准确 |
| 3 | **把 feedback 移出 system prompt**，追加到 messages 末尾 | cortex.md:152 + [cortex.py:161](../minitap/mobile_use/agents/cortex/cortex.py#L161) | 低，但必须真机 A/B |
| 4 | 拆 device 行：`to_str()` 留前，date/前台app 移后 | [cortex.py:166-174](../minitap/mobile_use/agents/cortex/cortex.py#L166-L174) | 低 |
| 5 | thoughts 改「只追加、不截断」；若不肯，就把分层里"前缀永远不变"的表述删掉 | [cortex.py:180-186](../minitap/mobile_use/agents/cortex/cortex.py#L180-L186) | **高**：token 成本上升，需配合 §五·一/§5.3 的减量 |
| 6 | 末尾易变层固定条数，缺内容填 `(none)` | [cortex.py:188-206](../minitap/mobile_use/agents/cortex/cortex.py#L188-L206) | 低 |
| 7 | replan 改追加式（保前缀单调） | [planner.py:155-160](../minitap/mobile_use/agents/planner/planner.py#L155-L160) | **高**：改动计划语义，需单独评估 |
| 8 | 接 token telemetry（否则一切收益不可量化） | [llm.py:135-165](../minitap/mobile_use/services/llm.py#L135-L165) | 无（前置项） |
| 9 | 若要 Anthropic 生效：显式打 `cache_control` 断点 | [llm.py:274](../minitap/mobile_use/services/llm.py#L274) | 中，需按 provider 分别处理 |

**顺序建议**：8（telemetry）→ 3、4、6（低成本、确定收益）→ 1、2（拆 Subgoal）→ 5（thoughts 策略，需先定）→ 7（replan 追加式）。

### 四·一、首轮落地状态（2026-10-19）

用户决定「先把骨架做好」后，以下已实施：

| # | 状态 | 落地位置 |
|---|---|---|
| 3 | ✅ feedback 移出 system prompt | [cortex.py](../minitap/mobile_use/agents/cortex/cortex.py#L232) 末尾消息 |
| 2 | ✅ 消息按 恒定→只追加→每轮变 分层 | [cortex.py](../minitap/mobile_use/agents/cortex/cortex.py#L171-L212)；完成情况以 `Subgoal status` **索引**排在 thoughts **之后**（替换式、按计划序固定槽位）；子目标骨架随计划日志留在 L1 |
| 2b | ✅ **计划日志也挪到 thoughts 之后**（2026-10-21） | 计划日志只追加，但**排在 thoughts 前面时**，replan 的追加点之后全部重算 → thoughts 全长满价重算。挪到 thoughts 之后后实测（真 API，122 条 thoughts）：replan 当轮 cached 2,329 → **10,411**，满价重算 8,261 → **183**。见 [cache-hit-rate-audit-20261020.md](cache-hit-rate-audit-20261020.md) |
| 6 | ✅ 静态段不留 `{{ }}` 动态值 | [cortex.md](../minitap/mobile_use/agents/cortex/cortex.md) 整段静态 |
| 4 | ✅ 设备基本信息与 date / 前台 app 拆开 | 恒定块 [cortex.py](../minitap/mobile_use/agents/cortex/cortex.py#L181-L193)、易变块 [cortex.py](../minitap/mobile_use/agents/cortex/cortex.py#L230-L239) |
| 1 | ✅ 计划骨架 / 完成情况拆成纯函数 | [planner/utils.py](../minitap/mobile_use/agents/planner/utils.py#L67-L102)（`render_plan_skeleton` 骨架 + `render_subgoal_status` 完成索引） |
| 5 | ✅ thoughts 改「只追加、不截断」 | [cortex.py](../minitap/mobile_use/agents/cortex/cortex.py#L42-L52)（去掉 `[-25:]` 滑窗，注释写明代价） |
| 7 | ✅ replan 改追加式（计划日志） | 新增 state 字段 [state.py](../minitap/mobile_use/graph/state.py#L36-L43)、追加逻辑 [planner.py](../minitap/mobile_use/agents/planner/planner.py#L108-L119)、渲染 [cortex.py](../minitap/mobile_use/agents/cortex/cortex.py#L187-L189) |
| 8 | ✅ 接 token telemetry | 捕获槽 [llm_usage.py](../minitap/mobile_use/utils/llm_usage.py#L31-L67) + 客户端 generate/agenerate 包装 [llm.py](../minitap/mobile_use/services/llm.py#L200-L250)；测试 [test_llm_usage_capture.py](../tests/mobile_use/test_llm_usage_capture.py) |
| 9 | ✅ Anthropic 显式 `cache_control` 断点 | [AnthropicCacheControlChat](../minitap/mobile_use/services/llm.py#L346-L366) |

**残留（已知，未做）**：

- `#5` 的代价是 token 随轮次线性增长。压成本的杠杆在别处：工具 schema 结构手术（token 报告 §五·一，−716/次）与截图降分辨率（§5.3）。
- `#7` 之后模型看到的是**计划历史**（最后一版为当前计划），语义变了，需真机 A/B 验证完成判定不受影响。
- `#8` 的类级包装只覆盖 `ChatOpenAI`（含 MiniMax / DeepSeek 子类）与 `ChatAnthropic`；Google / Vertex 的客户端是惰性导入的重依赖，未包装，仍记 0。provider 本身不回报 usage 时也仍记 0（不是我们丢的）。
- `#9` 只对 Anthropic 生效；OpenAI 兼容端点（默认 `google/gemini-2.5-pro`）依赖自动前缀缓存，不需要打点。

**遗留副作用（须处理）**：

- ✅ `cortex.zh.md` 已同步重写为同结构（静态、无 emoji、无 Input 段）。`probe_out/probe_prompt_parity.py` 复跑 **PASS**：fenced code blocks en=4/zh=4 逐字节相同、jinja vars en=10/zh=10 完全一致。
- ⚠️ [prompt-i18n-ab.md](prompt-i18n-ab.md) 里的 token 绝对值（en 2270 / zh 3216）已是旧结构的数，两版提示词重写后不再可比，方向性结论仍成立。
- [提示词组成v1.md](提示词组成v1.md) 的 23 块表已成历史快照，反映的是重构前形态。

---

## 五、收益量级（先把话说满，别高估）

1. **缓存救不了 O(n)**：thoughts 是唯一随轮次线性增长的项（长任务占比 >50%）。缓存只让**固定底数**变便宜。所以**先减重发量**（工具 schema、截图、thoughts 策略），**再让重发变便宜**。
2. **稳定段 ≈ 30%**：即使把 Q1–Q6 全修好，可稳定复用的也只有 system + 设备基本信息 + 计划骨架 + 画像 ≈ 2k/6.9k。占比不高，但 cortex **每轮都跑**，绝对值可观。
3. **真正的额外收益在"同屏连续轮"**：UI 树 + 截图占 67%，屏幕不变的那些轮次整体命中，这部分收益取决于任务节奏，**无法先验估计**。
4. **不要为了缓存牺牲正确性**：Q6（ID 可见性）、Q9（thoughts 不能变 assistant 回合）都是"改了会让模型变傻"的红线。

---

## 六、四个决策（已定，2026-10-19）

1. **thoughts「真累加」还是「滑窗」？** → 定：**真累加（只追加、不截断）**。代价是 token 线性增长，用工具 schema / 截图两项减量去抵。
2. **只做分层，还是同时接缓存机制？** → 定：**都做**。Anthropic 打显式 `cache_control` 断点；OpenAI 兼容端点（默认 `gemini-2.5-pro`）依赖其自动前缀缓存，不需要打点。
3. **要不要新增「用户画像」字段？** → 定：**不新增**。画像属于场景化内容，走用户提示词（`initial_goal` 那段恒定块），天然稳定。
4. **先做哪一个？** → 定：**先做分层**（已完成），再补 telemetry / thoughts / replan / cache_control。

---

## 七、验证方式

- **离线（已建）**：[probe_cortex_prompt_variance.py](../probe_out/probe_cortex_prompt_variance.py) 用真模板 + 真 render 函数逐块 diff 两轮，产物 `probe_out/cortex_prompt_variance.txt`。当前实测：L0 逐字节相同、L1 在 replan 后**只有追加行**、完成情况索引（`Subgoal status`）按计划序固定槽位且排在 thoughts 之后（乱序完成也不插行）。
- **离线（已建）**：[test_llm_usage_capture.py](../tests/mobile_use/test_llm_usage_capture.py) 用 stub 模型把「结构化输出路径 → 捕获槽 → telemetry」整条链路跑通，并验证 Anthropic payload 带 `cache_control`。
- **真机 A/B**：仿 [probe_out/probe_i18n_ab.py](../probe_out/probe_i18n_ab.py) 的框架（同任务、同设备、仅提示词结构不同），比较完成率、轮数、**真实 usage**（telemetry 现已能取到，见 #8）。
- **确定性核查**：Q12 的 UI/OCR 渲染逐字节复现性。

---

## 附：与 v1 的核心差异一句话总结

v1 回答「现在有什么、哪块会变」；v2 回答「**应该**怎么分层、你的分层错在哪」。v1 的结论仍然有效，且恰好证明了 v2 的必要性：**v1 表里 20/21/22 三块会变的，正是 v2 要压到末尾的那三块（计划进度 / current subgoal / feedback）；它们现在全在 system prompt 尾部 —— 这是当前唯一的结构性问题。**
</content>
