# Agent thoughts history 压缩 —— 方案与落地记录

> 方案调研：[thoughts-compaction-plan.md](./thoughts-compaction-plan.md)
> 本文件：**落地方案 + 实施记录 + 验证结果存档索引**
> 实施日期：2026-10-20
> 状态：**已实现，离线验证全过；真机 A/B 未做**

---

## 一、一句话

把发给 LLM 的 thoughts 历史里的**散文**（cortex 推理）压成首句 + 末句，
**事实**（executor 工具结果、orchestrator 里程碑）逐字保留。
因为压缩是**位置无关**的，前缀缓存不因压缩而击穿。

末轮 prompt 降 **40.3%**，全链路 token 降 **34.2%**，满价重算反而降 **9.2%**。

---

## 二、方案

### 2.1 为什么不是滑窗

`state.agents_thoughts` 是只追加的，长任务里线性膨胀（实测 122 条 = 22,229 token，
占 cortex prompt 的 65%）。但直接截断会让提示词前缀每轮整体左移 → 前缀缓存
永远命中不了 → 满价重算涨 4.7 倍。

实测对照（真机归档 122 轮）：

| 方案 | 命中率 | 满价重算 |
|---|---:|---:|
| 基线（全量只追加） | 93.5% | 117,605 |
| **本方案（位置无关规则）** | **91.1%** | **106,828** |
| 滑窗 25 | 47.2% | 550,545 |
| 摘要（epoch 重写） | 86.9% | 118,172 |

> ⚠️ **这里的「命中率」是模拟值，不是 provider 回报的实测值。**
> 122 轮 thoughts 是真机归档，但算命中率用的 prompt 是 `"S"*6085 + "D"*1419 +
> thoughts + "X"*6004` 拼的重复字符，**从未打过真 API**。
>
> 真机归档里 provider 报的真实值是 **10.4%**（`usage_212.json`：98,605 / 948,633）——
> 那次运行跑的是 `9af5a62`（2026-10-18 14:15），即本方案与 cortex 提示词重构
> **之前**的架构，所以两个数不矛盾，只是量的是不同东西。
>
> 本表的正确读法是**方案之间的相对比较**（滑窗 47.2% vs 只追加 93.5%，
> 说明位置无关性是收益来源），**不是**「线上真实命中率」。
> 真 API 实测见 [cache-hit-rate-audit-20261020.md](cache-hit-rate-audit-20261020.md)。

### 2.2 关键性质：位置无关

判据不是「命中率」（那是结果），而是**字节前缀**：

> `f(ts[:k])` 恒为 `f(ts[:k+1])` 的字节前缀 → 已发出的字节永不改变 → 前缀永不击穿。

- 滑窗 / 摘要重写是**位置相关**的（第 26 条进来，第 1 条的渲染结果变了）
- 本方案是**位置无关**的（第 26 条进来，前 25 条一字不变）

这条性质是**全部收益的根据**。破了它，压缩就从「省 34%」变成「满价重算涨 4.7 倍」。

### 2.3 分类处理

| 角色 | 内容 | 性质 | 处置 |
|---|---|---|---|
| `cortex` | `Decisions reason` / `Goals completion reason` | 散文 | 压：首句 + 末句 |
| `executor` | 工具结果 | **事实** | 逐字保留 |
| `orchestrator` | 子目标里程碑 | **事实** | 逐字保留 |

事实不能压，两个硬理由：

1. 归档分析的事实统计口径就是数 `[executor]` 开头行（`perception-staleness-analysis.md` §7.2）——压了它口径就废了。
2. `cortex.md` 硬规则 2 要求模型「不重复失败动作」——失败事实就是判断依据。

### 2.4 只改渲染，不改 state

`state.agents_thoughts` 保持完整原文，它有 4 个非 LLM 消费方：
`sdk/agent.py`（trace 导出 / steps_count）、`tui/events.py`（实时渲染）、
`outputter`（最终输出，要的是完整推理过程）、归档分析工具。

---

## 三、改动清单

| # | 文件 | 改动 |
|---|---|---|
| 1 | `minitap/mobile_use/utils/thoughts_compaction.py` | **新增**。`split_role()` / `split_sentences()` / `compress_thought()` / `render_thoughts_history()` —— 纯函数，无副作用 |
| 2 | `minitap/mobile_use/context.py` | `PerceptionConfig.thoughts_compaction: bool = True` —— 回退 / A-B 开关 |
| 3 | `minitap/mobile_use/agents/cortex/cortex.py` | 渲染改用 `render_thoughts_history()`；标题行加压缩说明 |
| 4 | `minitap/mobile_use/agents/orchestrator/orchestrator.py` | 同样改用 `render_thoughts_history()`（与 cortex 同一函数，口径不分裂） |
| 5 | `minitap/mobile_use/agents/cortex/cortex.md` | 硬规则 1 补一句：历史是压缩过的，缺细节就按当前屏幕行事 |
| 6 | `minitap/mobile_use/agents/cortex/cortex.zh.md` | 同上（中文） |
| 7 | `tests/mobile_use/test_thoughts_compaction.py` | **新增** 25 项单测，锁不变量 |
| 8 | `probe_out/probe_thoughts_compaction_e2e.py` | **新增** 端到端验证（真机归档） |
| 9 | `probe_out/archive_verification.py` / `archive_baseline_compare.py` | **新增** 验证结果存档脚本 |
| 10 | `minitap/mobile_use/agents/orchestrator/human.md` | **段落重排**：`Agent thoughts` 提到 `Initial goal` 之后，`Subgoal plan` / `Subgoals to examine` 挪到它下面（见 4.5） |

**没改的**：`state.py`（state 语义不动）、`planner.py` / `contextor.py`（已有 25 条上限，
不是瓶颈）、`outputter.py`（只付一次，且要完整推理过程）。

---

## 四、验证结果

### 4.1 离线验证（全过）

| 项 | 结果 | 存档 |
|---|---|---|
| 新增单测 | **25 passed** | `20261020_pytest_compaction.txt` |
| 端到端（真机归档 122 条） | **全部通过** | `20261020_e2e_real_archive.txt` |
| 提示词模板校验 | **24/24 渲染通过** | `20261020_prompt_templates.txt` |
| 类型检查（pyright） | **0 errors** | `20261020_pyright.txt` |
| lint（本次改动文件） | **All checks passed** | `20261020_ruff.txt` |
| 全量测试 | 379 passed / 6 failed / 4 errors | `20261020_pytest_full.txt` |
| **基线对照** | **10 个失败项逐条一致** | `20261020_baseline_compare_*.txt` |

> 存档目录：`probe_out/verification/`

### 4.2 基线对照（重要）

全量测试有 6 failed + 4 errors。这**不是本次改动引入的** ——
`probe_out/archive_baseline_compare.py` 用 `git stash` 把本次改动摘掉，
跑同一批失败项，存档结论是「**逐条一致: 是**」（10 个失败项，两边完全相同）。

失败项是既有的：
- `test_minimax_provider.py`（6 failed）—— `MINIMAX_BASE_URL` 环境变量未设
- `test_idb_client.py`（4 errors）—— iOS 客户端依赖缺失

与 `doc/perception-staleness-analysis.md` §6 记录的基线一致。

### 4.3 一个必须记下来的坑：flaky 测试

存档过程中出现过一次 **8 failed**（比基线多 2 项 `test_deepseek_provider`）。
没有当成「偶发」放过，而是查了根因（`20261020_flaky_analysis.txt`）：

| 证据 | 结果 |
|---|---|
| 单独跑该文件，连跑 3 次 | **2 次失败 / 1 次全过** |
| 失败报错 | `ValueError: MiniMax response had no tool_calls and no content. finish_reason='tool_calls'` |
| 根因 | 该测试**真实调用 LLM API**（`structured.ainvoke(...)`），不是 mock。服务端返回 `finish_reason='tool_calls'` 却没带 tool_calls —— 模型侧响应不稳定 |

**结论：flaky，与本次改动无关**（本次未触碰 deepseek / llm 任何代码）。

两点处置：
1. 全量测试以 `-p no:randomly` 固定顺序重跑，得到稳定数字（6 failed / 379 passed）并存档。
2. 这类依赖外部服务的测试**不能作为「是否引入回归」的判据** —— 判据是 §4.2 的基线对照。

> 记录这条是为了以后有人看到 deepseek 失败时，能直接翻到证据，
> 而不是重新怀疑一遍「是不是我上次改坏了」。

### 4.4b orchestrator 段落重排（`Agent thoughts` 提前）

把 `Agent thoughts` 从最后一段移到 `Initial goal` 之后，
`Subgoal plan` / `Subgoals to examine` 挪到它下面。

**动机**：让 orchestrator 先看「实际发生了什么」，再对着计划判断哪些子目标真完成了。

**技术风险**：直觉上「把长内容放在可变内容之前」会伤前缀缓存。所以单独量了
（`probe_out/probe_orchestrator_reorder.py`，用真实归档模拟 orchestrator 的 19 次调用）：

| | 相邻调用平均公共前缀 | 单次 prompt 体积 |
|---|---:|---:|
| 重排前（thoughts 最后） | 11,526 token | 22,346 token |
| **重排后（thoughts 靠前）** | **11,650 token** | 22,345 token |
| 变化 | **+1.1%** | −0.0% |

**没有变差，反而微升。** 原因是 orchestrator 不是每轮都跑 —— 它在**每完成一个
子目标**时才被调用，两次调用之间 thoughts 涨了几条、plan/subgoals 变了一轮。
也就是说，**thoughts 才是那段「几乎不变的长内容」**，plan/subgoals 才是每轮变的短内容。
把 thoughts 提前 = 让长历史贴着稳定前缀，把易变内容挪到后面 —— 正是前缀缓存想要的方向。

| 断言 | 结果 |
|---|---|
| 渲染后段落顺序 | Initial goal → Agent thoughts → Subgoal plan → Subgoals to examine ✅ |
| 提示词模板校验 | 24/24 ✅ |
| 前缀缓存命中面 | +1.1%（无损伤）✅ |
| 全量测试基线对照 | 10 项失败逐条一致 ✅ |

存档：`20261020_orchestrator_reorder.txt`、`20261020_orchestrator_render_order.txt`

### 4.5 端到端验证的 5 项断言

在**真实归档**（exec 212，122 条 thoughts）上跑：

| # | 断言 | 结果 |
|---|---|---|
| 1 | 压缩确实省 token | 22,229 → 11,452（**48.5%**） |
| 2 | 122 轮前缀逐字节不变 | ✅（滑窗对照组在第 26 轮击穿） |
| 3 | 事实条目一条不少 | `[executor]` 51 条、`[orchestrator]` 19 条 **逐条一致** |
| 4 | 开关能回退 | `enabled=False` 逐字节等于原样 ✅ |
| 5 | 满价重算未变多 | 117,605 → **106,828**（降 9.2%）✅ |

端到端模拟 cortex 每轮请求：

| | 总请求 | 命中率 | 满价重算 | 末轮 prompt |
|---|---:|---:|---:|---:|
| 基线（关） | 1,817,500 | 93.5% | 117,605 | 26,734 |
| **压缩（开）** | **1,200,862** | **91.1%** | **106,828** | **15,957** |
| 对比 | −33.9% | −2.4pt | **−9.2%** | **−40.3%** |

---

## 五、真机 A/B（未做 —— 上线前必须做）

**离线验证只证明了「压缩是划算的」，没有证明「压缩是安全的」。**

### 5.1 怎么做

同一任务、同一设备、同一提示词，**仅切换 `thoughts_compaction` 开关**：

```python
# ctx.perception.thoughts_compaction = True / False
```

### 5.2 看什么

| 指标 | 口径 | 为什么 |
|---|---|---|
| **完成率** | 任务成功 / 总数 | 主判据 |
| **轮数** | cortex 轮次 | 压缩若让模型变笨，轮数会涨 |
| **点击成败** | 数 `[executor]` 开头行（`perception-staleness` §7.2） | 压缩最可能的失效模式在这里 |
| **真实 usage** | telemetry（v2 文档 #8 已能取到） | 验证 token 收益在真实链路上成立 |
| **重复动作** | 同一目标连续出现 | **回归重点**，见下 |

### 5.3 回归重点

**长任务（≥50 轮）里模型是否出现「重复已做过的动作」。**

这是压缩最可能引入的失效模式：模型靠历史避免重复，而压缩恰好削掉历史里的细节。
判断依据：压缩后历史里的**结论**仍在（首句）、**失败事实**仍在（逐字保留）、
**完成理由**仍在（末句）—— 但中间的铺陈没了。

如果 A/B 出现重复动作率上升，退路是：
1. 把压缩范围从「所有 cortex 散文」收窄到「只压 N 轮之前」（但这会变成位置相关，击穿缓存）
2. 改成保留首句 + **末两句**（多留一点策略信息，仍位置无关）
3. 直接关掉开关（`thoughts_compaction = False`）

---

## 六、存档索引

```
probe_out/verification/
  20261020_pytest_full.txt                    全量测试（379 passed / 6 failed / 4 errors）
  20261020_pytest_compaction.txt              新增单测 25 项全过
  20261020_e2e_real_archive.txt               端到端（真机归档 122 条）
  20261020_prompt_templates.txt               提示词模板校验 24/24
  20261020_pyright.txt                        类型检查 0 errors
  20261020_ruff.txt                          lint All checks passed
  20261020_baseline_compare__WITH_changes.txt     改动后的失败项
  20261020_baseline_compare__WITHOUT_changes.txt  改动前的失败项（stash 后）
  20261020_baseline_compare__SUMMARY.txt           对照结论（10 项逐条一致）
  20261020_flaky_analysis.txt                      flaky 测试根因判定
  20261020_flaky_deepseek_run{1,2,3}.txt           flaky 复现证据（3 次）
  20261020_orchestrator_reorder.txt                orchestrator 段落重排的缓存影响
  20261020_orchestrator_render_order.txt            重排后渲染快照（段落顺序）

doc/thoughts-compaction-plan.md               方案调研（数据来源与选型过程）
doc/thoughts-compaction-implementation.md     本文件
```

复现验证：

```powershell
python probe_out/archive_verification.py       # 跑全部验证并存档
python probe_out/archive_baseline_compare.py   # 基线对照（会 stash / pop）
python probe_out/archive_flaky_analysis.py     # flaky 根因判定
python probe_out/archive_orchestrator_reorder.py    # orchestrator 段落重排验证
python probe_out/probe_thoughts_compaction_e2e.py   # 只跑端到端
```

---

## 七、待办

- [ ] **真机 A/B**（第 5 节）—— 上线前必须做
- [ ] A/B 结果补进本文件第 5 节
- [ ] 若 A/B 通过：把开关默认值从 `True` 保留，并在 `doc/` 记一笔上线日期
- [ ] 若 A/B 失败：按第 5.3 节的退路逐个试，或直接关开关
