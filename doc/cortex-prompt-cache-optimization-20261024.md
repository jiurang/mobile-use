# Cortex 提示词与缓存优化方案（2026-10-24）

> 目标：**cortex 这个 agent 的提示词成本**（不是命中率百分比本身）。
> 命中率是「分母被摊薄后的比值」，真正花钱的是 **满价重算的那部分 token**。
>
> 全部数字来自 **861 号真机执行**（Redmi Turbo 4 / MiniMax-M3.1-Flash /
> 12 次 cortex 调用），复现命令见附录。

---

## 一、一句话结论

**cortex 的提示词里 57% 是「执行器的逐步流水」，而 cortex 按自己的模板契约根本不需要它。**
砍掉之后（**真 API A/B 实测**）：input 总量 **−22.1%**，**满价（要花钱的）部分 −22.4%**，
命中率百分比基本不动（19.7% → 20.0%）。

---

## 二、现状实测

### 2.1 账

| 指标 | 值 |
|---|---|
| cortex 调用 | 12 次 |
| input | 61,518 tok |
| cached | 23,925 tok（**38.9%**） |
| **满价部分（input − cached）** | **37,593 tok** |
| prompt 字符（末轮 / 平均） | 21,432 / 14,085 |
| 字符级缓存断点 | **11/11 次都落在 thoughts 块末尾** |
| 字符级命中（理论上限） | 64.8% ~ 93.5%，均值 **82.4%** |

⚠️ 实测 38.9% 远低于字符级上限 82.4% ⇒ **provider 的缓存粒度比字符级粗**
（MiniMax 会周期性整块失效，见 `doc/executor-prompt-composition-and-cache-20261024.md` §7.5）。
⇒ 所以优化目标不是「命中率涨几个点」，是**把要重算的字节本身变少**。

### 2.2 末轮 prompt 的块构成

| 块 | 内容 | 字符 | 占比 | 变化性 |
|---|---|---|---|---|
| 0 | system（`## You are the Cortex`） | 6,831 | 31.9% | 恒定 |
| 1 | device info + initial goal | 292 | 1.4% | 恒定 |
| **2** | **Agent thoughts history** | **12,068** | **56.3%** | **只追加 = 断点** |
| 3 | Subgoal plan log | 299 | 1.4% | 只追加（replan 时） |
| 4 | Subgoal status | 111 | 0.5% | 每轮变 |
| 5 | Failed targets | 30 | 0.1% | 偶尔变 |
| 6 | Executor report | 1,765 | 8.2% | 每子目标变 |
| 7 | Failed target facts | 36 | 0.2% | 只增不减 |

断点固定在块 2 末尾 ⇒ **块 3~7（2,241 字符）每轮满价重算**，
加上 thoughts 的新增部分，每轮重算平均 **2,475 字符**（17.6%）。

### 2.3 thoughts 里装的是什么（12 次累计 66,657 字符 / 414 行）

| 角色 | 字符 | 占比 | 行数 | 平均行长 |
|---|---|---|---|---|
| **executor** | **40,599** | **60.9%** | **208** | 195 |
| cortex | 11,346 | 17.0% | 66 | 172 |
| orchestrator | 9,890 | 14.8% | 108 | 92 |
| （无角色前缀） | 4,822 | 7.2% | 32 | 151 |

executor 那 208 行是什么（末轮 44 行 / 7,655 字符的样本）：

```
14×  Tap on element ...
 3×  The screen is ...
 1×  The device is unlocked and on the MIUI home screen ...
 1×  App '应用商店' launched successfully.
 1×  Text erased successfully.
 1×  Key Enter pressed successfully.
 1×  Typed '番茄畅听' using coordinates.
```

⇒ **全是执行器的逐步流水**（点了什么、屏幕是什么、输入了什么）。
这次执行零失败，所以失败类 0 行；但设计上失败类必须留（见 §3.2）。

### 2.4 模板契约怎么说（`cortex.md`）

- `## What you are given` 列了 5 项：**Subgoal plan / Subgoal status / Executor report /
  Failed targets / Failed target facts** —— **没有 thoughts history 这一项**。
- `## Your Loop` 第 1 步：「Read, in this order: the plan log, the subgoal status line,
  the Executor report (if there is one), then the two failure lists.」—— **也没有 thoughts**。
- 对 executor report 的定义：「It is **the only thing** that tells you what actually happened.」

⇒ **thoughts 里的 executor 流水，对 cortex 而言是冗余的**：
执行结果走 report，失败走两个失败块，走过的路线走 report 的 `actions taken`。

---

## 三、方案

### 方案 1（主攻）：规划者视角的 thoughts —— 砍掉执行器的成功流水

**做什么**：在**发给 LLM 的渲染层**过滤 thoughts 里的 `[executor]` 行，
只保留：
- `[cortex]` 行（自己的决策历史：Issued goal / judged COMPLETE / reasoning）
- `[orchestrator]` 行（子目标里程碑）
- **失败/异常类的 `[executor]` 行**（唯一带具体原因的失败依据）

**改哪里**：`utils/thoughts_compaction.py`
给 `render_thoughts_history` 加一个**受众**参数，默认不变：

```python
def render_thoughts_history(
    thoughts: list[str],
    *,
    enabled: bool = True,
    audience: str = "full",        # "full" = 现状（逐条对应，逐字保留）
) -> str:
    # audience="planner"：只留 cortex / orchestrator 行 + 失败类 executor 行
```

调用方（`agents/cortex/cortex.py:457`、`agents/orchestrator/orchestrator.py:102`）
传 `audience="planner"`。
orchestrator 一起改 —— 它也是规划者，也不需要执行器流水，
而且**两个 agent 用同一套规则才不会「口径分裂」**（现有注释就是为这个）。

**不动的**：`state.agents_thoughts` 原文、planner（读 state 原文，不走这个函数）、
trace / TUI / outputter / 归档（读 state）。

#### 3.1 前缀安全（必须满足，否则整个方案白做）

过滤规则必须**逐条独立**（每条 thought 只看自己就能决定去留）⇒ **位置无关**
⇒ 已发出的字节永不改变 ⇒ 追加语义保留 ⇒ 前缀缓存不击穿。
这与 `thoughts_compaction` 现有的「位置无关压缩」是同一个性质。

❌ 反例（位置相关，会击穿）：「只保留最近 3 条 executor 行」「只保留当前子目标的 executor 行」
—— 新条目进来时，老条目的去留变了 ⇒ 旧字节改变 ⇒ 断点前移。

#### 3.2 为什么保留失败类 executor 行

cortex.md 规则 4「Never build a goal around a failed target」。
两个失败块里只有**名字**（`"点赞" like_button`），
**具体原因**（`no element with resource_id='...' on the current screen`）只在 executor 的失败行里。
⇒ 失败行是 cortex 判断「这条路走不通」的唯一具体依据。

判定用正则（宁可多留 —— 多留只是费 token，不会错）：

```python
_FAIL_RE = re.compile(
    r"fail|error|not on the current screen|no element found|exception|"
    r"timeout|denied|refus|cannot|unable",
    re.IGNORECASE,
)
```

#### 3.3 风险清单

| 风险 | 判断 | 兜底 |
|---|---|---|
| cortex 失去「执行器卡在哪」的中间感知 | **低**：cortex 只在子目标结束时被调用（cortex.md:「called once per subgoal attempt」），那时 report 一定在 | 真机对照跑，看是否多出 replan |
| 归档口径「数 `[executor]` 开头行」被破坏 | **无**：`audience="full"` 默认行为逐条不变；归档读 state 原文 | 加测试锁死默认参数下的行数一致 |
| 失败类正则漏判 | 中 | 正则宁可放宽；真机对照跑专门挑会失败的任务 |
| executor 行其实藏着 cortex 需要的东西 | **待验**：`actions taken` 只记动作不记推理，执行器的**目标级观察**（如「前台仍是下载队列页」）可能被砍 | 真机对照跑，盯子目标质量 |

#### 3.4 预期收益 —— ⚠️ 以**真 API 实测**为准（2026-10-24 修正）

**先说一件事**：本文档初稿写的「满价部分 −34%」是**离线推算**，被真 API 实测推翻了。
中间还出现过一个「满价只降 4.9%」的错误推算。**两个都不对**。

真 API A/B（`scripts/probe_cortex_planner_cache_ab.py`，拿 861 的**真实 12 轮 prompt**，
两种渲染各发一遍，MiniMax-M3.1-Flash-Preview，每组用唯一 tag 冲刷缓存）：

| 口径 | A 现状 | B 方案 1 | 变化 |
|---|---|---|---|
| input 总量 | 56,562 | 44,073 | **−22.1%** |
| cached 总量 | 11,124 | 8,832 | −20.6% |
| **满价部分（input − cached）** | **45,438** | **35,241** | **−22.4%** |
| 命中率百分比 | 19.7% | 20.0% | +0.3（**基本不动**） |

**为什么离线推算会错两次**：
- 第一次错在把「input 少 34%」直接当成「满价少 34%」—— 被砍的字节里有一大块
  **本来就在断点之后、每轮满价重算**，砍掉它们省的是**全价**；
- 第二次错在反方向：以为被砍的字节都「本来能命中缓存」⇒ 分子分母同减 ⇒ 满价不变。
  —— 也不对：**被砍的是新追加的 executor 行，它们在断点之后，本来就是全价的**。

⇒ 真相：被砍掉的正是**每轮新追加、落在断点之后**的那部分 ⇒ 直接是满价 ⇒ 砍掉即省。
**满价部分 −22.4%**（这才是真金白银那部分）。

⚠️ **命中率百分比不能当收益看**：它基本不动（19.7% → 20.0%）。理由同上 ——
被砍的字节原本就不在命中区里，分子分母同减。

复现：`.venv\Scripts\python.exe scripts\probe_cortex_planner_cache_ab.py 861`

**离线的价值仍然在**：它能定位「断点在哪个块」（§3.1），
只是**不能用来推算命中率/满价** —— 那必须真 API。

---

### 方案 2（次攻）：Executor report 瘦身 —— `actions taken` 去掉 `agent_thought`

> ⚠️ **2026-10-24 决定：用户选择放弃方案 2**，保持「事实层逐字搬运」的契约不变。
> 本节保留作为**下次想做时的现成方案**（细节都齐了），本轮**没有实施**。
> 理由：改契约的风险 > 省下的 ~2,300 tok。

**现状**：末轮 report 1,765 字符，其中 `actions taken` 4 条，每条约 220 字符，
**每条的 `agent_thought` 占约 200 字符**：

```
- tap({"target": {"index": "14"}, "agent_thought": "上一次操作没有改变前台应用，仍停留在桌面。
   屏幕上第 14 行有一个可点击且可聚焦的『番茄畅听』图标（位于 896,239 的 286x310 区域）…"})
```

⇒ 那 200 字符是**执行器的推理**，而 cortex 只需要「试过哪些路线」。
cortex.md 原文：「the report's `actions taken` … say what was already tried」—— 要的是**动作**，不是每次的推理。

#### 4.1 ⚠️ 契约冲突：必须由人裁决

`cortex.py:172::render_executor_report` 的注释写着「事实层**逐字搬运**，不重写」。
直接在渲染时删 `agent_thought` 会改「逐字搬运」的定义。三条路：

| 选项 | 做法 | 评价 |
|---|---|---|
| (i) 放弃方案 2 | 什么都不改 | 最保守，放弃 ~2,350 tok |
| (ii) 渲染时过滤 | `render_executor_report` 里剔掉 `agent_thought` | 快，但**改了契约定义** |
| **(iii) 在 Reporter 采集端定契约**（推荐） | `build_actions_taken` 采集时就只取 `action` + `target`，不取 `agent_thought` | **「逐字搬运」依然成立**（搬的就是 Reporter 采到的原文），砍掉的是执行器的**散文自述**；契约在「采什么」，不在「怎么写」 |

⇒ 推荐 (iii)：改 `agents/reporter/reporter.py::build_actions_taken`。
`agent_thought` 属于**散文**（`thoughts_compaction` 自己也把 cortex 的散文压了、把 executor 的当事实 ——
但 executor 行的**工具回执**是事实，**agent_thought 是推理**，两者混在一行里，
这才是问题根源）。

#### 4.2 预期收益

| 指标 | 现状 | 方案 2a 后 |
|---|---|---|
| 末轮 report | 1,765 字符 | ~945（−46%） |
| 12 次调用 input | 61,518 tok | ~59,200（再 −2,300 tok） |

⚠️ report 在块 6，**每轮满价重算** ⇒ 省的每一轮都实打实。

#### 4.3 `evidence` 先不动

report 的 `evidence` 里是**整屏 OCR 原文**（~700 字符：
`雄工通知：| Android 系统通知：| ull 90 | 5G | 12:45 . | 全网最火的DJ歌曲 | …`）。
cortex.md 规则 3「Never mark a subgoal done without evidence」⇒ evidence 是判完成的依据，
虽然噪声大。**这一项要单独评估**：可以试「只保留与子目标关键词相关的行」，
但那是语义活儿，风险高，本轮不做。

---

### 方案 3（已排除）：调整块的顺序

把「每轮变」的块（status / report）挪到 thoughts **之前**？
❌ 断点会提前到第一个每轮变的块 ⇒ 命中前缀**更短**。
现在的顺序（恒定 → 只追加 → 每轮变）已经是对的 —— 代码注释和
`doc/executor-prompt-composition-and-cache-20261024.md` §3.2 的实测（顺序值 20.9 个点）都是这个结论。

把「只追加」的 plan log 挪到 thoughts 之前？
❌ 日常轮次没变化（plan 不变时断点还在 thoughts 末尾），但 **replan 当轮**会击穿 thoughts 之后全部 ——
`cortex.py:474` 的注释里已经有实测（replan 当轮 cached 2,329 vs 10,411）。**不要动**。

---

## 四、实施顺序与验收

### 4.1 顺序

1. **方案 1**（主攻）→ 单独提交
2. ~~**方案 2a(iii)**（次攻）~~ → **用户决定放弃**：不改「事实层逐字搬运」的契约
3. 跑对照

### 4.2 每步的验收（不许只看命中率）

| 层 | 检查 | 判据 | 结果 |
|---|---|---|---|
| 单测 | `render_thoughts_history(audience="planner")` 的**前缀性质** | 已发字节逐条不变 | ✅ `TestPlannerAudience` |
| 单测 | 默认参数下 `[executor]` 行数 | 与现在**逐条一致**（归档口径不破） | ✅ `test_full_audience_unchanged` |
| 单测 | planner 版保留失败类行 | 构造含 failed 行的样本，断言它还在 | ✅ `test_keeps_failed_executor_line` |
| 单测 | 全量 | 全绿 | ✅ **811 passed / 1 skipped / 0 failed** |
| 真 API A/B | 861 的真实 12 轮 prompt，两种渲染各发一遍 | 满价部分下降、命中率不作为判据 | ✅ 满价 **−22.4%** |
| **真机** | 同任务、同模型、同设备，新代码跑一次 | ① cortex 满价 token 降；② **子目标数不增加**；③ 「judged COMPLETE」不误判；④ 最终状态一致 | ✅ 见 §4.5 |

### 4.3 验收指标（关键）

> **别拿命中率百分比验收** —— 真 API 实测：方案 1 前后命中率**几乎不动**
> （19.7% → 20.0%），而满价部分降了 22.4%。
> 拿 **满价 token（input − cached）** 和 **决策质量（子目标数 / 误判数 / 最终状态）** 验收。

### 4.4 真机对照结果（2026-10-24 14:00）

对照对象：**861**（改前，manager 记录）vs **本地 runner 实跑**（改后，仓库代码）。
同任务（下载番茄畅听）、同模型（minimax/MiniMax-M3.1-Flash）、同设备（Redmi Turbo 4）。

| 口径 | 改前 861 | 改后（新代码） | 变化 |
|---|---|---|---|
| cortex 调用次数 | 12 | 10 | — |
| **input / 次** | 5,126 | 3,878 | **−24.3%** |
| cached / 次 | 1,994 | 1,073 | −46.2% |
| **满价 / 次** | 3,133 | 2,805 | **−10.4%** |
| 命中率 | 38.9% | 27.7% | ← 不是收益指标 |

**决策质量（不退化）**：

| | 改前 861 | 改后（新代码） |
|---|---|---|
| 下发子目标次数 | 6 | 5 |
| judged COMPLETE | 5 | 5 |
| 其它 judged（失败判定） | 0 | 0 |
| 不同子目标数 | 6 | 5 |
| 最终状态 | SUCCEEDED | **COMPLETED**（error=null，番茄畅听下载安装并成功启动） |

⇒ **子目标数没增加、COMPLETE 数一致、零误判、任务完成** —— 决策质量没有退化。

⚠️ **为什么真机的「满价/次 −10.4%」比真 API A/B 的「−22.4%」小**：
真机两次跑的**轨迹不同**（这次 executor 跑了 44 轮 vs 861 的 28 轮 ⇒ 元素表大小、
历史长度都不同），端到端数字里混了轨迹噪声。
**真 API A/B 才是单变量的数字**（同一序列、只变渲染）⇒ 以 **−22.4%** 为收益口径，
真机数字只用来证明「**没有变差**」。

### 4.5 真机对照的坑（踩过的）

- **runner-host 是常驻进程，Python 不会重载改过的模块** ⇒ 代码改了必须**重启它**，
  否则跑的还是旧代码。用 prompt 里的措辞指纹验版本（`scripts/probe_exec_prompt_fingerprint.py`）。
- 重启命令（`runner_host` 不在 mobile-use 的 venv 里，要靠 PYTHONPATH 拼）：

  ```powershell
  $env:PYTHONPATH = 'E:\MyCoding\mobile-manager\runner-host\src'
  Start-Process -FilePath 'E:\MyCoding\mobile-use\.venv\Scripts\python.exe' `
    -ArgumentList '-m','runner_host.main','--port','9199','--host-id','pc-e2e' `
    -WorkingDirectory 'E:\MyCoding\mobile-manager\runner-host' `
    -RedirectStandardOutput 'E:\MyCoding\mobile-manager\runner-host\logs\runner-host.out.log' `
    -RedirectStandardError  'E:\MyCoding\mobile-manager\runner-host\logs\runner-host.err.log' `
    -WindowStyle Hidden
  ```

- **发任务必须显式带渠道**：`POST /api/tasks/{id}/run-now` 带 `{"llmChannelId": N}`，
  否则报「必须先选择模型渠道」。⚠️ 渠道 id 会变（861 跑的时候 id=14 是 MiniMax，
  现在 id=14 是 deepseek-flash，MiniMax 挪到了 id=1）⇒ **别照抄历史 id，先查渠道列表**。
- ⚠️⚠️ **manager 会把任务派给「设备已初始化」的那条 runner 链路** —— 本机 runner
  重启后 slot 是 `agentInitialized=false`（**空闲态的正常值**，`ready = busy or agent_ready`），
  于是任务全被派给**手机上的 runner（phone-begonia）**，那台跑的是**设备上装好的旧版包**。
  实测：这样跑出来的 prompt 里还是旧 header ⇒ 对照完全无效。
  ⇒ **要么绕开 manager 直接 POST 到本地 runner 的 `/api/tasks`**
  （`mobile-manager/scripts/run_local_runner_task.py`），要么确认任务落在本机 runner 上。
- **改完代码必须重启 runner-host**：Python 进程不会重载改过的模块。
  **验证靠指纹**（`scripts/probe_cortex_planner_fingerprint.py` 数新 header 命中数），
  不要靠「我刚改完」或「提交时间在启动时间之后」。

---

## 五、附录：复现命令

```powershell
cd E:\MyCoding\mobile-manager
$env:PYTHONIOENCODING='utf-8'
$py='E:\MyCoding\mobile-use\.venv\Scripts\python.exe'

# 块构成 + 字符级断点
& $py scripts\probe_prompt_cache_breakpoint.py 861 --agent cortex

# thoughts 成分（按角色）
& $py scripts\probe_cortex_thoughts_mix.py 861

# executor 行的成功/失败占比
& $py scripts\probe_executor_line_kinds.py 861

# 末轮各块原文（看 report 里塞了什么）
& $py scripts\probe_cortex_prompt_blocks.py 861

# 逐 agent × 逐调用的缓存命中
& $py scripts\probe_exec_per_agent_cache.py 853 854 860 861
```

相关文档：
- `doc/executor-prompt-composition-and-cache-20261024.md` —— Executor 侧的组成、命中率、
  DeepSeek 缓存规则、Hermes 对照
- `doc/perception-staleness-analysis.md` §7.2 —— 归档分析的事实统计口径（数 `[executor]` 行）
- `doc/thoughts-compaction-plan.md` —— 「位置无关」压缩为什么不会击穿前缀
