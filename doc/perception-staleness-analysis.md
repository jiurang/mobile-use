# 感知块「过时」问题：定位、实测与解决方案

> 调研日期：2026-10-20
> 调研范围：`minitap/mobile_use/agents/{contextor,cortex,executor}`、`minitap/mobile_use/tools/`、
> `minitap/mobile_use/utils/screen_elements.py`、`minitap/mobile_use/utils/failed_targets.py`；
> 配套改动落在 `mobile-manager`（任务级参数下发）与 `mobile-use`（参数落地）
> 方法：**真机实测**（Redmi Turbo 4，serial `89R8ZXNBGQIRJRLF`）+ **真实归档分析**
> （manager 落库的 prompt / 事件流，exec 330~342）。所有数字均为实测值，非估算。
> 复现脚本见文末第八节。

---

## 一、问题现象与最初的假设

一个很自然的问题：**「给 LLM 做决策的屏幕状态，可能部分过时，甚至全部过时」**。

这个假设可以拆成三个可分别检验的子命题：

| # | 子命题 | 隐含的解决方案 |
|---|---|---|
| ① | 屏幕在变，抓屏时它可能正在「抖」 | 等屏幕稳定（固定 400ms）再提取状态 |
| ② | 变化率有上限，超过就不可用 | 量化「多少变化可以接受」 |
| ③ | 剩下的（反复点击没响应） | 智能体设计层解决：主动刷新屏幕状态 |

本记录的核心结论是：**① 在物理上不成立，② 真正的约束不是百分比而是时间，
③ 才是唯一真正要动手的部分**。下面逐条给证据。

---

## 二、关键反直觉结论：屏幕树**根本不抖**

### 2.1 帧间变化（真机连打 dump）

| 场景 | 帧间**身份**变化 | 帧间**位置**变化 |
|---|---|---|
| 上滑后连打 12 帧（~6s，每帧一次 dump ≈ 0.5s） | 0 ~ 3.2% | **恒为 0%** |
| 稳定页（时钟）连打 4 帧 | 0%（1 次 2.9%） | **恒为 0%** |
| 快手推荐流（视频在播、弹幕在滚）连打 6 帧 | 0 ~ 1.3% | **恒为 0%** |

元素总数随页面不同在 120~168 之间，**同一屏内保持不变**。

> **位置变化恒为 0%** 这一条最关键：它意味着「模型看到的位置过期了」这类问题
> **根本不存在**。元素不移动，模型按位置寻址不会偏。

### 2.2 画面在动，树不动

同一个视频播放期间，2~3 秒窗口内：

- **画面**变化 4% ~ 74%（弹幕在滚、画面在动）
- **树**身份变化中位 2.6%（最大 13.3%）

树是 UI 层结构，**不会**因为视频在播而改变；变的只有文字类内容（点赞数、
弹幕文本、时钟）。

### 2.3 结论

> ①「等 400ms 让屏幕稳定」**买不到任何信息**。等 400ms 抓到的树和不等到
> 的是同一份。更糟的是：它修的是一个 0.4s 的问题，而真正让点击打空的是
> 十几秒的量级（见第三节）。

唯一实测到的、值得等的场景是**控件挂载延迟**：上滑换页后，新页面的 like 类
resource-id 在动作后 **2.0s / 2.5s** 才出现在层级树里（此前根本不存在），
约 8s 后又会消失。抓太早会拿到一棵「控件还没挂上」的树，看起来像
「这个按钮不存在」。

---

## 三、真正的时窗：快照年龄

**快照年龄 = 动作落地时刻 − 感知块生成时刻**（≈ LLM 思考 + 工具往返）。

exec 340（快手 12 分钟，28 次动作）实测：

| 指标 | 数值 |
|---|---|
| 快照年龄 | 最小 **8.2s** / 中位 **19.2s** / 最大 **58.5s** |
| cortex 轮间隔 | 6.5s ~ 79.6s |

所以：**400ms 只占这个时窗的 0.7% ~ 5%**。

### 3.1 真正打空点击的是什么

按「thoughts 历史里每条工具结果只写一次」的口径去重（口径见第七节），
四次运行的真实点击结果：

| 执行 | 版本 | 点击结果 | 成功 | 失败 | 失败原因 |
|---|---|---:|---:|---:|---|
| 330 | 基线（抄坐标 + IoU 门限） | 14 | 12 | 2 | 1 次 IoU 拒绝 + 1 次目标不在屏上 |
| 338 | 统一索引 + IoU 门限 | 10 | 9 | 1 | IoU 拒绝（其 id 兜底同时「不在屏上」） |
| 340 | 统一索引 + 身份重定位 | 11 | 9 | 2 | **2 次都是「目标不在屏上」** |
| **342** | **本轮改动后** | **7** | **7** | **0** | — |

> 身份重定位上线后，**失败原因收敛成一类**：不是「点歪了」，而是
> **目标已经不在屏幕上了**。旧版的 IoU 拒绝（模型目测的坐标对不上）消失了。

失败原文（exec 340，两次）：

```
Failed to tap on element. Attempts: resource_id=com.kuaishou.nebula.corona_base_plugin:
    id/lv_bottom_like_view_container: no element with resource_id='...' on the current screen

Failed to tap on element. Attempts: resource_id=com.kuaishou.nebula:id/root:
    no element with resource_id='com.kuaishou.nebula:id/root' on the current screen;
    text='播放视频' (index=0): No element found with text='播放视频'
```

两个 id 背后是**同一个病**：目标所在的页面/容器在点击落地前被 App 换掉了。

**机制（从归档逐帧还原，第一次失败）**：

1. 感知时它在屏幕上，模型也看见了它；
2. 同一时刻，视频进度是 **00:07 / 00:08**（还剩 1 秒）；
3. 模型思考了 **17.0s** 才动手；
4. 点击落地时，视频早已播完、**横屏全屏插件页已被 App 带走** → 当屏上找不到这个 id。

**第二次失败**（`id/root`，feed 的根容器）是同一类：快手 App **自己在横竖屏之间转**
（归档里 `id/groot_view_pager (0,0,2712x1220)` 与 `id/root (0,0,1220x2517)` 交替出现），
页面一转，旧的根容器就没了。

> **一次性页面的寿命（~1s）≪ 决策延迟（中位 19.2s）→ 点击必然打空。**
> 这不是「屏幕在抖」，是**屏幕整个换掉了**。等 400ms 对此毫无帮助。

### 3.2 ② 的答案：可接受的变化率不是一个百分比

约束的真实形式是：

> **决策延迟 < 目标所在页面的剩余寿命**

| 页面类型 | 寿命 | 当前决策延迟（中位） | 是否可行 |
|---|---|---|---|
| 稳定页（列表 / 设置 / 信息流） | 几十秒起 | 19.2s | ✅ 毫无压力 |
| 一次性页（横屏全屏 / 短内容） | **~1s** | 19.2s | ❌ **超标 19 倍** |

树本身的漂移可以忽略（≤3% / 10s，位置 0%），**稳定页上 19s 的决策延迟完全
可以接受**。问题只出在一次性页面上，而那里的解法不是「等得更久」，是
**失败之后能立刻换目标**。

---

## 四、提示词与延迟的关系（为什么最后没做「裁历史」）

### 4.1 最后一轮 cortex 提示词构成（exec 340，call 37，38,715 字符）

| 段 | 字符 | 占比 |
|---|---:|---:|
| system prompt | 6,085 | 16% |
| 设备信息 + 目标 + 计划日志 | 1,419 | 4% |
| **thoughts 历史** | **25,207** | **65%** |
| 子目标状态 | 323 | 1% |
| executor 反馈 | 954 | 2% |
| 设备状态 | 79 | 0.2% |
| 屏幕元素表 | 4,648 | 12% |

thoughts 历史内部构成：`cortex: Decisions reason` 58 条 / 15,654 字符（62%）、
`Goals completion reason` 42 条 / 4,992、orchestrator 9 条 / 2,091、
executor 工具结果 15 条 / 1,668。

### 4.2 延迟到底由什么驱动

| 假设 | 实测相关性 | 结论 |
|---|---|---|
| 提示词越大越慢 | **0.57**（剥掉 `wait_for_delay` 混淆后） | 有趋势，**不是主因** |
| 输出越长越慢 | **−0.20** | 无关 |
| ——反例—— | 提示词 29,887 → **58.5s**；31,284 → **9.4s** | **同尺寸差 6 倍** |

结论：**延迟的主因是模型服务端的波动**，提示词大小只是次要趋势项。

### 4.3 所以「裁剪历史」没做

`cortex.py` 顶部原有的注释已经写明：thoughts 历史**只追加、不截断**，因为
滑窗会让提示词前缀每轮整体左移，**前缀缓存永远命中不了**。

把两件事放一起看：

| 裁历史的代价 | 数值 |
|---|---|
| 满价输入 | 3.2K → **接近整段提示词（约 9.4K token/轮，约 3 倍）** |
| 前缀缓存命中 | 65.9% → **击穿** |
| 换来的延迟收益 | 不确定（同尺寸提示词 9.4s ~ 58.5s，波动主导） |

> **结论：不裁。** 长任务里缓存命中的部分按 cache-read 计价，比滑窗更省；
> 用大约 3 倍的满价输入去换一个被服务端波动主导、且方向不确定的秒数，
> 是负期望。这条被写进代码注释，避免以后有人「顺手优化」回去。

---

## 五、落地的改动

### 5.1 动作后沉降等待（新增，任务级可配）

**语义是「下限」不是「每轮固定等 N 秒」**：距上一次「动过屏幕」的动作，
至少要过这么久才允许再抓一次屏。

| 环节 | 落点 |
|---|---|
| 配置 | `PerceptionConfig.settle_delay_ms`，默认 **3000** |
| 记动作时刻 | `agents/executor/tool_node.py`：工具执行完统一打戳（只有真正动屏的工具才记，`wait_for_delay` / 记笔记不记） |
| 等待 | `agents/contextor/contextor.py::_wait_for_settle`：抓屏前补等**差额** |
| 任务字段 | manager 任务表 `perception_settle_delay_ms`（可空列，`null` → 3000） |
| 契约 | `perception.settleDelayMs` → runner-host `PerceptionConfig` → mobile-use |
| 重建键 | 进 `as_key()`：它绑在 Agent 上，不进键就会「改了不生效」 |
| UI | Manager 任务表单「感知通道」下方新增「动作后等待」(ms)，0 = 关闭 |

> 为什么默认 3s 而不是实测的 1~2s：控件挂载时间随 App 和机型浮动，
> 宁可多等 1s，也不要因为差 300ms 抓了个半成品。
>
> 它是**下限**所以几乎不产生成本：实测决策延迟中位 19.2s，绝大多数轮次
> 「距离上一个动作」早就超过 3s，压根不会触发。

### 5.2 失败文案说清因果（L2）

原话：

```
no element with resource_id='com.kuaishou.nebula.corona_base_plugin:id/lv_bottom_like_view_container' on the current screen
```

真机实测里，模型看到这句话时的反应是**误诊**成「索引指错了人」（原话：
「该索引实际是弹幕文本元素（danmaku_content_text_view），不是点赞按钮」）。
真相是**东西没了**。

新文案（`tools/live_target.py`）明说三件事：东西不在屏上了（页面可能已切走/
视频已播完）、**这不是索引指错了**、该重读屏幕挑现在还在的元素。

### 5.3 提示词规则（L3-1）

`cortex.md` / `cortex.zh.md` 各加两条：

1. 硬规则 2 补一句：executor 说「元素已不在屏上」→ 屏幕变了 → 重读屏、
   换一个**现在还在**的元素，不要再报同一个目标；
2. 「元素定位」段补一段：**索引只属于当前这一屏**（表每轮重建并重新编号，
   上一轮的 `[48]` 这一轮可能是完全不同的元素）。

### 5.4 失败账本 + 熔断（L3-2）

新增 `utils/failed_targets.py`：

- **记账**：失败写进 `state.failed_targets`，成功清零（换了目标，「连续失败」到此为止）；
- **熔断**：**同一目标连失 2 次后，第 3 次直接拒执行**，要求模型换目标
  （省掉一次 1.2s 的 live dump，也不会点了别的东西）；
- **进上下文**：下一轮提示词里出现 `Targets that failed and are no longer on
  screen — do not reuse them: ...`；
- **键的取法**：`index` + 元素身份（resource-id / 可见文字）拼在一起。
  只用 index 不行——索引每轮重编号，只看它会把「模型换了个目标」误判成
  「模型在重复犯同一个错」。

---

## 六、验证

| 项 | 结果 |
|---|---|
| 源仓全量测试 | **354 passed**，失败 6 + error 4 = **与改动前基线逐条一致**（既有的 `test_minimax_provider` / `test_idb_client`） |
| 新增测试 | `test_failed_targets.py`(9) / `test_settle_delay.py`(5) / `test_tap_failure_ledger.py`(3) / `test_executor_tool_node.py`(+2) |
| runner-host | `test_host.py` 29 项全过（含「settle 变化必须重建 Agent」） |
| manager-server | 205 项 Java 测试全过（含新增契约用例） |
| manager-web | `vue-tsc --noEmit` + `vite build` 通过 |
| APK 内容 | 26 个新代码指纹全部命中（拆 `app.imy` 搜符号） |
| 参数端到端 | 老任务读回 3000；写 5000 存得进读得出；0 能关；负数被拒 400；执行事件回报 `...@settle5000ms`（**实际生效值**，非请求值） |
| 真机 12 分钟（exec 342） | 37 轮，**点击 0 失败**；新规则出现在 29 轮提示词里；`perception` 事件 `ui_hierarchy+ocr@settle3000ms` |

---

## 七、统计口径的坑（这一节最值得记住）

从归档里统计智能体行为，**数错得非常容易**，而且错的方向总是「显得更糟」。

### 7.1 同一条工具结果会在之后每轮 prompt 里重复出现

对话历史带着它，所以第 N 轮之后每一轮都带着同一条。

以 exec 340 为例，同一批真实失败：

| 数法 | 结果 |
|---|---|
| 正则数 prompt 里 `"Failed to tap"` 的**出现次数** | **35 次** |
| 按 `tool_call_id` 去重 | 1 次（**也漏了**：那种消息格式只覆盖一部分） |
| **thoughts 历史行首锚定**（每条结果只写一次） | **2 次** ✅ |

- 那 35 次里 34 次是**同一条失败**被后续 34 轮反复携带；
- 差点据此得出「快手点赞按钮有严重活锁，22/24 轮全败」——真相是只失败过 2 次。

> 连「按 id 去重」都不够稳：消息的 JSON 格式（`content="..."` / `content='...'`）
> 在不同来源里不一致，按固定正则取会漏。**先找到「每条行为只写一次」的
> 那个不变量，再数。**

### 7.2 权威口径：thoughts 历史里每个工具结果只写一次

`state.agents_thoughts` 是**只追加**的，每次工具调用写一行 `[executor] <结果>`。
最后一轮提示词带着完整历史 → **数历史里 `[executor]` 开头的行**即可，每条只出现一次。

且必须**行首锚定**：模型的「Decisions reason」行里也会提到
"Tap on element"，不锚定会把理由行算成一条结果。

### 7.3 另外两个坑

| 坑 | 现象 | 正解 |
|---|---|---|
| 一段反馈文本里含 N 条结果 | 只取第一个 `tool_call_id` → N 条算成 1 条 | 用 `findall` 逐条取 |
| `responseJson` 里的 action 计数 | 同一份 decisions 在响应里出现两次 → 派发数翻倍 | 派发数不可靠，以工具结果为准 |

> 一句话：**归档里的重复出现 ≠ 行为发生次数。** 先找到「每条行为只写一次」
> 的那个不变量，再数。

---

## 八、复现方法

### 真机探针（`mobile-manager/probe_out/`）

| 脚本 | 干什么 |
|---|---|
| `probe_screen_stability.py` | 连打 dump，量帧间身份/位置变化 |
| `probe_settle_time.py` | 动作后连打 12 帧，对照 3 秒后的参考帧，测沉降 |
| `probe_like_container.py` | 记录 like 类 id 在动作后的挂载/消失时刻 |
| `probe_corona_page_life.py` | 一次性页面的寿命（进不去时用归档替代） |
| `probe_snapshot_age.py` / `probe_action_age.py` | 算快照年龄 |
| `probe_latency_driver.py` / `probe_latency_clean.py` | 延迟驱动因素相关性（剥掉 `wait_for_delay`） |
| `probe_prompt_composition.py` / `probe_thoughts_composition.py` | 提示词分段构成 |
| `count_tap_results.py` | **点击成败的权威口径**（thoughts 历史行首锚定，逐条打印） |
| `dedupe_taps_final.py` / `compare_340_342.py` | 四次运行的同口径对比 |
| `dump_fail_texts.py` / `dump_executor_history.py` | 失败原文 / executor 历史原样导出 |
| `check_new_behavior.py` / `count_new_features.py` | 新规则 / 失败账本是否进了提示词 |
| `verify_settle_delay_e2e.py` | 任务级参数端到端 |

### 归档分析

```
GET  /api/executions/{id}                  # 状态 / token
GET  /api/executions/{id}/prompts          # 每轮 prompt 的元信息（含 ts）
GET  /api/executions/{id}/prompts/{callIndex}   # promptJson / responseJson
GET  /api/executions/{id}/events?limit=500      # 事件流（不全，别指望）
```

> prompt 正文**不是** `{"content": "..."}`，而是
> `{"role": ..., "blocks": [{"type":"text","text":"..."}]}`，工具结果还裹在
> 转义 JSON 里。用 `messages[].blocks[].text` 取，别用 `"content"` 正则。

---

## 九、遗留与建议

1. **决策延迟仍是最大杠杆**。中位 19.2s 对一次性页面（寿命 ~1s）仍然超标。
   本轮没动它（原因见 4.3），但压缩**轮次**（每轮一个动作）是更划算的方向：
   轮次减半，快手这类短内容任务才有希望。
2. **快手点赞不必切横屏全屏页**：信息流里就有稳定的
   `like_button` / `like_element_click_layout`（实测 9 秒内不消失），
   切进去反而制造一次性页面。属于任务提示词层面，可按需加一句。
3. **OCR 通道的漂移单独量**。本记录量的是树；OCR 行跟的是画面（弹幕、
   计数），漂移率必然更高，但它们恰好是「变了也不影响寻址」的内容
   （无 id 的行只占模型点击的 4%）。
4. 失败账本目前只覆盖 4 个点击类工具（`tap` / `long_press_on` /
   `focus_and_input_text` / `focus_and_clear_text`）。滑动、返回这类动作
   没有目标可记，暂不纳入。

---

## 附：相关文件

| 文件 | 作用 |
|---|---|
| `minitap/mobile_use/utils/failed_targets.py` | 失败账本（记账 / 熔断 / 渲染） |
| `minitap/mobile_use/utils/screen_elements.py` | 统一索引的元素表（上一轮） |
| `minitap/mobile_use/tools/live_target.py` | 按身份在当屏重定位 + 失败文案 |
| `minitap/mobile_use/agents/executor/tool_node.py` | 动屏动作打戳 |
| `minitap/mobile_use/agents/contextor/contextor.py` | 抓屏前的沉降等待 |
| `minitap/mobile_use/agents/cortex/cortex.md` / `.zh.md` | 索引有效性 / 失败后换目标 |
| `minitap/mobile_use/graph/state.py` | `last_action_at` / `failed_targets` |
| `minitap/mobile_use/context.py` | `PerceptionConfig.settle_delay_ms` |
