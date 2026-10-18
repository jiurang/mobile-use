# cortex 提示词中/英 A/B 真机实测

> 实测日期：2026-10-18
> 对象：`minitap/mobile_use/agents/cortex/cortex.md`（EN）vs `cortex.zh.md`（ZH）
> 复现：`probe_out/probe_i18n_ab.py`（实测器）、`probe_out/probe_prompt_parity.py`（字面量校验）

## 为什么是 cortex

全项目 8 个 agent 提示词的 token 底数（tiktoken `cl100k_base` 实测）：

| 提示词 | tokens | 调用频率 |
|---|---:|---|
| **cortex/cortex.md** | **2,270** | **每轮循环必跑** |
| planner/planner.md | 915 | 初始 + 每次 replan |
| executor/executor.md | 339 | 每个 tool-call 轮 |
| video_analyzer.md | 299 | 视频任务 |
| contextor.md | 283 | app-lock 冷门分支 |
| 其余 6 个小文件 | 653 | 偶发 |
| 合计 | 4,759 | |

cortex 一家占 48%，且每轮循环重发一遍 —— 中英对比里唯一有统计意义的标的。

## 翻译口径：只翻散文

`cortex.md` 里的表格、JSON 示例、工具名、字段名（`resource_id`、`swipe_percentages`、
`Decisions Reason`）是**模型必须原样吐出的字面量**。改了就跑不通，token 差会混进功能退化。

所以口径是：散文译中文，所有字面量与代码块保持英文原样。这条有硬校验
（`probe_prompt_parity.py`）：

```
[1] fenced code blocks: en=5 zh=5
    OK 全部逐字节相同
[2] jinja vars: en=14 zh=14
    OK 完全一致
```

## 一、受控测量（最可靠的数）

同一份 context 渲染两版 system prompt，只换语言：

| 变体 | EN | ZH | 差值 | 比值 |
|---|---:|---:|---:|---:|
| 模板（分支全关） | 2,270 | 3,216 | +946 | 1.42x |
| 渲染后（OCR 开、未锁应用） | 2,139 | 3,039 | +900 | 1.42x |

**中文提示词比英文多约 900 token/轮，是英文的 1.42 倍。**

## 二、真机端到端（5 个幂等任务 × EN/ZH）

设备 `hywcdudunrswbihe`（Redmi Note 8 Pro / Android 11），全部节点 `minimax/MiniMax-M2.7`，
两臂唯一差异是 cortex 提示词语言。

| 指标 | EN | ZH | 差异 |
|---|---:|---:|---:|
| 完成任务数 | 2/5 | 3/5 | +1 |
| 期望字面量命中 | 1/5 | 1/5 | 0 |
| cortex 调用轮数 | 69 | 75 | +6（+8.7%） |
| cortex 文本 token | 402,066 | 459,354 | +57,288（**+14.2%**） |
| 全链路文本 token | 440,950 | 488,353 | +47,403（+10.8%） |
| cortex 截图张数 | 69 | 75 | +6 |
| 总耗时（秒） | 951 | 900 | −51 |

逐任务：

| task | 完成 EN/ZH | cortex 轮 EN/ZH | cortex 文本 token EN/ZH | 单次 token EN→ZH | 命中 EN/ZH |
|---|---|---|---|---|---|
| `about` | ❌ / ✅ | 21 / 12 | 112,675 / 76,183 | 5,365 → 6,349 | — / ✅ |
| `calc` | ❌ / ✅ | 10 / 4 | 50,383 / 21,574 | 5,038 → 5,394 | — / — |
| `calc_sub` | ✅ / ❌ | 11 / 17 | 58,894 / 109,608 | 5,354 → 6,448 | ✅ / — |
| `clock_app` | ❌ / ❌ | 24 / 24 | 165,455 / 141,774 | 6,894 → 5,907 | — / — |
| `wifi_state` | ✅ / ✅ | 3 / 18 | 14,659 / 110,215 | 4,886 → 6,123 | — / — |

最终答案（节选，完整见各任务 `.answer.json`）：

- `about` EN 超时；ZH 正确答出「Redmi Note 8 Pro / Android 11」
- `calc` EN 超时；ZH 正确答出 56,088
- `calc_sub` EN 正确完成 100−58；ZH 超时
- `wifi_state` 两臂都答对「Wi‑Fi 打开，已连接 CMCC-evGt」，ZH 的证据链更长更具体

## 三、怎么读这两组数

**端到端 +14.2% 比受控测量的 +900/轮（约 +18%）偏小**，原因是任务执行路径不同：
同一批任务两臂走了不同轮数，累积的 `agents_thoughts`、UI 层级树、OCR 结果都不一样。
单次 token 那列才是可比量，方向一致（ZH 普遍更高）。

**成功率差异不可判定**：5 个任务 × 1 次，样本量不够，而且失败在两臂之间来回翻
（`calc` 中文成功英文超时，`calc_sub` 英文成功中文超时）。这说明**任务级波动盖过了
语言差异**。只能说「没看到中文带来退化」，不能说「中文更好」。

## 四、测量方法与踩过的坑

这一节是为了让结论可复现、可审计 —— 每一条都是「不这么做就会得出错误结论」的。

### 1. 不能用项目自带的 token 埋点

```python
# minitap/mobile_use/services/llm.py:100-105
注意 usage 必须从**原始响应**上取：cortex / planner 等用的是
`with_structured_output`，返回的是 pydantic 对象而不是 AIMessage，
那种情况下读不到 usage，只能记 0 —— 这是已知限制。
```

telemetry 里 cortex 的 token 恒为 0。拿它做成本分析会得出「cortex 不花钱」。

### 2. langchain callback 在结构化路径上不触发

`with_config(callbacks=[...])` 实测行为：

| 调用路径 | `on_llm_end` 是否触发 |
|---|---|
| `llm.invoke()` | ✅ |
| `llm.ainvoke()` | ✅ |
| `llm.with_structured_output(S).invoke()` | ❌ |
| `llm.with_structured_output(S).ainvoke()` | ❌ |

而 cortex 用的正是最后一条。可用方案是**类级同时打 `generate` + `agenerate`**：
同步 `invoke` 走 `generate`，异步 `ainvoke` 走 `agenerate`，只补一个会漏。

两个坑：`MiniMaxChatOpenAI` 是 pydantic 模型，**实例属性赋值直接 `ValueError: no field`**，
只能改类；而且 agent 里实际用到**两个**类（`MiniMaxChatOpenAI` 和 `ChatOpenAI`），
按调用点懒打才保险 —— 预判类名的第一版一个数都没抓到。

### 3. MiniMax 在 agent 路径下不回报 usage

独立探针里 usage 拿得好好的；放进 agent 后实测：

```
llm_output                = {'token_usage': {}, 'model_name': ...}
AIMessage.usage_metadata  = None
AIMessage.response_metadata = {'finish_reason', 'model_name', 'model_provider'}   # 没有 usage
```

那条路径把 usage 丢了，`extract_usage` 三种形状全落空 —— 它返回 None 是正确行为，
不是 bug。**拿不到的数不能当成本结论**，所以本报告改用本地 tiktoken 对真实 messages
计数：两臂同一 tokenizer，比值和差值依然成立，且不依赖任何一方是否老实回报账单。

### 4. 截图必须先剥掉再计 token

截图以 data URL 的 base64 形式传输，直接送进分词器会被拆成几万个无意义 token
（实测单次 cortex 因此虚高到 25k，真实用量约 6k）。两臂会同步虚高、差值不受影响，
但绝对值完全没法解读，所以按 `data:image/...;base64,` 整段剔除、改记张数。

## 五、结论与建议

1. **要控成本，提示词保持英文。** 中文版每轮多约 900 token（+42%），而 cortex 每轮循环
   都会重发，这是纯成本，没有任何收益。
2. **真要中文化的话，省 token 的杠杆不在语言，在篇幅。** 压缩规则散文、合并重复表格、
   砍掉与当前场景无关的分支，比换语言有效得多。
3. **成功率没有证据支持任何一方更好。** 想判定需要每任务重复 3 次以上，当前样本量
   （5×1）只能证明「中文没有明显退化」。
4. **顺带修了一个观测盲区**：项目埋点对 cortex 记 0 token 的问题已定位到
   `_ToolCallStructuredRunnable` 这条路径丢 usage，值得单独开 issue 修。

## 附：复现方式

```powershell
# 字面量零漂移校验（离线，秒级）
.venv\Scripts\python.exe probe_out\probe_prompt_parity.py

# 单臂实测（--only 可指定任务）
.venv\Scripts\python.exe -u probe_out\probe_i18n_ab.py --lang zh --timeout 240
.venv\Scripts\python.exe -u probe_out\probe_i18n_ab.py --lang en --timeout 240

# 汇总对比
.venv\Scripts\python.exe probe_out\probe_i18n_summarize.py
```

切换提示词不需要改代码：

```powershell
$env:MOBILE_USE_CORTEX_PROMPT = "<仓库>/minitap/mobile_use/agents/cortex/cortex.zh.md"
```

不设这个变量时行为与从前逐字节一致（读同目录 `cortex.md`）；路径不可读时只 warning
并回落默认模板。
