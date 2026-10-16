# 截图降分辨率：调研结论与实施方案

> 状态：**改动 1、2 均已实现并通过真机验证**。改动 3（telemetry）、改动 4 待做。
> 真机：Redmi Turbo 4（`89R8ZXNBGQIRJRLF`，1220×2712，Android 16）。
> 真机验证脚本：`scripts/verify_bounds_validation.py`、`scripts/verify_screenshot_scale.py`。

---

## 一、结论

**截图不是点击坐标的来源，但它是坐标正确性的隐式依赖。**

等比例缩小本身不会破坏任何代码路径——全仓库没有任何一处从截图像素反算坐标。真正的风险是：**坐标来自 UI 层级树，而层级树和截图在模型脑子里是绑定的**。模型看的是缩小后的图，手里抄的是层级树的原始坐标，两者一旦对不上，`tap.py` 的边界校验形同虚设，失败形态是**静默点错**而非报错。

因此建议：**先堵静默失败的洞，再降分辨率**。两步都做完，缩放才是安全的。

---

## 二、截图的用途与流向（代码实证）

### 唯一给 LLM 看的路径

```
uiautomator2 screenshot (PNG, 原生分辨率)
  └─ clients/ui_automator_client.py:307 → state.latest_screenshot
       └─ agents/cortex/cortex.py:97-102
            ├─ get_compressed_b64_screenshot()  ← 唯一压缩点
            └─ utils/conversations.py:22-31      ← image_url 消息，不带尺寸元数据
```

- **Executor 看不到截图**：`agents/executor/executor.py:52-57` 的消息只有 system + `cortex_last_thought` + `structured_decisions` + tool 消息，无任何图片。
- `cortex.py` 用完即清：`cortex.py:144-145` 把 `latest_ui_hierarchy` / `latest_screenshot` 置 `None`。

### 坐标的真实来源

```
层级树 bounds（设备像素，来自 uiautomator XML）
  → cortex.md:53-54 指示模型把 (x,y,WxH) 抄成 bounds 对象
  → CortexOutput.decisions（字符串化 JSON）
  → executor.py:55 传给 Executor
  → tools/types.py:22  Target.bounds : ElementBounds
  → tap.py:53-70   bounds 优先 → get_center() → tap_at(x, y)
```

设计意图在 `cortex.md:36-37` 写得很清楚：

| Sense | Use For | Limitation |
|---|---|---|
| UI elements | Find elements by resource-id, text, **bounds** | No visual info |
| Screenshot | Visual context, verify visible, 颜色/徽标/图标 | **Can't reliably extract precise element coordinates from pixels** |

**截图负责「判断状态」，坐标负责「定位」，两者分工。** 这个设计是缩放可行的前提。

---

## 三、真机实测数据

### 分辨率与 token 计价

截图分辨率 = `wm size` = **1220×2712**，坐标系一致（层级树 bounds 与截图像素同源）。

Gemini tile 计价：`(w//768+1) × (h//768+1) × 258`

| scale | 尺寸 | tiles | tokens | JPEG q50 | vs 原始 PNG |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 1220×2712 | 2×4 | **2,064** | 172,223 B | 92.9% |
| 0.75 | 915×2034 | 2×3 | 1,548 | 117,449 B | 95.2% |
| **0.50** | **610×1356** | **1×2** | **516** | **69,174 B** | **97.1%** |
| 0.40 | 488×1084 | 1×2 | 516 | 50,542 B | 97.9% |

原始 PNG 2,425,553 B。

**关键发现：0.4 与 0.5 的 token 完全相同（都是 1×2 tile）。**
低于 0.5 之后继续缩小只省带宽，**不再省 token**。缩放的目标应该是「跨过 tile 边界」，而不是「越小越好」。

- 0.5 档：**省 75% token**（2,064 → 516），是唯一有意义的台阶。
- 0.75 档只省 25%，因为宽度 915 仍占 2 列 tile。

> ⚠️ tile 计价是**公式估算**，不是 API `usage` 实测。真实值需接 telemetry 校验（见第六节）。

### 0.5 档肉眼可读性（人工比对 `probe_out/screenshot_scale/`）

**仍然清晰**：底部导航「首页 / 朋友 / 去赚钱 / 我」、互动数「4.5万 / 994 / 5813 / 3735」、正文「@摄影鹿」「想逗你开心 #喜爱度激励计划」、头像上的小红点数字「15」「1」。

**开始发虚**：
- 底部「开宝箱」角标文字 —— **糊到勉强**
- 顶部导航「同城 / 商城 / 生活」灰色文字 —— 明显失真

**结论：0.5 是临界点，不是舒适点。** 若任务涉及小字角标、灰字导航、状态图标判别，建议 0.6~0.75；若只依赖底部导航和正文，0.5 安全。

---

## 四、真正的风险：静默点错

这是本次调研最重要的发现，**优先级高于降分辨率本身**。

假设模型未遵循提示词、真的从缩小后的图上目测坐标（必然偏小）：

```
tap.py:58   validate_coordinates_bounds() → 只检查 0 ≤ x < device_width
            半分辨率坐标当然在屏幕内 → 校验通过
tap.py:70   input tap 300 900 → 设备返回成功，error=None
tap.py:71   success = True
tap.py:152  "Tap on element with coordinates (300, 900) was successful."
```

**点到了错误的地方，但工具报告成功。**

三重放大：

1. `bounds` 排在 fallback 链**第一位**（`tap.py:52` 注释：`Try with COORDINATES FIRST`），一旦「成功」，第 2/3 级 `resource_id` / `text` 兜底**根本不会执行**。
2. `ExecutorToolNode._has_tool_call_failed`（`tool_node.py:159-162`）看到 `status="success"`，不会中断后续 tool call，cortex 会基于假成功继续推进。
3. `cortex.md:83` 假设的纠错信号是 `"Out of bounds" = stale bounds`，而半分辨率坐标**永远触发不到**这个信号，提示词里的自纠机制被整条绕过。

**缩放只是把这个隐患从「理论」变成「高概率」。**

---

## 五、改动点

### 改动 1：坐标一致性校验 ✅ 已完成

**目标**：让「缩放安全」从提示词请求变成代码保证。

**落地文件**：
- `minitap/mobile_use/tools/bounds_validation.py`（新增，校验核心）
- `minitap/mobile_use/tools/mobile/tap.py:52-90`（接入）
- `minitap/mobile_use/tools/mobile/long_press_on.py:51-88`（接入）
- `tests/mobile_use/test_bounds_validation.py`（新增 12 条单测）

**最终判据：归一化 IoU >= 0.9**

真机验证结果（`scripts/verify_bounds_validation.py`，4 屏 68 个可操作节点）：

```
好坐标接受: 68/68 = 100.0%
坏坐标拦截: 68/68 = 100.0%
PASS: 零误拒 + 零漏放
```

#### 判据是怎么定出来的（四次失败才收敛）

这一段值得完整记录，因为它推翻了我最初的方案三次：

| 轮次 | 判据 | 好坐标接受 | 坏坐标拦截 | 结论 |
|---|---|---:|---:|---|
| 1 | 点是否落在某节点内 | 100% | **0%** | 废：31 个大容器使条件恒真 |
| 2 | 中心点命中 + 面积门限 0.6 | 100% | **0%** | 废：容器面积与控件重叠 |
| 3 | 纯 IoU >= 0.9 | 100% | 100% | 废：外扩 10% 即掉到 0.826，与坏坐标上界撞车 |
| 4 | 中心点命中最佳匹配节点 | 100% | **0%** | 废：半尺寸框匹配到大容器，中心点自然落在里面 |
| **5** | **归一化 IoU >= 0.9** | **100%** | **100%** | **采用** |

**关键在「归一化」**：先把模型给的框按最佳匹配节点做尺度对齐（以中心为锚点），
再比 IoU。尺寸漂移被完全消除，IoU 只反映位置差异。

```
归一化 IoU  good: min=1.000 max=1.000     ← 对齐后必然完全重合
            bad:  p50=0.322 max=0.816     ← 上界 0.816
                    阈值 0.90，两侧各留 ~0.08 余量
```

**面积判据为什么走不通**（`scripts/probe_container_area.py`，4 屏 68 可操作 / 454 其它节点）：
真控件最大 **75.92%**（整个设置列表）、时钟 21.44%，而最小的大容器只有 5.35%，
两者面积严重重叠。阈值 0.05 时容器混入 330/454、控件只保留 40/68 —— 无解。

**失败即放行**：dump 失败或返回空时**不阻断点击**。校验器自己出错不能导致 agent 全面瘫痪，
宁可放过一次坏坐标。

**接入方式**：`tap.py` 在原有屏幕矩形校验之后、真正 `tap_at` 之前插入校验，
失败时把原因写进 `attempts`（保留排查痕迹），并自然落到第 2/3 级
`resource_id` / `text` 兜底 —— 这正是原设计想要的降级。

### 改动 2：缩放 ✅ 已完成

**落地文件**：
- `minitap/mobile_use/utils/image_compress.py`（新增，共用实现）
- `android_controller.py` / `ios_controller.py` / `cloud_device_controller.py` —— 三处重复实现收敛为一行调用
- `utils/recorder.py:22-30` —— 录制路径显式不缩放
- `tests/mobile_use/test_image_compress.py`（新增 21 条单测）

**真机端到端结果**（`scripts/verify_screenshot_scale.py`）：

```
LLM 路径（cortex.py:99）:   640x1422   1x2 tile   516 token   47.6 KB
录制路径（recorder.py:26）: 1220x2712  原分辨率   —           111.4 KB
等比例偏差: 0.00022
节省: 2064 -> 516 token (-75%)
```

#### 默认值为什么是 640 而不是 768

这是实测踩到的坑。视觉接口按 `(w//768+1) * (h//768+1) * 258` 计价，
「压到 tile 边长 768」这个直觉是**错的**：

| 宽度 | 等比高度 | tiles | tokens |
|---:|---:|---:|---:|
| 1220 | 2712 | 2×4 | 2,064 |
| 768 | 1707 | **1×3** | 1,548 ← 只省 25%，高度多吃一行 |
| 690 | 1534 | 1×2 | 516 |
| **640** | **1422** | **1×2** | **516** ← 默认 |

要落进 2 行，高度必须 ≤1536，对应宽度上限 690。取 640 留余量，且仍是 1×2。
见 `scripts/probe_tile_target.py`。

**「越小越好」也是错的**：0.5 档（610×1356）与 0.4 档（488×1084）**同为 1×2 tile，
同为 516 token** —— 同一 tile 内继续缩小只省带宽不省 token，纯白丢清晰度。
但要跨到 1×1（省一半），宽高都得 ≤768，等比下宽度要压到 345，清晰度不可接受。

#### 可读性实测

640×1422 下用真机**通知栏**验证（比抖音首页更能说明问题 —— 通知栏天然全是纯视觉线索：
状态栏时间/日期/运营商、各家 App 小图标颜色、每条通知的日期、灰色摘要文字）：

**全部清晰可读**，包括「2026/10/16」这种小字和「点击即可关闭 USB…」的灰字摘要。
抖音界面下的已知薄弱点：「开宝箱」角标与顶部灰字导航发虚。若任务强依赖这两类信息，
把 `DEFAULT_MAX_WIDTH` 调到 700~750 换清晰度（省 25%）。

#### 三个必须守住的点

1. **必须等比例**。`compute_scaled_size` 两个上限取更严格者，保证宽高比不变。
   只缩宽或只缩高会让图像与层级树坐标语义彻底错乱 —— 而坐标校验依赖两者同源。
2. **录制路径不能缩**。`recorder.py` 存的是复盘证据，默认改成显式
   `max_width=None, max_height=None`。否则排查 bug 时拿到一张糊图。
3. **RGBA 修复顺带普及**。原 iOS controller 单独处理 alpha 通道（PNG 透明），
   Android / 云设备没有。收敛到共用函数后三端都受益。

> 顺带修掉一个隐患：Pillow 9.1 起重采样枚举从 `Image.LANCZOS` 移到
> `Image.Resampling.LANCZOS`。仓库锁 `pillow>=11.0.0`，直接用新路径。
> 旧写法在单测里能过（残留别名），只有 pyright 静态检查能抓到。

---

## 六、前置条件

### 1. 接 token telemetry ✅ 已完成

**为什么必须先做**：`doc/token-context-analysis.md:301-305` 早就指出，
本报告所有 token 数字都是**离线公式估算**（截图按 Gemini tile 公式、
文本按 tiktoken），不接 usage 就无法验证端到端收益，也无法证明质量没掉。

**落地**：
- `minitap/mobile_use/utils/llm_usage.py`（新增）—— token 解析
- `minitap/mobile_use/services/telemetry.py` —— 新增 `capture_llm_usage`
- `minitap/mobile_use/services/llm.py:34-100` —— 在
  `invoke_llm_with_timeout_message` 统一埋点
- 8 个 agent 的 15 处调用点传入 `agent_name` / `task_id` / `used_fallback`

**关键设计：埋点放在唯一汇聚点**

`invoke_llm_with_timeout_message` 是**所有** agent 调 LLM 的必经之路。
埋在这里，一处改动覆盖 cortex / executor / planner / orchestrator / contextor /
outputter / hopper / video_analyzer，不必逐个 agent 改，也不怕以后新增 agent 漏埋。

**复用而非重写**：`tui/events.py` 里本来就有一套成熟的 usage 解析（覆盖
langchain 各版本与 provider 的多种形状，还有单测锁定）。把它提到
`utils/llm_usage.py` 让两边共用同一份 —— usage 形状是最容易漂移的部分，
两处各写一份必然分叉。TUI 原有 3 条测试继续通过，证明重构未改变行为。

**真实链路验证**（`scripts/verify_telemetry_usage.py`，DeepSeek 实际调用）：

```
provider: deepseek/deepseek-chat
  model          deepseek-flash
  input_tokens   5
  output_tokens  111
  duration       1.40s
  used_fallback  False
PASS
```

**三条安全约束**（都有单测锁定）：
1. **埋点异常绝不吞掉主流程** —— `capture_llm_usage` 抛异常时，LLM 结果照常返回。
2. **读不到 usage 记 0 而不是报错** —— OpenAI 兼容端点可以不返回 usage。
3. **没传 `agent_name` 就不记录** —— 避免把调用归到 `unknown` 污染统计。

**已知限制**：`with_structured_output` 返回 pydantic 对象而非 AIMessage，
那种调用读不到 usage，只能记 0。cortex / planner / contextor / outputter /
orchestrator 都走这条路。**趋势分析可用，绝对值偏低**，后续若要精确需在
provider 响应层取数。

### 2. 真机 A/B

固定 4~6 个任务，原分辨率 vs 压缩后各跑 N 轮，比对：
- `tap` 成功率（`capture_executor_action` 的 `success`）
- subgoal 完成率
- 「点了但没反应」的静默失败次数（需人工抽查录像）

任务集要覆盖：纯层级树可定位的、依赖小字判别的、依赖颜色/徽标的。

**视觉质量阈值是经验问题，代码读不出来，这一步不能省。**
现在有了 usage 埋点，可以把「省了多少 token」量化；但「质量掉没掉」仍然只能实测。

---

## 七、顺带发现

**`contextor.py:74` 写入了一个 `State` 里不存在的字段**：

```python
"screen_size": (device_data.width, device_data.height),
```

`graph/state.py:36-41` 的 `State` 只定义了 `latest_ui_hierarchy` / `latest_screenshot` / `focused_app_info` / `device_date`，**没有 `screen_size`**。该键被静默丢弃，无人读取。

与本次改动无直接关系（尺寸走 `ctx.device.device_width/height`，`controller_factory.py:29-30`），但属于同类问题：死写入会误导后来者以为尺寸有 state 来源。建议独立提交清理。

---

## 八、执行顺序与当前状态

| # | 动作 | 收益 | 风险 | 状态 |
|---|---|---|---|---|
| 1 | 接 token telemetry | 验证手段本身 | 无 | ✅ 真实链路验证通过 |
| 2 | **坐标一致性校验** | 堵住静默点错 | 低 | ✅ 真机 100%/100% |
| 3 | **截图降分辨率** | **−75% 截图 token** | 中 | ✅ 真机验证通过 |
| 4 | 清理 `screen_size` 死写入 | 无直接收益 | 无 | 待做（1 行，可随时做） |
| 5 | 真机 A/B | 验证质量没掉 | 中 | 待做，**现在有数据可量化了** |

## 九、验证记录

**真机回归 —— 坐标校验**（`scripts/verify_bounds_validation.py`，4 屏）：
```
好坐标接受: 62/62 = 100.0%
坏坐标拦截: 62/62 = 100.0%
PASS: 零误拒 + 零漏放
```

**真机回归 —— 缩放**（`scripts/verify_screenshot_scale.py`）：
```
LLM 路径:   640x1422    1x2 tile   516 token    47.6 KB
录制路径:   1220x2712   原分辨率                111.4 KB
等比例偏差: 0.00022
PASS
```

**真实 LLM 链路 —— telemetry**（`scripts/verify_telemetry_usage.py`，DeepSeek）：
```
model: deepseek-flash   input: 5   output: 111   duration: 1.40s   fallback: False
PASS
```

**单测**：新增 46 条，全绿。
- `test_bounds_validation.py` 12
- `test_image_compress.py` 21
- `test_llm_usage_telemetry.py` 13

**全量测试**：基线 `6 failed, 64 passed` → 改动后 `6 failed, 110 passed`。
存量失败为 `test_minimax_provider`（MagicMock 注入）与 `test_idb_client`（需 macOS），与本次改动无关。

**类型检查**：pyright 对改动文件 `0 errors, 0 warnings`。
全仓仍有 4 个存量 error，全在 `clients/wda_client.py:246,259` 的
`os.killpg` / `os.getpgid` —— POSIX 专属符号，iOS 走 WebDriverAgent 只在 macOS 跑，
本机 win32 静态分析看不到，**非本次引入**。

**Lint**：ruff `All checks passed!`（全仓 minitap / tests / scripts）。
