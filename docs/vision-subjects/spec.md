# Vision 感知重设计：Executor 按需调用，vision 返回全部视觉主体

> **2026-10-29 修订：引入感知帧 `ScreenFrame`。**
>
> 下文 3.2、3.3、4.1、4.3–4.6 描述的是「扫描时另截一张图 + state 存视觉表 +
> 扫描前元素表指纹」的旧实现。现在所有感知产物都挂在同一份
> `minitap/mobile_use/perception/frame.py::ScreenFrame` 上：
>
> - `screenshot_base64`、`screen_elements`、`ocr_results` 来自同一次
>   `get_screen_data`，扫描不再自己截屏；
> - `epoch` 由元素表（或 OCR）内容派生，取代 `vision_scan_fingerprint`；
> - `vision_elements` 只属于帧：新帧 `epoch` 相同才继承，不同即作废；
> - `state` 不再有 `vision_elements` / `vision_scan_fingerprint`；
> - 工具回执只报数量，编号表只在每轮 Executor prompt 里出现一次。
>
> 接口也随之改名：`scan_current_screen` → `scan_frame(ctx, frame)`，
> `render_vision_scan_result` → `render_scan_receipt(count)`；
> `VisionElement` / `render_vision_rows` / `resolve_vision` 迁到 `perception/frame.py`。

## 1. 问题

现行设计里，Executor 在 `Target.vision` 写一句话描述，框架据此截图 + 调一次视觉模型
定位。这等于让**看不见图的 Executor 猜「屏上有什么」**，再让 vision 去找它描述的东西：

- 描述不准（模糊、带方位）时，视觉模型会**高置信度地给错框**；真机实测（2026-10-28，
  Redmi Turbo 4 / MiniMax-M3.1）置信度 88~93 的框仍然指向错误对象，`MIN_CONFIDENCE=60`
  挡不住。
- Executor 必须提前知道主体「长什么样、叫什么」，而它只有文字通道，天然做不到。
- 每步最多只能定位一个主体，且定位调用串在执行前，失败就整步不派发。

## 2. 目标

**vision 是感知通道，不是定位工具；但由 Executor 按需触发。**

1. 新增一个无参数工具 `scan_visual_subjects`：Executor 在决策时发现 UI 元素表 / OCR
   不足以确认屏上有什么，就调用它。
2. 该工具**不接受任何目标描述**。它截当屏、让 vision **列出全部视觉主体**
   （图标 / 图片 / 画布 / 无文字控件），编号后作为工具结果返回，同时缓存进 state。
3. Executor 下一步只按编号选：`Target.vision_index: int`。
4. 提示词必须写死一条硬规则：**UI/OCR 不足以支持决策时，必须先扫描，不能凭猜测
   下结论或执行动作；没有百分之百确认，就要先扫描。**
5. 旧的 `Target.vision` 描述路径整体删除（类型、派发前解析、失败回灌、提示词、测试）。
6. `PerceptionConfig.vision` 控制这个工具是否可用，**默认开启**：按需调用的常态成本
   为零，只有真正需要时才付一次截图 + 一次视觉调用。

非目标：不把截图塞进 Executor prompt；不让 Executor 决定「识别什么主体」；不改元素表 /
OCR 的编号契约。

## 3. 新契约

### 3.1 vision 扫描输出（LLM）

```python
class VisionSubject(BaseModel):
    description: str          # 3~15 字，能让人据此选中它（是什么 + 在哪）
    x1: int; y1: int; x2: int; y2: int   # 0-1000 归一化

class VisionScanOutput(BaseModel):
    subjects: list[VisionSubject] = []
```

系统提示词要求：列出所有图标、图片、插画、画布、自绘控件、无文字按钮；跳过纯文字、
段落、时钟、电量；最多 20 项，按重要性排序；**不许编造**，没有就返回空列表。

### 3.2 state 里的缓存表

```python
class VisionElement(BaseModel):
    index: int                 # 1-based，和工具结果 / 提示词里的 [i] 一致
    label: str                 # 即 description
    bounds: ElementBounds      # 已换算成像素
```

- `state.vision_elements`：最近一次扫描的结果；`state.vision_scan_fingerprint`：扫描时
  那一屏的元素表指纹。
- Contextor 每步重建元素表后：指纹一致 ⇒ 保留缓存；不一致 ⇒ 清空（`[]` + `None`）。
  这样「扫描之后屏幕变了」时，旧编号绝不会被继续使用。
- 开关关闭 / 息屏 / 截图失败 / 视觉调用失败 ⇒ `[]`。工具会如实回报原因。

### 3.3 工具结果（回给 Executor 的文本）

```
Visual subjects on the current screen (1-based, valid only for this screen):
[1] "gear icon, top right" (910,120,60x60)

To act on one, use target {"vision_index": 1}. Do not add bounds of your own.
If the screen changes before you act, scan again.
```

无主体：明确说「没有找到视觉主体，不要编造；改用元素表 / OCR」。
失败：明确说原因（息屏 / 截图失败 / 模型不可用），并要求改用文字通道或先处理屏幕。

### 3.4 Executor 提示词（核心）

静态提示词新增一个 **Visual perception** 段落，措辞按用户要求「大文豪」式强化：

- 元素表和 OCR 是文字通道；它们看不见图标、图片、插画、画布和无文字控件，
  也说不清一张图里到底是什么。
- **只要这两条通道不足以让你百分之百确认「这是什么 / 它在哪里」，就不要下结论、
  不要执行动作；下一步必须是 `scan_visual_subjects`。**
- 扫描很便宜（一次截图 + 一次小视觉调用，图不会进入上下文）；点错很贵。
- 扫描返回当屏全部主体，编号只对当屏有效；选一个用 `{"vision_index": i}`。
- 元素表里已经有名字的对象优先用 `index`，不要浪费一次扫描。
- 扫描后屏幕变了就重扫；不要复用旧编号。

工具表只在 `perception.vision` 开启时列出 `scan_visual_subjects`。

## 4. 逐文件改动（mobile-use）

### 4.1 `minitap/mobile_use/services/vision.py` — 重写为扫描服务

保留：`SCREENSHOT_JPEG_QUALITY`、`NORMALIZED_MAX`、`_clamp`、
`bounds_from_normalized_box()`、`capture_screen_base64()`。

删除：`MIN_CONFIDENCE`、`VisionLocationOutput`、`locate_target_by_vision()`、
`_target_with_vision_bounds()`、`resolve_vision_targets()`。

新增：

| 名称 | 职责 |
|---|---|
| `VisionSubject` / `VisionScanOutput` | LLM 结构化输出 schema |
| `VisionElement` | 缓存行模型（index / label / bounds） |
| `MAX_SUBJECTS = 20` | 截断上限 |
| `scan_system_prompt(ctx)` | 中英提示词（`pick`），强调「列全部、别编造」 |
| `render_vision_rows(elements)` | `[i] "desc" (x,y,WxH)`，与解析共用编号契约 |
| `resolve_vision_index(elements, index)` | 1-based 反解，越界返回 None |
| `async scan_screen_subjects(ctx, screenshot_base64)` | 压缩 → 结构化调用 → 过滤/编号；失败返回 `([], reason)`，不抛 |
| `async scan_current_screen(ctx)` | 读电源状态 → 抓图 → 调上面的函数；返回 `(elements, reason)` |
| `render_vision_scan_result(elements)` | 工具结果正文（表头 + 行 + 使用说明） |

### 4.2 `minitap/mobile_use/context.py`

- `PerceptionConfig` 增加 `vision: bool = True`。
- 文档改为「三条通道：元素表 / OCR / 按需视觉扫描」，写明默认开、按需才产生调用、
  图不进 Executor prompt。

### 4.3 `minitap/mobile_use/graph/state.py`

```python
vision_elements: Annotated[list[VisionElement] | None, "...", take_last] = None
vision_scan_fingerprint: Annotated[str | None, "...", take_last] = None
```

### 4.4 `minitap/mobile_use/utils/screen_elements.py`

新增 `element_table_fingerprint(elements)`：对「归一化文字（纯数字噪音归一）+
每行 bounds」排序后哈希。用于判断扫描缓存是否仍然对应当前屏幕：滚动、翻页、弹窗
会变；时钟 / 点赞数跳动不会变。

### 4.5 `minitap/mobile_use/agents/contextor/contextor.py`

- 不扫描（扫描由 Executor 按需触发）。
- 建完新元素表后：`new_fp = element_table_fingerprint(screen_elements)`；
  若 `state.vision_scan_fingerprint == new_fp` 且 `state.vision_elements` 非空 ⇒ 原样保留；
  否则写入 `[]` + `None`。
- update 带上这两个字段。

### 4.6 `minitap/mobile_use/tools/mobile/scan_visual_subjects.py`（新文件）

- `@tool async def scan_visual_subjects(agent_thought, tool_call_id, state) -> Command`
- 文档字符串写给模型：何时用、返回什么、编号只对当屏有效。
- 调 `scan_current_screen(ctx)`：
  - 成功 ⇒ `vision_elements=elements`、`vision_scan_fingerprint=element_table_fingerprint(state.latest_screen_elements)`，
    `ToolMessage(status="success", content=render_vision_scan_result(elements))`；
  - 失败 ⇒ 清空缓存，`ToolMessage(status="error", content="Visual scan failed: ...")`。
- `scan_visual_subjects_wrapper = ToolWrapper(...)`。

### 4.7 `minitap/mobile_use/tools/index.py`

- 注册新 wrapper。
- 新增 `executor_wrappers(ctx)`：基础工具 +（视频录制开启时）录制工具 +
  （`ctx.perception.vision` 时）视觉扫描工具。Executor 节点和 prompt 共用它，
  保证「工具表里有的」和「真的能调到的」完全一致。

### 4.8 `minitap/mobile_use/tools/types.py`

`Target.vision: str | None` → `Target.vision_index: int | None`，描述说明它是工具结果
「Visual subjects」里的 1-based `[i]`，只用于元素表 / OCR 都叫不出名字的对象，
不要与 `index` / `bounds` 同时使用。

### 4.9 `minitap/mobile_use/tools/utils.py`

- `target_key()`：`index` 取 `index or ocr_index`；`vision_index` 以
  `label=f"vision#{vision_index}"` 进入身份。
- `target_label()`：vision 目标显示 `vision#[i]`。
- `has_valid_selectors()`：增加 `vision_index is not None`。
- `resolve_target()`：
  - `index` / `ocr_index` 分支保持原样；
  - 新增 `vision_index` 分支：取 `state.vision_elements`；空 ⇒ 报「no visual subjects
    on this screen — call scan_visual_subjects first」；越界 ⇒ 报有效范围；命中 ⇒
    `model_copy(update={"bounds": element.bounds, "text": element.label})`。
  - 若 `state.vision_scan_fingerprint` 与 `element_table_fingerprint(state.latest_screen_elements)`
    不一致 ⇒ 报「the visual list is from a previous screen; scan again」。

### 4.10 `minitap/mobile_use/tools/bounds_validation.py`

`if target.vision:` → `if target.vision_index is not None:`（视觉框仍跳过 IoU）。

### 4.11 `minitap/mobile_use/agents/executor/prompt.py`

- 删除 `render_vision_failure()`。
- 新增 `render_vision_subjects_message(elements, *, language)`：缓存表非空时渲染
  「本屏视觉主体」块（在元素表之前；元素表必须仍是最后一条消息）。
- `build_executor_messages()` 增加 `vision_elements` 参数。
- `_executor_wrappers` 改用 `tools.index.executor_wrappers`。
- 模块文档同步为「三条文字通道」。

### 4.12 `minitap/mobile_use/agents/executor/executor.py`

- 删除 `resolve_vision_targets` / `render_vision_failure` 的导入与派发前解析分支。
- 工具集改用 `executor_wrappers(self.ctx)`。
- `build_executor_messages(..., vision_elements=state.vision_elements)`。
- `no_perception` 判据改为 `not (ui_hierarchy or ocr or vision)`。

### 4.13 `executor.md` / `executor-zh.md`

- 新增 **Visual perception** 段落（见 §3.4，按「大文豪」语气写足）。
- 工具表增加 `scan_visual_subjects` 行（仅 `perception.vision` 开时渲染）。
- 把 `target.vision` 契约改为 `vision_index`。
- 删除 `vision_failure_message` 宏；新增 `vision_subjects_message(rows)` 宏。
- 中英严格对齐。

### 4.14 历史改写

`rewrite_history_tool_calls()` 删除「vision 目标只留描述、剥掉 bounds」的分支。

### 4.15 SDK / README

- `sdk/builders/agent_config_builder.py::with_perception` 文档补第三条开关。
- `README.md` 若有 `Target.vision` 描述，改为 `vision_index`（实施时确认）。

## 5. 逐文件改动（manager-server）

| 文件 | 改动 |
|---|---|
| `domain/TaskDef.java` | 新增 `perception_vision`（Boolean，可空）+ `isPerceptionVision()`（null ⇒ **true**）+ setter；`applyPerceptionDefaults()` 写入；`hasNoPerception()` 三条全关才 true |
| `web/dto/TaskRequest.java` | 增加 `Boolean perceptionVision` |
| `web/dto/TaskDto.java` | 增加 `boolean perceptionVision` 并映射 |
| `web/TaskController.java` | `apply()`：非 null 才赋值；`previewPrompt()`：`PerceptionPayload` 带上 vision；全关警告改为三条 |
| `web/dto/PerceptionPayload.java` | 增加 `boolean vision`；`defaultPerception()` 传 true |
| `service/ExecutionService.java` | 构造 `PerceptionPayload` 时传 `task.isPerceptionVision()` |

`RunOnceRequest` 不增加该字段（一次性运行使用默认 true）。

若 runner-host 在仓库内映射 `uiHierarchy`，同步增加 `vision`；否则按契约 JSON
`perception.vision` 兼容（缺省 true）。

## 6. 逐文件改动（manager-web）

| 文件 | 改动 |
|---|---|
| `src/types/index.ts` | Task / TaskForm 增加 `perceptionVision: boolean` |
| `src/views/TasksView.vue` | 表单默认 `true`；编辑回填 `row.perceptionVision ?? true`；提交载荷带上；开关列表加「视觉主体（按需）」；全关警告包含 vision |

## 7. 测试改动

1. 重写 `tests/mobile_use/test_vision_target.py`：
   - 归一化框换算；
   - `scan_screen_subjects`：正常映射、非法框/空描述过滤、20 项截断、异常返回 `([], reason)`；
   - `render_vision_scan_result` / `render_vision_rows` 含 `vision_index` 指引；
   - `resolve_target`：命中填 bounds+text；无缓存 / 越界 / 指纹过期给出明确错误；
   - `Target(vision_index=1)` 可用；`bounds_validation` 对其放行；
   - 中英模板都含 `vision_index` 与 `vision_subjects_message`，不含 `vision_failure_message`。
2. `tests/test_prompt_templates.py`：宏名单替换。
3. `tests/mobile_use/test_executor_autonomy.py`：确认无 `Target.vision` 构造点；元素表仍是
   最后一条消息；新增「vision 工具开关」用例（关时工具表无 `scan_visual_subjects`）。
4. manager-server：`ApiContractTest` 增加 `perceptionVision` 默认 true / 可关断言。

## 8. 验证

```powershell
# mobile-use
uv run pytest tests/mobile_use/test_vision_target.py tests/mobile_use/test_executor_autonomy.py tests/mobile_use/test_llm_role_overrides.py tests/test_prompt_templates.py -q
uv run ruff check minitap/mobile_use/services/vision.py minitap/mobile_use/tools/mobile/scan_visual_subjects.py minitap/mobile_use/tools/utils.py minitap/mobile_use/agents/contextor/contextor.py minitap/mobile_use/agents/executor minitap/mobile_use/graph/state.py minitap/mobile_use/utils/screen_elements.py

# manager-server
mvn -q -Dtest=ApiContractTest test   # 目录 manager-server

# manager-web
npm run type-check
npm run build
```

真机（可选）：开启开关，跑一个图库任务，观察（a）执行器是否在不确定时主动扫描，
（b）`vision_index` 命中率，（c）扫描调用次数与每步耗时。

## 9. 风险与取舍

- **按需扫描**：默认开启但常态零成本；只有模型调用时才产生一次截图 + 一次视觉调用。
- **缓存失效**：屏幕指纹变化即清空缓存；模型必须在新的屏上重新扫描。
- **列表质量取决于模型**：`vision` 角色已可单独指定 grounding 更强的模型。
- **iOS / 云设备**：`capture_screen_base64` 失败时工具如实报错，行为等同不可用。
