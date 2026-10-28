# Vision 视觉主体感知 — 实施任务

> 依据：[spec.md](./spec.md)。按顺序执行，每步完成后勾选。

## T1. vision 服务重写（扫描 + 编号表）

- [x] 文件：`minitap/mobile_use/services/vision.py`
- [x] 保留坐标换算与截图快路；删除定位 API（`VisionLocationOutput`、
  `locate_target_by_vision`、`resolve_vision_targets`、`MIN_CONFIDENCE`）。
- [x] 新增 `VisionSubject` / `VisionScanOutput` / `VisionElement` / `MAX_SUBJECTS` /
  `scan_system_prompt` / `render_vision_rows` / `resolve_vision_index` /
  `scan_screen_subjects` / `scan_current_screen` / `render_vision_scan_result`。
- [x] 验收：单文件可导入；扫描异常返回 `([], reason)`，不抛。

## T2. 开关、state 字段与屏幕指纹

- [x] 文件：`minitap/mobile_use/context.py`、`graph/state.py`、`utils/screen_elements.py`
- [x] `PerceptionConfig.vision: bool = True`；`State.vision_elements` / `vision_scan_fingerprint`；
  `element_table_fingerprint(elements)` / `scan_cache_fingerprint(elements, ocr)`。
- [x] 验收：默认开启；指纹对时钟/点赞数不敏感，对滚动/翻页敏感。

## T3. Contextor 缓存校验

- [x] 文件：`minitap/mobile_use/agents/contextor/contextor.py`
- [x] 每步建表后比对指纹：一致保留，变化清空。
- [x] 验收：扫描后不动屏，缓存跨步保留；动作改变屏幕后缓存为空。

## T4. 按需扫描工具

- [x] 文件：`minitap/mobile_use/tools/mobile/scan_visual_subjects.py`（新建）、
  `tools/index.py`
- [x] 无目标描述参数；结果写回 state 缓存；成功/无主体/失败三种回执；
  `executor_wrappers(ctx)` 统一工具集（Executor / prompt / graph / cortex / planner 共用）。
- [x] 验收：开关关时工具不存在；开时可被 dispatcher 调用。

## T5. Target / 解析 / 校验

- [x] 文件：`tools/types.py`、`tools/utils.py`、`tools/bounds_validation.py`
- [x] `vision_index` 取代 `vision`；`resolve_target` 反解并检查指纹过期；账本键和标签；
  IoU 放行。
- [x] 验收：命中填 bounds+text；无缓存 / 越界 / 过期给出明确错误文案；不静默退化。

## T6. Executor prompt / 节点 / 模板

- [x] 文件：`agents/executor/prompt.py`、`agents/executor/executor.py`、
  `executor.md`、`executor-zh.md`
- [x] 「UI/OCR 不足时先扫描」硬规则；视觉主体块渲染在元素表之前；删除派发前定位与
  失败回灌；工具表按开关渲染；删除旧宏、新增新宏。
- [x] 验收：元素表仍是最后一条消息；中英模板对齐；硬规则措辞明确。

## T7. mobile-use 测试

- [x] 文件：`tests/mobile_use/test_vision_target.py`（重写）、
  `tests/test_prompt_templates.py`、`tests/mobile_use/test_executor_autonomy.py`
- [x] 覆盖：框换算、扫描映射/过滤/截断/异常、vision_index 解析与过期、IoU 放行、
  模板契约、工具开关与无参数派发。
- [x] 验收：相关 pytest 套件全绿（148 passed）。

## T8. manager-server 全链路

- [x] 文件：`TaskDef.java`、`TaskRequest.java`、`TaskDto.java`、`TaskController.java`、
  `PerceptionPayload.java`、`ExecutionService.java`；
  `runner-apk/app/src/runner_host/models.py`、`host.py`。
- [x] 默认 true；preview / 执行载荷 / DTO 全带上；runner-host 重建键与翻译同步。
- [x] 验收：`ApiContractTest` 17/17；runner-host `test_host.py` 36 passed。

## T9. manager-web 任务表单

- [x] 文件：`src/types/index.ts`、`src/views/TasksView.vue`
- [x] 类型、表单默认值、编辑回填、提交载荷、开关项、全关警告。
- [x] 验收：`npm run type-check` 与 `npm run build` 通过。

## T10. 回归与真机

- [x] 跑 T7/T8/T9 全部验证命令。

### 后续可选（需真机与用户参与，本次未执行）

- 开启开关跑一次图库任务，记录扫描触发时机、主体列表、命中率、每步耗时；
  确认关闭时零额外调用。

> 全量 `tests/mobile_use` 除环境原因外全绿：`test_siliconflow_provider` 因 API Key
> 401 失败、`test_idb_client` 因未设置 `IOS_UDID` 跳过失败，均与本次改动无关。
