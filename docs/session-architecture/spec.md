# 会话架构 P1：局部历史的段边界

> 依据：[会话架构方案v1.md](../会话架构方案v1.md)。P0 已落地并完成真机验收；本文只覆盖 P1。

## 1. P0 真机结论（前置事实）

设备：Redmi Turbo 4（`89R8ZXNBGQIRJRLF`），模型 MiniMax-M3.1-Flash-Preview。四次运行覆盖四条路径：

| 路径 | 结果 | 事件链要点 |
|---|---|---|
| 正常完成 | `turn/end completed` | planner → cortex → 每个子目标一个 executor 会话 → outputter → 层内全部 close → `session/end` |
| 进程被杀 | 无 `turn/end` | 悬空 step 不会自动收尾；resume 必须把「有 `step/start` 无 `step/end`」视为中断 |
| 同子目标换发 | 两个 `subgoal:<id>` 会话 | 前一个 `close(completed)` 后才开下一个，未出现并发打开 |
| 结构化输出 | 多一个 `outputter` 会话 | outputter 仅在 `-o`/`output_format` 时出现，不是每任务固定层 |

结论：四层边界的埋点与真实执行顺序一致；事件日志可作为排障与 resume 的真相源。

## 2. 决策：数据归 state，AgentSession 只记段边界

`executor_messages` **不迁出** LangGraph state。

理由：
- 它是 `add_messages` reducer 字段，也是 checkpointer 的恢复真相；搬进 Python 对象等于切断 P3 的 resume。
- 换取清空已经是显式 `RemoveMessage(REMOVE_ALL_MESSAGES)`，不存在「隐式状态机」这个待修的问题。
- 搬走会绕过 `asanitize_update`（trace 录制、thoughts 归一化都在那里）。

⇒ P1 的价值不是换存储位置，而是**让每段历史在事件流里可定位**：知道哪个会话、多少条消息、从哪一步开始。

## 3. 契约

- `step/end` payload 新增 `history`：该次 emit 之后 `executor_messages` 的条数。仅 executor 的 Step 携带。
- `agent_session/close` payload 新增 `history`：该会话内最后一次 Step 的 `history`；无 Step 时省略。
- 语义：
  - 同一会话内 `history` 单调不减（历史只追加）。
  - 换取（`open_agent_session` 新开会话）后，新会话的 `history` 从清空后的条数开始（≈1，种子消息），**不继承**旧值。
  - `history` 是观测值，不参与任何判定；缺失时下游不得报错。

## 4. 非目标

- 不拆 `agents_thoughts`。它会同时改变 planner（末 N 条）、cortex（压缩素材）、outputter（全量）的 prompt 字节，收益未证；留到有 A/B 手段时再谈。
- 不改 `RemoveMessage` 清空路径，不改图拓扑、预算口径、提示词文案。
- 不做 P2 的组装管线，也不做 P3 的 resume。

## 5. 变更点

| 文件 | 变更 |
|---|---|
| `minitap/mobile_use/runtime/session.py` | `AgentSessionRecord` 增 `history: int`；`StepHandle.end` 从 extra 取 `history` 并记到 record；`_finalize_agent_session` 在 payload 里带 `history`（仅 >0 时） |
| `minitap/mobile_use/agents/executor/executor.py` | `emit()` 计算 `history_after = len(state.executor_messages) + len(messages)`，经 `step.end(..., {"history": history_after})` 落盘；四条出口共用同一份计算 |
| `tests/mobile_use/test_runtime_session.py` | 新增 3 条用例（见 §6） |
| `docs/会话架构方案v1.md` | 章节 9 的 P1 状态改为「已落地」，并记录 §2 的决策 |

接口签名见代码；本文只定边界与语义。

## 6. 验证

单测（离线，不碰设备）：
1. 同一 executor 会话内连续 3 个 Step，`history` 单调不减。
2. `agent_session/close` 的 payload 里 `history` 等于最后一个 Step 的值；无 Step 的会话不带该键。
3. 换取后新会话的 `history` 从清空后的条数开始（显著小于旧会话末值）。

回归：`uv run pytest tests/mobile_use -m "not ios_simulator"`，数量不减、全绿。

真机（`uv run mobile-use ... -o ...`）：一次任务后，`agent_session/close` 的 `history` 与该会话 `step/end` 的数量相称；与改造前相比动作步数不变（P0 已证行为零改动，本阶段同样不改任何判定）。

## 7. 验收门槛

- 全部单测与回归通过。
- 真机一次任务产出含 `history` 的完整事件链。
- `agents_thoughts` 拆分明确推迟，不夹带。

需批准后执行。
