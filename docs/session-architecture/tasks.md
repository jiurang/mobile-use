# 会话架构 P1 — 实施任务

> 依据：[spec.md](./spec.md)。按顺序执行，每步完成后勾选。

## T1. runtime 记录 history

- [ ] 文件：`minitap/mobile_use/runtime/session.py`
- [ ] `AgentSessionRecord` 增 `history: int = 0`；`_begin_step` 之前不改；`StepHandle.end` 读取 `extra["history"]` 并写回 record（取最大值，单调不减）。
- [ ] `_finalize_agent_session` 的 `agent_session/close` payload 带 `history`（仅 `> 0` 时）。
- [ ] 验收：无 `history` 键时不报错；`enabled=False` 时完全静默。

## T2. executor 落 history

- [ ] 文件：`minitap/mobile_use/agents/executor/executor.py`
- [ ] `emit()` 内计算 `history_after = len(state.executor_messages) + len(messages)`，四条出口统一。
- [ ] `step.end` 的 extra 加 `history`；`done` / `failed` / `rejected` / `action` 四条出口都带。
- [ ] 验收：`step/end` 的 payload 含 `history`，且与 emit 后的历史条数一致。

## T3. 单测

- [ ] 文件：`tests/mobile_use/test_runtime_session.py`
- [ ] 用例 1：同一会话 3 个 Step，`history` 单调不减。
- [ ] 用例 2：`close` payload 的 `history` == 末步值；无 Step 的会话不带该键。
- [ ] 用例 3：换取后新会话 `history` 从清空后条数开始（小于旧会话末值）。
- [ ] 验收：3 条新用例通过，既有 10 条不回归。

## T4. 回归与静态检查

- [ ] `uv run pytest tests/mobile_use -m "not ios_simulator"`
- [ ] `uv run ruff check .` / `uv run ruff format --check .`
- [ ] `uv run pyright`（改动模块）
- [ ] 验收：全绿；既有 2 failed + 4 errors 仍为环境问题，不新增。

## T5. 真机验证与文档

- [ ] 真机跑一次带 `-o` 的任务，检查 `agent_session/close` 的 `history`。
- [ ] `docs/会话架构方案v1.md` 章节 9：P1 标为已落地，补「数据归 state」决策与 `history` 语义。
- [ ] 验收：事件链含 `history`；文档与实现一致。
