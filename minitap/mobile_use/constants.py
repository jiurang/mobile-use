# 任务默认的动作步预算。
#
# ⚠️ **口径变更（2026-10-21，方案 §11.5 M12）**：这个值以前叫 `RECURSION_LIMIT`、
# 是 400，含义是「LangGraph 的 `recursion_limit`」—— 而那个量数的是**图的
# super-step**（节点执行次数），不是「动作步」。
#
# 改造前实测：400 个 super-step ≈ **66 个动作步**（6.05 super-step/动作步）。
# 现在 `max_steps` 的语义统一成「**动作步**」，由 `sdk/agent.py` 的
# `compute_recursion_limit()` 换算成 super-step 上限。
#
# 所以默认值必须跟着下调 —— 否则同一个默认值的最坏情况花费会涨 6 倍多：
# 一次跑飞的任务从烧 66 步变成烧 400 步，正好和这轮改造的降本目标相反。
#
# **80 是这么定的**：实测新拓扑典型 5.2 super-step/动作步、最坏 9 个
# （`scripts/probe_superstep_budget.py`），80 × 5.2 ≈ 416 —— 和改造前 400 个
# super-step 的预算基本对齐；而每步的开销从「一次带截图的 cortex 调用」变成
# 「一次 executor 调用」，单步比改造前便宜，所以 80 是保守的。
#
# ⚠️ 显式调用 `with_max_steps(N)` 的语义**也**变了：N 现在真的是 N 个动作步
# （改造前 N 个 super-step ≈ N/6 个动作步）。这属于把一个一直被误用的字段
# 拉回它字面上的意思，不是新加的限制。
#
# 🔴 **2026-10-26：上面那句「真的是 N 个动作步」不准确，实话是这样** ——
#
# ``max_steps`` **从来没有**被当成动作步执行过。它只被乘进 super-step 上限
# （``sdk/agent.py::compute_recursion_limit``：``N × 9 + 32``），
# 而真实循环里**一个动作步 ≈ 3 个 super-step**
# （contextor + executor + executor_tools；Summarizer 与 convergence 节点
# 已于 2026-10-29 移除）。
# ⇒ N=80 实际放行 ``(752 − 2 − 2S) / 3 ≈ 243~248`` 个动作步，
# 是名义值的 **约 3.0~3.1 倍**。
#
# 实际最坏是每个动作步 5 个 super-step（每个动作步都自成子目标，K=0）；
# ``SUPER_STEPS_PER_ACTION_STEP`` 保持 9，是比实际最坏更保守的天花板。
#
# 真机实测（mobile-manager execution 880，Summarizer 移除前）：跑了 **186**
# 个动作步才撞上限 —— 正好落在当时 4 super-step/动作步的区间里。
#
# ⇒ **为什么不把它改成「真的按动作步计数」**（那要加一个硬上限）：
# 撞上限现在是**优雅收尾 + 上报**（`sdk/budget.py`），不是崩；
# 而硬砍会把「这个任务确实需要 120 步」的任务腰斩成失败。
# 名字与实不符这件事，靠**文档说实话**解决，不靠加限制。
DEFAULT_MAX_STEPS = 80

EXECUTOR_MESSAGES_KEY = "executor_messages"

# 🔴 **2026-10-27 移除：`EXECUTOR_SCREENSHOT_KEY`（`"executor_screenshot"`）。**
#
# 截图这条通道已整条下线（含按需工具 `take_screenshot`），模型收不到任何图，
# state 里也就没有必要留「当屏截图」这个字段。
#
# 截图本身仍用于 **OCR** 与**坐标校验**（框架内部取数，不进消息）。
#
# 备份 / 恢复说明：`backup/20261027-remove-screenshot-tool/README.md`
