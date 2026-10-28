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
# 而真实循环里**一个动作步 ≈ 4 个 super-step**
# （contextor + executor + executor_tools + summarizer）。
# ⇒ N=80 实际放行 ``(752 − 32 − 5S) / 4 ≈ 176~188`` 个动作步，
# 是名义值的 **2.3 倍**。
#
# 真机实测（mobile-manager execution 880）：跑了 **186** 个动作步才撞上限
# —— 正好落在这个区间里。
#
# ⇒ **为什么不把它改成「真的按动作步计数」**（那要加一个硬上限）：
# 撞上限现在是**优雅收尾 + 上报**（`sdk/budget.py`），不是崩；
# 而硬砍会把「这个任务确实需要 120 步」的任务腰斩成失败。
# 名字与实不符这件事，靠**文档说实话**解决，不靠加限制。
DEFAULT_MAX_STEPS = 80

MAX_MESSAGES_IN_HISTORY = 25
EXECUTOR_MESSAGES_KEY = "executor_messages"

#: Executor「当屏截图」单独存，不进局部历史。
#:
#: 为什么单独存
#: ------------
#: 截图的 **图片块会进 provider 的前缀缓存键**（真 API 实测：图在请求中段时，
#: 把它删掉或换一张，命中量立刻掉到「图之前」的位置）。
#: ⇒ 图一旦写进局部历史，它就被固定在历史中段；下一次拍图要「顶替」它时，
#: 那个位置的字节就变了 ⇒ **前缀从历史中段起全部作废**。
#:
#: 真机实测（mobile-manager execution 893）：14 次顶替让 47,181 个本可命中的
#: token 变成满价重算，缓存率从 82% 掉到 41%。
#:
#: ⇒ 图改放 **prompt 末尾**（元素表旁边 —— 那本来就是每轮都变的易变区），
#: 局部历史里一个字节都不动 ⇒ 前缀永远稳定。
EXECUTOR_SCREENSHOT_KEY = "executor_screenshot"
