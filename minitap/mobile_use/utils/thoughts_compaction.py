"""Agent thoughts history 的压缩：发给 LLM 的那份，压掉散文，保留事实。

为什么要单独一个模块
--------------------
`state.agents_thoughts` 是只追加的，长任务里它会线性膨胀。实测（真机归档
exec 212，122 条真实 thoughts）：单条中位 133 token、末轮 thoughts 段占整个
cortex prompt 的 65%，是 prompt 里最大的一段，也是唯一的 O(n) 项。

但它**不能直接截断**：`cortex.py` 的注释和 `doc/perception-staleness-analysis.md`
§4.3 都写明，滑窗会让提示词前缀每轮整体左移 → 前缀缓存永远命中不了 → 满价
输入涨约 3 倍。实测滑窗命中率只有 47.2%（基线 93.5%）。

本模块的做法
------------
**位置无关**的规则压缩：每一条 thought，无论新旧，都用**同一条规则**渲染。

这一条是全部收益的来源。它保证：

    f(ts[:k]) 恒为 f(ts[:k+1]) 的字节前缀

也就是「已经发出去的字节永不改变」—— 追加语义完整保留，前缀缓存不因压缩而
击穿（实测命中率 91.1%，基线 93.5%；满价重算反而比基线低 9.2%）。

对比一下就知道差别在哪：

- 滑窗 / 摘要重写是**位置相关**的（第 26 条进来，第 1 条的渲染结果变了）
  → 位置相关 → 周期性击穿。
- 本模块是位置无关的 → 位置无关 → 永不击穿。

分类处理，不是一刀切
--------------------
thoughts 里有三类内容，性质完全不同：

| 角色 | 内容 | 性质 | 处置 |
|---|---|---|---|
| `cortex` | `Decisions reason` / `Goals completion reason` | **散文** | 压：留首句 + 末句 |
| `executor` | 工具结果（`Tap ... was successful` 等） | **事实** | 逐字保留 |
| `plan_manager` | `Starting / NEXT SUBGOAL: ...` 等里程碑 | **事实** | 逐字保留 |

事实不能压，有两个硬理由：

1. `doc/perception-staleness-analysis.md` §7.2：归档分析的事实统计口径就是数
   `[executor]` 开头行。压了它，这条口径就废了。
2. `cortex.md` 硬规则 2 要求模型「不重复失败动作」—— 失败事实就是它判断的依据。

散文为什么留首句 + 末句
----------------------
- 首句 = 结论（「已完成子目标 X」「当前在第 3 个视频」）
- 末句 = 策略变更 / 完成理由（实测 `Goals completion reason` 多在末尾）
- 中间是铺陈（引用的视频内容、和画像的匹配过程），对下一轮决策价值最低

本模块只做**渲染**，不改 state
------------------------------
`state.agents_thoughts` 必须保持完整原文 —— 它有 4 个非 LLM 消费方：
`sdk/agent.py`（导出 trace / steps_count）、`tui/events.py`（实时渲染）、
`outputter`（最终输出，它要的是完整推理过程）、以及归档分析工具。
"""

from __future__ import annotations

import re
from typing import Literal

# 会被压缩的角色：只有它的内容是散文。
# executor / plan_manager 写的是**事实**，必须逐字保留。
COMPRESSIBLE_ROLES = frozenset({"cortex"})

# 句子切分：中英文标点都算句末。
# 保留标点本身（切完的句子仍带句号），这样拼回去读起来是完整的。
_SENTENCE_SPLIT = re.compile(r"(?<=[。.!?；;])\s*")

# 角色前缀：[cortex] / [executor] / [plan_manager] / [contextor]
_ROLE_PREFIX = re.compile(r"^\[([a-z_]+)\]\s*")

# 首尾句之间的连接符。用 " … " 而不是省略号，避免和原文里的省略号混淆。
_ELLIPSIS = " … "

# ---------------------------------------------------------------------------
# 受众：规划者（cortex / plan_manager）
# ---------------------------------------------------------------------------
#
# 为什么需要「受众」这个维度
# ------------------------
# thoughts 是所有 agent 共用的一条流水账。cortex / plan_manager 是**规划者**：
# 它们不下发具体动作，只决定「下一个子目标是什么」「上一个子目标完成没有」。
#
# 而 executor 写进 thoughts 的绝大多数是**逐步流水**：
#
#     [executor] Tap on element with coordinates (610, 300) was successful.
#     [executor] The screen is on the lock screen (AOD) showing time and date...
#     [executor] Text erased successfully. 0 characters were erased.
#
# 实测（2026-10-24，exec 861，Redmi Turbo 4 / MiniMax-M3.1-Flash）：
# executor 的行占 cortex thoughts 的 **60.9%**（40,599 / 66,657 字符，
# 208 / 414 行），而 `cortex.md` 的「What you are given」里**没有 thoughts
# 这一项** —— 执行结果走 Executor report，失败走两个失败块，走过的路线走
# report 的 actions taken。⇒ 对规划者而言，执行器的成功流水是**冗余**的。
#
# 为什么失败类必须留
# ------------------
# `cortex.md` 规则 4「Never build a goal around a failed target」——
# 两个失败块里只有**名字**（`"点赞" like_button`），**具体原因**
# （`no element with resource_id='...' on the current screen`）只在
# executor 的失败行里。失败行是判断「这条路走不通」的唯一具体依据。
#
# 位置无关（缓存安全）
# --------------------
# 去留判定**逐条独立**：只看这一条自己，不看前面有没有什么。
# ⇒ 已发出的字节永不改变 ⇒ 追加语义保留 ⇒ 前缀缓存不击穿。
# （与本模块「位置无关压缩」是同一个性质 —— 见文件头的说明。）
#
# ⚠️ 反例（位置相关，会击穿）：「只保留最近 3 条 executor 行」——
# 新条目进来时老条目的去留变了 ⇒ 老字节改变 ⇒ 断点前移。

#: 受众：**全量** —— 每条 thought 都给出去（归档口径与既有测试都靠它）。
FULL_AUDIENCE: Literal["full"] = "full"

#: 受众：规划者（cortex / plan_manager）。
PLANNER_AUDIENCE: Literal["planner"] = "planner"

#: 受众的取值集合。用 ``Literal`` 而不是裸 ``str``：传错一个值（比如
#: ``"planner "`` 带空格）应当**在类型检查时**就暴露，而不是静默落回
#: ``"full"`` 那种「多给了东西但没人发现」的行为。
#:
#: ⚠️ ``Literal`` 的成员必须是**字面量**，不能是变量 —— 所以这里写字符串本身，
#: 上面的两个常量则声明成 ``Literal[...]``，这样它们本身也能传给本类型。
Audience = Literal["full", "planner"]

# executor 的「失败/异常」特征。
#
# 宁可多留 —— 多留只是费 token，不会让模型判断错；漏留会让模型拿不到
# 「这条路为什么走不通」的具体原因。
_EXECUTOR_FAILURE_RE = re.compile(
    r"fail|error|not on the current screen|no element found|exception|"
    r"timeout|denied|refus|cannot|unable|invalid|not found|aborted",
    re.IGNORECASE,
)


def is_planner_relevant(thought: str) -> bool:
    """这条 thought 对规划者（cortex / plan_manager）有没有用？

    判定**只看这一条自己**（位置无关 ⇒ 缓存不击穿）。

    * `cortex` —— 自己的推理与决策：留。
    * `plan_manager` —— 子目标里程碑：留。
    * `executor` —— 逐步流水：默认**丢**；失败/异常类**留**（原因只在这里）。
    * 其它角色 / 没有角色前缀 —— **留**（不知道是什么，别砍）。
    """
    role, body = split_role(thought)
    if role != "executor":
        return True
    return bool(_EXECUTOR_FAILURE_RE.search(body))


def split_role(text: str) -> tuple[str, str]:
    """拆出角色和正文。

    返回 (role, body)。没有 `[role]` 前缀时 role 为 "unknown"。

    `state.py:164` 给每条 thought 加的是 `[cortex] ` 这种前缀，格式固定。
    """
    match = _ROLE_PREFIX.match(text)
    if not match:
        return "unknown", text.strip()
    return match.group(1), text[match.end() :].strip()


def split_sentences(text: str) -> list[str]:
    """把正文切成句子列表（保留句末标点）。

    中英文都处理。空段落不产出元素。
    """
    return [s for s in (part.strip() for part in _SENTENCE_SPLIT.split(text)) if s]


def compress_thought(thought: str) -> str:
    """压缩单条 thought。

    - 非散文角色（executor / plan_manager / 未知）→ **原样返回**
    - 散文（cortex）→ 只留首句 + 末句
    - 只有一句 → 原样保留
    - 角色前缀**只出现一次**（正文里已有的前缀不重复加）
    """
    role, body = split_role(thought)
    if role not in COMPRESSIBLE_ROLES:
        return thought

    sentences = split_sentences(body)
    if len(sentences) <= 1:
        return f"[{role}] {body}"

    if sentences[0] == sentences[-1]:
        # 首尾同句（正文只有一句但被切出多个片段）→ 不重复拼
        return f"[{role}] {sentences[0]}"

    return f"[{role}] {sentences[0]}{_ELLIPSIS}{sentences[-1]}"


def render_thoughts_history(
    thoughts: list[str],
    *,
    enabled: bool = True,
    audience: Audience = FULL_AUDIENCE,
) -> str:
    """渲染发给 LLM 的 thoughts 段（不含标题行）。

    `enabled=False` 时原样拼接 —— 这是回退开关，关掉就回到压缩前的行为。

    `audience` 控制**给谁看**：

    * `FULL_AUDIENCE`（默认）—— 全部保留，保持**逐条对应**：每条 thought 渲染成一行，
      归档统计的行首锚定口径（数 `[executor]` 开头行）因此不破。
    * `PLANNER_AUDIENCE` —— 给规划者（cortex / plan_manager）：丢掉 executor
      的**成功流水**，只留失败类。判定逐条独立 ⇒ 已发出的字节永不改变 ⇒
      前缀缓存不击穿。

    ⚠️ `audience=FULL_AUDIENCE` 的行为**一个字都没变** —— 归档口径和既有测试都靠它。
    """
    if not thoughts:
        return ""
    if audience == PLANNER_AUDIENCE:
        thoughts = [t for t in thoughts if is_planner_relevant(t)]
    if not enabled:
        return "\n".join(thoughts)
    return "\n".join(compress_thought(t) for t in thoughts)
