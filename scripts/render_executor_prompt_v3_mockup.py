"""渲染「失败事实绑定到动作」的提示词样张（2026-10-24）。

干什么
------
给落地前的**措辞评审**用：把提议中的每个块用**真模板引擎**渲染出来，
让人直接读模型看到的那段文字 —— 不是手写示意，评审完的措辞可以直接搬进
`executor-zh.md` / `executor.md`。

产出
----
`doc/executor-prompt-v3-mockup.md`（布局图 + 宏原文 + 渲染样张 + 改动清单）

跑法::

    uv run python scripts/render_executor_prompt_v3_mockup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from jinja2 import Template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))
sys.stdout.reconfigure(encoding="utf-8")

OUT = ROOT / "doc" / "executor-prompt-v3-mockup.md"


# ---------------------------------------------------------------------------
# 提议中的新宏 / 改写后的宏（**候选措辞**，评审通过后再搬进 executor*.md）
# ---------------------------------------------------------------------------

ZH_MACROS = """
{%- macro step_failure_message(label, id_tail) -%}
[这一步失败] {{ label or id_tail }}{% if label and id_tail %} {{ id_tail }}{% endif %}
{%- endmacro -%}

{%- macro step_failure_message_same_prefix(label, id_tail) -%}
[上一步的结果] {{ label or id_tail }}{% if label and id_tail %} {{ id_tail }}{% endif %} 这一步失败了
{%- endmacro -%}

{%- macro step_failure_message_with_reason(label, id_tail, reason) -%}
[这一步失败] {{ label or id_tail }}{% if label and id_tail %} {{ id_tail }}{% endif %} —— 原因：{{ reason }}
{%- endmacro -%}

{%- macro failures_message(failures) -%}
这些目标在本任务里已经失败过 —— 不要再用它们，换一个不同的元素：{% for fact in failures %}{{ fact.desc }} ({{ fact.reason[:80] }}){% if not loop.last %}; {% endif %}{% endfor %}
{%- endmacro -%}
"""


def render_step_failure(label: str, id_tail: str) -> str:
    t = Template(ZH_MACROS)
    return t.module.step_failure_message(label=label, id_tail=id_tail).strip()


def render_step_failure_same_prefix(label: str, id_tail: str) -> str:
    t = Template(ZH_MACROS)
    return t.module.step_failure_message_same_prefix(label=label, id_tail=id_tail).strip()


def render_step_failure_with_reason(label: str, id_tail: str, reason: str) -> str:
    t = Template(ZH_MACROS)
    return t.module.step_failure_message_with_reason(
        label=label, id_tail=id_tail, reason=reason
    ).strip()


# ---------------------------------------------------------------------------
# 样张素材（真机味道：一次点「点赞」失败、随后换成「关注」）
# ---------------------------------------------------------------------------

HISTORY = [
    {
        "role": "assistant",
        "tool_calls": {
            "name": "ExecutorOutput",
            "args": {
                "actions": [{"action": "tap", "target": {"index": 12}}],
                "agent_thought": "第 1 步：先点顶部的「WLAN」进去。",
            },
        },
    },
    {"role": "tool", "content": "Tapped element (610, 300) successfully."},
    {
        "role": "user",
        "content": "[上一步的结果] 前台应用变成了 com.android.settings；屏幕上新出现了：WLAN、移动网络、蓝牙",
    },
    {
        "role": "assistant",
        "tool_calls": {
            "name": "ExecutorOutput",
            "args": {
                "actions": [{"action": "tap", "target": {"index": 48}}],
                "agent_thought": "第 2 步：点「点赞」按钮。",
            },
        },
    },
    {
        "role": "tool",
        "content": "Tap failed: no element with resource_id='com.xingin.app:id/like_button' on the current screen",
    },
    {
        "role": "user",
        "content": "[这一步失败] \"点赞\" like_button",
    },
    {
        "role": "user",
        "content": "[上一步的结果] 前台应用没变（还是 com.android.settings）；屏幕上新出现了：更多设置",
    },
]

CROSS_SUBGOAL_FAILURES = [
    {"desc": '"点赞" like_button', "reason": "no element with resource_id on the current screen"},
    {"desc": '"关注" follow_btn', "reason": "输入框聚焦失败，输入没有落进去"},
]

DEVICE_STATE = "设备日期：2026-10-24 10:05:33\n前台应用：com.android.settings"

SCREEN_TABLE = """屏幕元素，每行一条：[i] "text" | id=<id 尾段> | Class | (x,y,WxH) | flags
flags：C=可点击 L=可长按 F=可聚焦 K=复选框 S=已选 R=可滚动
所有行共用一个编号。没有 id= 的行是 OCR 文本。id= 是 resource-id 的尾段（省略包名），
只在文字缺失或重复时给出。
[1] "WLAN" TextView (60, 480, 320x120) C
[2] "移动网络" TextView (60, 640, 320x120) C
[3] "蓝牙" TextView (60, 800, 320x120) C
[4] "更多设置" TextView (60, 960, 320x120) C
[5] "WLAN" EditText (60, 420, 1080x160) F
[6] "已保存的网络" TextView (60, 300, 400x90)"""


def render_turn() -> str:
    """渲染一整轮（提议版）：system / 目标 / 历史 / 失败索引 / 设备状态 / 元素表。"""
    out: list[str] = []
    out.append("【1】system —— 不变（executor-zh.md 的 system_message 宏全文，1,203 tok）")
    out.append("      （略，见 minitap/mobile_use/agents/executor/executor-zh.md）")
    out.append("")
    out.append("【2】user —— 目标（不变）")
    out.append("你的目标：")
    out.append("进入 WLAN 页面并读出当前连接的 Wi-Fi 名称")
    out.append("")
    out.append("【3】局部历史（只追加）—— 唯一的新东西：失败事实跟着动作走")
    for m in HISTORY:
        if m["role"] == "assistant":
            tc = m["tool_calls"]
            out.append(
                f"assistant  ExecutorOutput(actions={tc['args']['actions']}, "
                f"agent_thought={tc['args']['agent_thought']!r})"
            )
        else:
            out.append(f"{m['role']}       {m['content']}")
    out.append("")
    out.append("【4】user —— 跨子目标失败索引（位置不变，措辞收紧：去掉裸 index）")
    out.append(
        Template(ZH_MACROS)
        .module.failures_message(failures=CROSS_SUBGOAL_FAILURES)
        .strip()
    )
    out.append("")
    out.append("【5】user —— 设备状态（不变）")
    out.append(DEVICE_STATE)
    out.append("")
    out.append("【6】user —— 屏幕元素表（不变）")
    out.append(SCREEN_TABLE)
    return "\n".join(out)


def main() -> int:
    body = f"""# Executor 提示词：失败事实绑定到动作（措辞样张 / 已评审）

> 日期：2026-10-24　状态：**已评审通过并落地**（2026-10-24）
> 依据：`doc/executor-prompt-composition-and-cache-20261024.md`（组成与命中率的实测）
> 渲染脚本：`scripts/render_executor_prompt_v3_mockup.py`（真模板引擎）
>
> ⚠️ **本文是「评审时的提案」，不是最终形态。** 落地后的实际样子用
> `scripts/render_executor_prompt_actual.py` 渲染（走**生产代码**）。
> 三处与提案不同：
> 1. 失败事实**并进了 `[上一步的结果]` 那一行**（不是单开一条）——
>    评审结论 2；
> 2. 跨子目标索引改成**中性说法** + **去掉裸 index** —— 评审结论 3；
> 3. 失败行**不带结论**（F1）—— 评审结论 1。

## 0. 这一版改了什么

| # | 改动 | 目的 |
|---|---|---|
| 1 | **失败事实绑定到动作**：工具层记失败事实的**同一个 update** 里，往局部历史追加一行 | 失败事实从「尾巴里每轮全价重发」变成「历史里只付一次」；而且结果贴着动作，模型不用跨块对账 |
| 2 | **跨子目标失败索引去掉裸 index** | 索引每轮重编号，留着 `[48]` 会让模型以为下一屏的 48 号还是它 |
| 3 | **设备状态不动** | 「前台 app 变了」这类**事件**已经在 `[上一步的结果]` 里；「当前前台 app / 当前时间」是**状态**，本来就该留在尾部 |
| 4 | **元素表不动** | 它是**当前屏**，追加就变成堆旧屏幕 |
| 5 | **不加条数上限** | 索引只压短不丢弃；上限会让「事实少了一条」变成静默的错误 |

## 0.5 现有提示词里已有的规矩（这一版必须遵守）

| 位置 | 原文 | 含义 |
|---|---|---|
| `executor-zh.md:69` | 「**先说上一步的结果（对照你要的东西），再说本动作的原因**」 | 模型侧：一次思考 = 上一轮结果 + 本轮理由，**两样，别的一律不要** |
| `executor-zh.md:73` | 「**你的历史里有「上一步的结果」**（框架记的**客观事实**：前台 app、屏幕上出现了什么）」 | 框架侧：历史里放的是**客观事实** |
| `contextor.py:90` | 「⚠️ **这里只陈述事实，不下结论。**「这个动作没用」是**相对于目标**的判断，框架不知道目标。」 | 框架侧：**不下结论**，判定留给模型 |

⇒ **所以失败行里不能写「不要再用这个目标」** —— 那是结论。
框架只负责把事实搬进历史：**「这一步失败了，目标是谁」**。
「换不换目标」是模型看着自己的目标做的判断。

「不要再用」这句指令**只保留在跨子目标索引里**（那里本来就有，现状措辞），
不重复进每一步。

## 1. 新增块的候选措辞（中文）

**候选 F1（推荐）** —— 纯事实，不带结论、不重复原因：

```jinja
{{%- macro step_failure_message(label, id_tail) -%}}
[这一步失败] {{{{ label or id_tail }}}}{{% if label and id_tail %}} {{{{ id_tail }}}}{{% endif %}}
{{%- endmacro -%}}
```

```
{render_step_failure("点赞", "like_button")}
```

```
{render_step_failure("", "download_btn")}
```

> 补的是工具回执**没有**的东西：**元素的可读名字**（回执里只有 resource-id）。
> 原因不用重复 —— 工具回执就在上一行，且错误结果**逐字保留**（不截断）。

**候选 F2** —— 原因也带上（更长，每轮多付 ~40 token）：

```
{render_step_failure_with_reason("点赞", "like_button", "no element with resource_id on the current screen")}
```

**候选 F3** —— 沿用「上一步的结果」这个前缀，两条事实行一个口径：

```
{render_step_failure_same_prefix("点赞", "like_button")}
```

> ⚠️ **已经否掉的两个**：
> - 「… —— 不要再用这个目标」= **结论**，违反 §0.5 的规矩；
> - 「… 已经不在屏幕上了」= **结论，而且可能是错的** —— 失败不总是「元素没了」，
>   也可能是「输入框聚焦失败」，那时元素明明还在。

## 2. 跨子目标失败索引（改措辞，位置不动）

```jinja
{{%- macro failures_message(failures) -%}}
这些目标在本任务里已经失败过 —— 不要再用它们，换一个不同的元素：{{% for fact in failures %}}{{{{ fact.desc }}}} ({{{{ fact.reason[:80] }}}}){{% if not loop.last %}}; {{% endif %}}{{% endfor %}}
{{%- endmacro -%}}
```

改动只有一处：`desc` 从 `[48] "点赞" like_button` 变成 `"点赞" like_button`（**去 index**）。
渲染出来：

```
{Template(ZH_MACROS).module.failures_message(failures=CROSS_SUBGOAL_FAILURES).strip()}
```

## 3. 一整轮的完整样张（模型实际看到的东西）

```
{render_turn()}
```

## 4. 要评审的几个点

1. **失败行的措辞**：选 F1 / F2 / F3？（F1 推荐 —— 纯事实，不下结论）
2. **失败行单独一条 user 消息，还是并进 Contextor 的 `[上一步的结果]` 那条**？
   现在是两条（工具层写一条、Contextor 写一条，产生时间不同）。
   并进同一条要新增一个「上一步的失败」瞬时字段给 Contextor 读 —— 多一个 state 字段。
3. **跨子目标索引里同一目标的重复条目** —— 现状会重复列（每条失败事件一条）。
   合并同类项要改 `append_failure` 的语义（现在「每条失败事件一条」），
   这属于行为变更，要单独确认。
4. **英文版 `executor.md` 同步改**，且必须同一个意思。
5. **失败行的标签来自哪**：工具层拿到的是 `target.text` 和 `target.resource_id`，
   `text` 为空时（纯图标按钮）退回 id 尾段 —— 上面第二条样例就是这个情况。
6. **`describe()` 要不要一起去掉 index**：它同时被 **Executor 的失败索引** 和
   **Reporter 的 failed_targets**（事实层、逐字搬运）用。改它会连 Reporter 的
   汇报格式一起变 —— 那条契约写着「逐字」。所以要么只给 Executor 另写一个
   渲染器（Reporter 不动），要么两处一起改并同步改契约。**这一条影响面最大。**
"""
    OUT.write_text(body, encoding="utf-8")
    print(f"已写出 {OUT}")
    print()
    print(render_turn())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
