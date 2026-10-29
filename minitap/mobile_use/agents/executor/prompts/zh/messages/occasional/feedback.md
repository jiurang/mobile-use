## 反馈

{%- if block == "dispatch_rejection" %}

你上一条动作**没有**被执行：{{ reason }}。请重新决定 —— 读工具表和当前的屏幕描述。
{%- elif block == "retry" %}

你上一条回答无法使用：{{ reason }}。你必须调用 ExecutorOutput，按 `## 输出` 里的字段作答；纯文本回答会被丢弃。
{%- elif block == "extra_actions" %}

你的 {{ count }} 个动作里只有第一个被执行了，其余的被丢弃，下一轮必须重新决定。
{%- elif block == "stall_notice" %}

[没有进展] {% if kind == "screen" %}最近 {{ window }} 步里，你回到了 {{ repeats }} 次**同一个页面**（屏幕指纹 {{ fingerprint }}）—— 页面上的东西没有变化。{% else %}最近 {{ window }} 步里，你重复了 {{ repeats }} 次**同一个动作**（{{ fingerprint }}）—— 结果没有变化。{% endif %}
先别重复刚才那个动作。想清楚下面两件事：
1. 这一条到底能不能做成？不能做就 `failed: true`（写清试过什么、为什么不行）—— 这是正常流程。
2. 能做的话，换一条**和刚才不同**的路走（换入口、换顺序、换页面），别原地再来一遍。
{% if vision_enabled %}
如果你还在同一个目标或同一块区域上打转，先调用 `scan_visual_subjects` 看一眼当前屏幕，再决定换哪条路。换一个 index 不算换路。{% endif %}
{%- endif %}
