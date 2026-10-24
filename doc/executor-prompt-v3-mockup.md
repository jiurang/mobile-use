# Executor 提示词：失败事实绑定到动作（措辞样张 / 待评审）

> 日期：2026-10-24　状态：**待评审**（还没改生产代码）
> 依据：`doc/executor-prompt-composition-and-cache-20261024.md`（组成与命中率的实测）
> 渲染脚本：`scripts/render_executor_prompt_v3_mockup.py`（真模板引擎，评审完可直接搬）

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
{%- macro step_failure_message(label, id_tail) -%}
[这一步失败] {{ label or id_tail }}{% if label and id_tail %} {{ id_tail }}{% endif %}
{%- endmacro -%}
```

```
[这一步失败] 点赞 like_button
```

```
[这一步失败] download_btn
```

> 补的是工具回执**没有**的东西：**元素的可读名字**（回执里只有 resource-id）。
> 原因不用重复 —— 工具回执就在上一行，且错误结果**逐字保留**（不截断）。

**候选 F2** —— 原因也带上（更长，每轮多付 ~40 token）：

```
[这一步失败] 点赞 like_button —— 原因：no element with resource_id on the current screen
```

**候选 F3** —— 沿用「上一步的结果」这个前缀，两条事实行一个口径：

```
[上一步的结果] 点赞 like_button 这一步失败了
```

> ⚠️ **已经否掉的两个**：
> - 「… —— 不要再用这个目标」= **结论**，违反 §0.5 的规矩；
> - 「… 已经不在屏幕上了」= **结论，而且可能是错的** —— 失败不总是「元素没了」，
>   也可能是「输入框聚焦失败」，那时元素明明还在。

## 2. 跨子目标失败索引（改措辞，位置不动）

```jinja
{%- macro failures_message(failures) -%}
这些目标在本任务里已经失败过 —— 不要再用它们，换一个不同的元素：{% for fact in failures %}{{ fact.desc }} ({{ fact.reason[:80] }}){% if not loop.last %}; {% endif %}{% endfor %}
{%- endmacro -%}
```

改动只有一处：`desc` 从 `[48] "点赞" like_button` 变成 `"点赞" like_button`（**去 index**）。
渲染出来：

```
这些目标在本任务里已经失败过 —— 不要再用它们，换一个不同的元素："点赞" like_button (no element with resource_id on the current screen); "关注" follow_btn (输入框聚焦失败，输入没有落进去)
```

## 3. 一整轮的完整样张（模型实际看到的东西）

```
【1】system —— 不变（executor-zh.md 的 system_message 宏全文，1,203 tok）
      （略，见 minitap/mobile_use/agents/executor/executor-zh.md）

【2】user —— 目标（不变）
你的目标：
进入 WLAN 页面并读出当前连接的 Wi-Fi 名称

【3】局部历史（只追加）—— 唯一的新东西：失败事实跟着动作走
assistant  ExecutorOutput(actions=[{'action': 'tap', 'target': {'index': 12}}], agent_thought='第 1 步：先点顶部的「WLAN」进去。')
tool       Tapped element (610, 300) successfully.
user       [上一步的结果] 前台应用变成了 com.android.settings；屏幕上新出现了：WLAN、移动网络、蓝牙
assistant  ExecutorOutput(actions=[{'action': 'tap', 'target': {'index': 48}}], agent_thought='第 2 步：点「点赞」按钮。')
tool       Tap failed: no element with resource_id='com.xingin.app:id/like_button' on the current screen
user       [这一步失败] "点赞" like_button
user       [上一步的结果] 前台应用没变（还是 com.android.settings）；屏幕上新出现了：更多设置

【4】user —— 跨子目标失败索引（位置不变，措辞收紧：去掉裸 index）
这些目标在本任务里已经失败过 —— 不要再用它们，换一个不同的元素："点赞" like_button (no element with resource_id on the current screen); "关注" follow_btn (输入框聚焦失败，输入没有落进去)

【5】user —— 设备状态（不变）
设备日期：2026-10-24 10:05:33
前台应用：com.android.settings

【6】user —— 屏幕元素表（不变）
屏幕元素，每行一条：[i] "text" | id=<id 尾段> | Class | (x,y,WxH) | flags
flags：C=可点击 L=可长按 F=可聚焦 K=复选框 S=已选 R=可滚动
所有行共用一个编号。没有 id= 的行是 OCR 文本。id= 是 resource-id 的尾段（省略包名），
只在文字缺失或重复时给出。
[1] "WLAN" TextView (60, 480, 320x120) C
[2] "移动网络" TextView (60, 640, 320x120) C
[3] "蓝牙" TextView (60, 800, 320x120) C
[4] "更多设置" TextView (60, 960, 320x120) C
[5] "WLAN" EditText (60, 420, 1080x160) F
[6] "已保存的网络" TextView (60, 300, 400x90)
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
