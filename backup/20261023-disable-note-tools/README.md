# 备份：临时停用「持久笔记（scratchpad）」三个工具

**日期**：2026-10-23
**原因**：临时把 Executor 面向 LLM 的笔记工具（`save_note` / `read_note` / `list_notes`）摘掉。
**性质**：**临时**。笔记的实现、state 字段、prompt 载荷豁免名单都**完全没动**。

> ⚠️ 本次备份的 `tools/index.py` 是**「宏工具已停用、笔记工具还在」**这个状态 ——
> 两次停用互相独立，可以单独恢复其中任意一次，不会互相覆盖。

## 备份的是什么

这次一共动了 4 个文件，原件在这里（目录结构与仓库一致）：

| 备份文件 | 对应仓库路径 | 改了什么 |
|---|---|---|
| `minitap/mobile_use/tools/index.py` | 同名 | 注释掉 3 个笔记 wrapper 的 import + `EXECUTOR_WRAPPERS_TOOLS` 里的 3 行注册 |
| `minitap/mobile_use/agents/executor/executor.md` | 同名 | 删掉工具表里 `save_note` / `read_note` 两行；`back`, `erase_one_char`, `list_notes` 那行改成 `back`, `erase_one_char` |
| `minitap/mobile_use/agents/executor/executor-zh.md` | 同名 | 同上（中文版） |
| `minitap/mobile_use/agents/planner/planner.md` | 同名 | 删掉整段 **Cross-app data transfer (using note tools)** 示例（109-118 行） |

## 怎么恢复

```powershell
$bk = 'E:\MyCoding\mobile-use\backup\20261023-disable-note-tools'
Copy-Item "$bk\minitap\mobile_use\tools\index.py"                        'E:\MyCoding\mobile-use\minitap\mobile_use\tools\index.py' -Force
Copy-Item "$bk\minitap\mobile_use\agents\executor\executor.md"           'E:\MyCoding\mobile-use\minitap\mobile_use\agents\executor\executor.md' -Force
Copy-Item "$bk\minitap\mobile_use\agents\executor\executor-zh.md"        'E:\MyCoding\mobile-use\minitap\mobile_use\agents\executor\executor-zh.md' -Force
Copy-Item "$bk\minitap\mobile_use\agents\planner\planner.md"             'E:\MyCoding\mobile-use\minitap\mobile_use\agents\planner\planner.md' -Force
```

恢复后 Executor 的工具数从 **14 回到 17**（开录屏 16 → 19）。

## ⚠️ 连带后果（停用期间的真实影响）

1. **planner 不再被教「用笔记做跨 app 数据搬运」** —— `planner.md` 那段示例已删。
   ⇒ 「把 A app 的内容搬到 B app」这类任务，planner 规划不出来（没有共同载体可搬运）。
2. **Executor 内部无法自己留便签** —— 原本用来「把整屏文本存下来、下一轮再读」的路子没了，
   只能靠上下文记忆。
3. **跨子目标的共享记忆没了** —— Planner / Cortex / Executor 原本共享同一个 `scratchpad`，
   现在没人能写了。

## 没动的东西（故意留着）

- `minitap/mobile_use/tools/scratchpad.py` —— 三个笔记工具的**实现**，一行没改。
  只是不再被 `tools/index.py` 注册，模型看不见也调不到；恢复注册即生效。
- `minitap/mobile_use/graph/state.py` 的 `scratchpad` 字段 —— 保留（`dict[str, str]` +
  `merge_dicts` reducer）。没人写它，但删了会让恢复变麻烦。
- `minitap/mobile_use/agents/executor/prompt.py` 的 `_PAYLOAD_TOOLS`
  （含 `read_note` / `list_notes`）—— **故意留着**。它是一个纯名字集合：
  - 工具没了也不影响任何功能（集合查找命中不了就正常走）；
  - 有 3 个测试直接依赖它（`test_executor_autonomy.py` 的 T23c、
    `test_structured_output_tool_name.py`），删了反而要改测试；
  - 留着 = 恢复工具时截断豁免自动生效，一行都不用动。
- `minitap/mobile_use/macros/compiler.py` —— 宏编译器本来就不录 scratchpad 那几个工具
  （第 138 行注释「scratchpad 那几个不录」），不受影响。

## 恢复后要跑的验证

```powershell
cd E:\MyCoding\mobile-use
.\.venv\Scripts\python.exe scripts\check_prompt_templates.py
.\.venv\Scripts\python.exe -m pytest tests\mobile_use -q
```
