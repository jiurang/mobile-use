# 备份：临时停用「宏 / 脚本库」三个工具

**日期**：2026-10-23
**原因**：临时把 Executor 面向 LLM 的宏工具（`find_macros` / `save_macro` / `run_macro`）摘掉。
**性质**：**临时**。宏引擎本体（`minitap/mobile_use/macros/`）**完全没动**，CLI、宏库、录制、回放都还在。

## 备份的是什么

这次一共动了 3 个文件，原件在这里（目录结构与仓库一致）：

| 备份文件 | 对应仓库路径 | 改了什么 |
|---|---|---|
| `minitap/mobile_use/tools/index.py` | `minitap/mobile_use/tools/index.py` | 注释掉 3 个宏 wrapper 的 import + `EXECUTOR_WRAPPERS_TOOLS` 里的 3 行注册 |
| `minitap/mobile_use/agents/executor/executor.md` | `minitap/mobile_use/agents/executor/executor.md` | 删掉工具表里宏的 3 行 |
| `minitap/mobile_use/agents/executor/executor-zh.md` | `minitap/mobile_use/agents/executor/executor-zh.md` | 删掉工具表里宏的 3 行 |

## 怎么恢复

```powershell
$bk = 'E:\MyCoding\mobile-use\backup\20261023-disable-macro-tools'
Copy-Item "$bk\minitap\mobile_use\tools\index.py"            'E:\MyCoding\mobile-use\minitap\mobile_use\tools\index.py' -Force
Copy-Item "$bk\minitap\mobile_use\agents\executor\executor.md"     'E:\MyCoding\mobile-use\minitap\mobile_use\agents\executor\executor.md' -Force
Copy-Item "$bk\minitap\mobile_use\agents\executor\executor-zh.md' 'E:\MyCoding\mobile-use\minitap\mobile_use\agents\executor\executor-zh.md' -Force
```

恢复后 Executor 的工具数从 **17 回到 20**（开录屏 22 → 24）。

## 恢复后要跑的验证

```powershell
cd E:\MyCoding\mobile-use
.\.venv\Scripts\python.exe scripts\check_prompt_templates.py
.\.venv\Scripts\python.exe -m pytest tests\mobile_use\test_macros.py tests\mobile_use\test_tool_schema_surgery.py -q
```

## 没动的东西（故意留着）

- `minitap/mobile_use/macros/**` —— 宏引擎本体（模型、编译器、存储、runner、CLI）。
- `minitap/mobile_use/tools/mobile/macros.py` —— 三个工具的**实现**。只是不再被 `tools/index.py` 注册，
  所以模型看不见也调不到；文件本身完好，恢复注册即生效。
- `tests/mobile_use/test_macros.py` —— 测的是宏引擎的纯逻辑，不经过注册表，全程绿。
- `scripts/compile_macro.py` / `scripts/compile_script_library.py` / `scripts/probe_macro_replay.py` —— 离线工具。
